import logging

import os
import argparse

from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, field

from pathlib import Path

import copy

import torch

import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import torchvision.utils as vutils
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
from torch.amp import GradScaler, autocast

from torch.distributed.checkpoint.stateful import Stateful

import torchvision

import torchvision.transforms as transforms

from torchdiffeq import odeint

from accelerate import Accelerator

import wandb

from tqdm import tqdm

from lingua.optim import OptimArgs, build_optimizer

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint

from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)

from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)

from lingua.profiling import ProfilerArgs, maybe_run_profiler

from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)

from apps.main.transfusion import Transfusion, TransfusionArgs, Modality, ModalityType
from apps.main.transfusion import ImageEncoder, ImageDecoder


logger = logging.getLogger()


@dataclass
class TrainArgs:
    name: str = "lingua_transfusion"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    device: torch.device = torch.device('cuda')

    # Nb optimizer steps to take
    steps: int = 1_000_000

    eval_every: int = 2048
    sample_image_every: int = 1024

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: TransfusionArgs = field(default_factory=TransfusionArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {args.model.vocab_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {args.checkpoint.path}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


@torch.no_grad()
def sample_image(model, device, class_idx):
    text_sequence = create_class_idx_sequence(class_idx, model.args)
    text_modality = Modality(ModalityType.TEXT, text_sequence.to(device))

    image_noise = torch.randn((8, model.args.dim), device=device)
    image_modality = Modality(ModalityType.IMAGE, image_noise)

    input_sequence = [[text_modality, image_modality]]

    def drift_function(t, state):
        input_sequence[0][1].data = state
        times = torch.tensor([[t]], device=device)
        _, drift_pred = model(input_sequence, times=times, encode_modality=False)
        
        return drift_pred.view_as(state)


    initial_state = input_sequence[0][1].data

    t = torch.tensor([0.0, 1.0], device=device)

    solved_states = odeint(drift_function, initial_state, t, method='midpoint')

    final_state = solved_states[-1]

    image = model.image_decoder(final_state)
    image = image.reshape((3, model.args.image_size, model.args.image_size))

    return image


def create_class_idx_sequence(class_idx, config):
    start_token = config.vocab_size - 4
    end_token = config.vocab_size - 3

    return torch.tensor([start_token, class_idx, end_token])


def train_autoencoder_epoch(image_encoder, image_decoder, train_dataloader, optimizer, scaler, device, epoch, max_steps=None):
    image_encoder.train()
    image_decoder.train()

    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (image, _) in enumerate(progress_bar):
        if max_steps is not None and batch_idx >= max_steps:
            break
        
        optimizer.zero_grad()

        image = image.to(device)

        batch_size = image.shape[0]
        batch_sequences = []

        with autocast(device_type=device):
            # print(image.shape)

            encoded = image_encoder(image)
            encoded = torch.lerp(encoded, torch.rand_like(encoded), 0.2)
            decoded = image_decoder(encoded)

            loss = F.mse_loss(decoded, image)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            encoded = image_encoder(image[:1])
            decoded = image_decoder(encoded)

            grid = vutils.make_grid([image[0], decoded[0]], normalize=True, nrow=2)
            vutils.save_image(grid, 'tmp.png')

            print('Saved')

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
    return total_loss / len(train_dataloader)


def train_transfusion(model: Transfusion, train_dataloader: DataLoader, evaluation_dataloader: DataLoader, train_state: TrainState, optimizer, scaler, scheduler, train_args: TrainArgs):
    model.train()

    total_loss = 0

    accelerator = Accelerator()

    model, optimizer, train_dataloader, train_state.scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, train_state.scheduler
    )

    _train_dataloader = iter(copy.deepcopy(train_dataloader))

    progress_bar = tqdm(total=train_args.steps, desc=f'Training')

    while train_state.step < train_args.steps:
        try:
            image, text = next(_train_dataloader)

        except StopIteration:
            _train_dataloader = iter(copy.deepcopy(train_dataloader))

            image, text = next(_train_dataloader)

        image = image.to(train_args.device)
        text = text.to(train_args.device)

        batch_size = image.shape[0]

        batch_sequences = []
        
        for idx in range(batch_size):
            start_token = model.args.vocab_size - 4
            end_token = model.args.vocab_size - 3

            text_sequence = torch.tensor([start_token, text[idx], end_token], device=train_args.device)

            image_modality = Modality(ModalityType.IMAGE, image[idx].unsqueeze(dim=0))
            text_modality = Modality(ModalityType.TEXT, text_sequence)

            batch_sequences.append([image_modality, text_modality])
            batch_sequences.append([text_modality, image_modality])

        with autocast(device_type=str(train_args.device)):
            loss = model(batch_sequences, times=None, compute_loss=True)

        optimizer.zero_grad()

        accelerator.backward(scaler.scale(loss)) # .backward()
        scaler.step(optimizer)
        train_state.scheduler.step()
        scaler.update()

        train_state.step += 1
        total_loss += loss.item()

        progress_bar.update()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (train_state.step + 1):.4f}'
        })
        
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/avg_loss': total_loss / (train_state.step + 1)
        })

        if (train_state.step + 1) % train_args.eval_every == 0:
            evaluate_transfusion(
                model,
                
                evaluation_dataloader,
                
                train_args.device,
                train_state
            )

        if (train_state.step + 1) % train_args.sample_image_every == 0:
            model.eval()

            samples = []

            for class_idx in range(10):
                samples.append(sample_image(model, train_args.device, class_idx))

            # print(samples[-1].shape)

            grid = vutils.make_grid(samples, normalize=True, nrow=5)

            wandb.log({
                'samples': wandb.Image(grid, caption=f'Batch {train_state.step}')
            })

            model.train()
    
    return total_loss / len(train_dataloader)


@torch.no_grad()
def evaluate_transfusion(model: Transfusion, evaluation_dataloader: DataLoader, device: torch.device, train_state: TrainState):
    model.eval()

    total_loss = 0

    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(evaluation_dataloader, desc=f'Evaluation batch {train_state.step}')
    
    for batch_idx, (image, text) in enumerate(progress_bar):
        image = image.to(device)
        text = text.to(device)

        batch_size = image.shape[0]
        batch_sequences = []
        
        for idx in range(batch_size):
            start_token = model.args.vocab_size - 4
            end_token = model.args.vocab_size - 3

            text_sequence = torch.tensor([start_token, text[idx], end_token], device=device)

            image_modality = Modality(ModalityType.IMAGE, image[idx].unsqueeze(dim=0))
            text_modality = Modality(ModalityType.TEXT, text_sequence)

            batch_sequences.append([image_modality, text_modality])

        loss = model(batch_sequences, compute_loss=True)
        text_logits, _ = model(batch_sequences, compute_loss=False)
        
        predictions = text_logits[:, 8].argmax(dim=-1)
        targets = text

        correct = (predictions == targets).sum().item()

        total_correct += correct
        total_samples += batch_size
        
        total_loss += loss.item()
        current_accuracy = total_correct / total_samples
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'accuracy': f'{current_accuracy:.4f}'
        })
        
        wandb.log({
            'val/batch_loss': loss.item(),
            'val/avg_loss': total_loss / (batch_idx + 1),
            'val/accuracy': current_accuracy
        })
    
    samples = []

    model.eval()

    for class_idx in range(10):
        samples.append(sample_image(model, device, class_idx))

    model.train()

    grid = vutils.make_grid(samples, normalize=True, nrow=5)

    wandb.log({
        'val_samples': wandb.Image(grid, caption=f'Validation Batch {train_state.step}'),
        'val/epoch_accuracy': total_correct / total_samples,
        'val/epoch_loss': total_loss / len(evaluation_dataloader)
    })
    
    return total_loss / len(evaluation_dataloader), total_correct / total_samples


def main():
    train_args = TrainArgs()

    train_args.model = TransfusionArgs(
        vocab_size=14,
        dim=512,
        n_layers=12,
        n_heads=8,
        max_seq_len=32,
        image_channels_num=3,
        image_size=128
    )

    train_args.data.root_dir = './data'
    train_args.checkpoint.path = 'checkpoints'

    transform = transforms.Compose([
        transforms.Resize((train_args.model.image_size, train_args.model.image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=train_args.data.root_dir,
        train=True,
        transform=transform,
        download=True
    )

    val_dataset = torchvision.datasets.MNIST(
        root=train_args.data.root_dir,
        train=False,
        transform=transform,
        download=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_args.data.batch_size,

        shuffle=True
    )

    evaluation_dataloader = DataLoader(
        val_dataset,
        batch_size=train_args.data.batch_size,

        shuffle=True
    )

    image_encoder = ImageEncoder(train_args.model).to(train_args.device)
    image_decoder = ImageDecoder(train_args.model).to(train_args.device)

    autoencoder_optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(image_decoder.parameters()), lr=1e-4)

    scaler = GradScaler()

    train_autoencoder = False

    if train_autoencoder:
        for epoch_idx in range(1):
            train_autoencoder_epoch(image_encoder, image_decoder, train_dataloader, autoencoder_optimizer, scaler, train_args.model.device, epoch_idx, max_steps=None)

        torch.save(image_encoder.state_dict(), 'image_encoder.pt')
        torch.save(image_decoder.state_dict(), 'image_decoder.pt')

        exit()

    image_encoder.load_state_dict(torch.load('image_encoder.pt'))
    image_decoder.load_state_dict(torch.load('image_decoder.pt'))

    for parameter in list(image_encoder.parameters()) + list(image_decoder.parameters()):
        parameter.requires_grad = False

    wandb.init(project=train_args.name)

    os.makedirs(train_args.checkpoint.path, exist_ok=True)
        
    model = Transfusion(train_args.model, image_encoder, image_decoder).to(train_args.device)
    wandb.watch(model)

    optimizer, scheduler = build_optimizer(model, train_args.optim, train_args.steps)
    
    train_state = TrainState(
        step=0,
        acc_step=0,
        scheduler=scheduler
    )
    
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    train_transfusion(
        model,
        
        train_dataloader,
        evaluation_dataloader,

        train_state,
        
        optimizer,
        scaler,
        scheduler,

        train_args
    )

    # val_loss, val_accuracy = evaluate(model, evaluation_dataloader, args.device, epoch)


if __name__ == '__main__':
    main()
