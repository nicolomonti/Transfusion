# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from enum import Enum

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.nn.utils.rnn import pad_sequence

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from torchvision.utils import make_grid
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    cross_entropy,
)

import torchvision.utils as vutils

import matplotlib.pyplot as plt


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class TransfusionArgs(BaseTransformerArgs):
    seed: int = 42

    vocab_size: int = 14
    weight_tying: bool = False

    sliding_window: Optional[int] = None

    n_heads: int = 4

    max_seq_len: int = 32

    padding_token: int = -1
    image_channels_num: int = 3
    image_size: int = 128
    image_encoder_dim: int = 256


class ModalityType(Enum):
    TEXT = 0
    IMAGE = 1


class Modality:
    def __init__(self, type: ModalityType, data: torch.Tensor):
        self.type: ModalityType = type
        self.data: torch.Tensor = data


def get_timestep_embedding(timesteps, embedding_dim): # From Stable Diffusion 3's repo
    '''
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of 'Attention Is All You Need'.
    '''

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1: # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    emb = emb.to(timesteps.dtype)

    return emb


class ImageEncoder(nn.Module):
    def __init__(self, args: TransfusionArgs):
        super().__init__()

        self.args: TransfusionArgs = args
        
        self.height = self.width = args.image_size
        
        self.out_tokens_num = 8

        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.args.image_channels_num,
                64,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.GELU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.GELU(),
            nn.Conv2d(
                128,
                self.out_tokens_num,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_output_dim = (args.image_size // 4) * (args.image_size // 4)
        
        self.projection = nn.Linear(self.encoder_output_dim, args.dim)


    def forward(self, x):
        assert x.ndim == 4, 'The image must be batched'

        batch_size = x.shape[0]

        x = self.encoder(x)
        x = x.reshape((batch_size, self.out_tokens_num, self.encoder_output_dim))
        
        x = self.projection(x)

        return x


class ImageDecoder(nn.Module):
    def __init__(self, args: TransfusionArgs):
        super().__init__()

        self.args: TransfusionArgs = args
        
        self.height = self.width = args.image_size
        
        self.projection = nn.Linear(args.dim, 4) # (4, 32, 32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                4,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding='same'
            ),
            nn.GELU(),


            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                128,
                3,
                kernel_size=3,
                stride=1,
                padding='same'
            ),

            nn.Tanh()
        )


    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.reshape((-1, 4, 32, 32))
        x = self.decoder(x)

        return x


class Transfusion(BaseTransformer):
    def __init__(self, args: TransfusionArgs, image_encoder, image_decoder):
        super().__init__(args)

        self.args: TransfusionArgs = args

        self.weight_tying = self.args.weight_tying
        self.sliding_window = self.args.sliding_window

        assert self.args.vocab_size > 0

        self.wte = torch.nn.Embedding(self.args.vocab_size, self.args.dim)

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

        if self.args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        self.image_encoder = image_encoder
        self.image_decoder = image_decoder

        self.fc = nn.Linear(self.args.dim, self.args.vocab_size)


    @property
    def dtype(self):
        return next(self.parameters()).dtype


    @property
    def device(self):
        return next(self.parameters()).device


    def forward(
        self,
        x: List[Modality],
        times: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        attn_impl: str = 'sdpa',
        compute_loss: bool = False,
        encode_modality: bool = True
    ):
        if compute_loss:
            self.train()

        batch_size = len(x)

        max_sequence_length = self.args.max_seq_len

        full_batch_data = torch.zeros((batch_size, max_sequence_length, self.args.dim), dtype=self.dtype, device=self.device)
        full_batch_target_data = torch.zeros((batch_size, max_sequence_length, self.args.dim), dtype=self.dtype, device=self.device)

        attention_mask = torch.ones((batch_size, 1, max_sequence_length, max_sequence_length), dtype=self.dtype, device=self.device).tril()
        image_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool, device=self.device)

        text_list = []
        image_targets_list = []

        total_text_tokens_num = 0
        total_image_tokens_num = 0

        if times is None:
            times = torch.rand((batch_size, 1), dtype=self.dtype, device=self.device)
        
        else:
            times = times.to(self.dtype)

        for idx, batch in enumerate(x):
            offset = 0

            current_text_list = []

            for modality in batch: # Tokens must be 1d
                if modality.type == ModalityType.TEXT:
                    tokens = modality.data

                    seq_len = tokens.shape[0]
                    total_text_tokens_num += seq_len

                    text_embeddings = self.wte(tokens)
                    full_batch_data[idx, offset:(offset + seq_len), :] = text_embeddings

                    offset += seq_len

                    current_text_list.append(tokens)

                elif modality.type == ModalityType.IMAGE:
                    embeddings = modality.data.to(self.dtype)

                    if encode_modality:
                        if embeddings.ndim == 3:
                            embeddings = embeddings.unsqueeze(dim=0)

                        embeddings = self.image_encoder(embeddings).view((-1, self.args.dim))

                    noise = torch.randn_like(embeddings)
                    noisy_embeddings = (times[idx] * embeddings) + ((1.0 - times[idx]) * noise)

                    targets = embeddings - noise

                    seq_len = noisy_embeddings.shape[0]
                    total_image_tokens_num += seq_len

                    conditioning_embeddings = get_timestep_embedding(
                        torch.tensor([times[idx] for _ in range(seq_len)], dtype=self.dtype, device=self.device),
                        self.args.dim
                    )

                    noisy_embeddings += conditioning_embeddings

                    full_batch_data[idx, offset:(offset + seq_len), :] = noisy_embeddings
                    full_batch_target_data[idx, offset:(offset + seq_len), :] = targets

                    image_mask[idx, offset:(offset + seq_len)] = True
                    attention_mask[idx, 0, offset:(offset + seq_len), offset:(offset + seq_len)] = 1.0

                    offset += seq_len

                    current_text_list.append(torch.full((seq_len,), self.args.padding_token, dtype=torch.int64, device=self.device))
                    image_targets_list.append(targets)

            attention_mask = attention_mask.masked_fill(attention_mask == 0.0, float('-inf'))

            text_list.append(
                torch.cat(current_text_list, dim=0) if len(current_text_list) > 0 else torch.empty(self.args.max_seq_len)
            )

        padded_text_list = pad_sequence(
            text_list + [torch.empty(self.args.max_seq_len)],

            batch_first=True,
            padding_value=self.args.padding_token
        )[:-1]

        # plt.imshow(attention_mask[0, 0].detach().int().cpu().numpy())
        # plt.show()


        h = super().forward(full_batch_data, tok_idx=tok_idx, mask=attention_mask, attn_impl=attn_impl)

        # Text

        text_logits = self.fc(h)

        # Images

        image_embeddings = torch.cat(
            [
                h[current_image_mask_idx][current_image_mask]
                for current_image_mask_idx, current_image_mask in enumerate(image_mask)
            ],
            dim=0
        )

        image_embeddings = image_embeddings.reshape((-1, 8, self.args.dim))
        
        image_drift_preds = image_embeddings
        
        if encode_modality:
            image_drift_preds = self.image_decoder(image_embeddings)

        image_targets = full_batch_target_data[image_mask].reshape((-1, self.args.dim))

        if not compute_loss:
            return text_logits, image_drift_preds

        text_logits = text_logits[:, :-1] # [~(image_mask[:, :-1])]
        text_targets = padded_text_list[:, 1:] # [~(image_mask[:, 1:])] # [:, 1:].flatten() # This is wrong! You must predict the next text token from the last image one

        loss_text = 0.0
        loss_images = 0.0

        if total_text_tokens_num > 0:
            loss_text = F.cross_entropy(
                text_logits.reshape((-1, self.args.vocab_size)),
                text_targets.flatten(),
                
                ignore_index=self.args.padding_token
        )

        if total_image_tokens_num > 0:
            loss_images = F.mse_loss(image_embeddings.view((-1, self.args.dim)), image_targets)

        loss = loss_text + loss_images

        return loss


    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: TransfusionArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: TransfusionArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
