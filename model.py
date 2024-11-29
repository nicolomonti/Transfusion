from dataclasses import dataclass
from typing import List
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import wandb

from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')


@dataclass
class TransfusionConfig:
    embed_dim: int = 256
    ff_dim: int = 512

    head_dim: int = None

    num_heads: int = 8

    layers_num: int = 6 
    image_token_dim: int = 28
    
    block_size: int = 28 + 3 # 28 image tokens, 3 caption tokens
    vocab_size: int = 14

    padding_token: int = -1

    image_channels_num: int = 1
    image_encoder_dim: int = 128


    def __post_init__(self):
        self.head_dim: int = self.embed_dim // self.num_heads


class ImageEncoder(nn.Module):
    def __init__(self, config: TransfusionConfig):
        super().__init__()

        self.config: TransfusionConfig = config

        self.encoder = nn.Sequential(
            nn.Conv2d(self.config.image_channels_num, self.config.embed_dim, kernel_size=3, stride=2, padding='same')
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.view((-1, self.config.embed_dim))

        return x


class ImageDecoder(nn.Module):
    def __init__(self, config: TransfusionConfig):
        super().__init__()
        
        self.config: TransfusionConfig = config

        self.decoder = nn.ConvTranspose2d(self.config.embed_dim, self.config.image_channels_num, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.decoder(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout_p=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.dropout_p = dropout_p

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout_p)

        self.scale = self.head_dim ** -0.5


    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=1).expand((-1, self.num_heads, -1, -1))
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class MLP(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc_0 = nn.Linear(dim, ff_dim)
        self.activation = nn.GELU()
        self.fc_1 = nn.Linear(ff_dim, dim)


    def forward(self, x):
        x = self.fc_0(x)
        x = self.activation(x)
        x = self.fc_1(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, config: TransfusionConfig):
        super().__init__()

        self.config: TransfusionConfig = config

        self.attention = Attention(self.config.embed_dim, self.config.num_heads)

        self.norm_0 = nn.LayerNorm(self.config.embed_dim)
        self.mlp = MLP(self.config.embed_dim, self.config.ff_dim)
        self.norm_1 = nn.LayerNorm(self.config.embed_dim)


    def forward(self, x, attention_mask):
        attn_output = self.attention(x, attention_mask)

        x = self.norm_0(x + attn_output)
        ffn_output = self.mlp(x)
        x = self.norm_1(x + ffn_output)

        return x


class Transformer(nn.Module):
    def __init__(self, config: TransfusionConfig):
        super().__init__()

        self.config: TransfusionConfig = config

        self.wpe = nn.Embedding(self.config.block_size + 1, self.config.embed_dim) # We have to consider 1 times conditioning token too

        self.times_embedding = nn.Sequential(
            nn.Linear(1, self.config.embed_dim),
            nn.GELU(),
            nn.Linear(self.config.embed_dim, self.config.embed_dim)
        )

        self.layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(config.layers_num)
        ])


    def forward(self, x, attention_mask, times=None):
        batch_size, block_size, _ = x.shape

        x += self.wpe(torch.arange(block_size, device=x.device).repeat((batch_size, 1)))

        # Times is a (b,) float tensor
        # Which becomes a (b, d) -> (b, 1, d) tensor

        # TODO: Ask for advice on a better way to do this
        # TODO: Concatenate on the embedding dimension, maybe embed every image with a different noise level differently?

        times = rearrange(self.times_embedding(times), 'b d -> b 1 d')
        x = torch.cat((times, x), dim=1)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = x[:, 1:] # Remove the conditioning token

        return x


class ModalityType(Enum):
    TEXT = 0
    IMAGE = 1


class Modality:
    def __init__(self, type: ModalityType, data: torch.Tensor):
        self.type: ModalityType = type
        self.data: torch.Tensor = data


class Transfusion(nn.Module):
    def __init__(self, config: TransfusionConfig):
        super().__init__()

        self.config: TransfusionConfig = config

        self.wte = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        self.transformer = Transformer(self.config)
        
        self.image_downsampler = nn.Sequential(
            nn.Linear(self.config.image_token_dim, self.config.embed_dim),
            nn.GELU(),
            nn.Linear(self.config.embed_dim, self.config.embed_dim)
        )

        self.image_upsampler = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.GELU(),
            nn.Linear(self.config.embed_dim, self.config.image_token_dim)
        )

        self.fc = nn.Linear(self.config.embed_dim, self.config.vocab_size)


    @property
    def device(self):
        return next(self.parameters()).device


    def downsample_image(self, image):
        return self.image_downsampler(image)


    def upsample_image(self, embedding):
        return self.image_upsampler(embedding)


    def forward(self, x, times=None, compute_loss=False):
        if compute_loss:
            self.train()

        batch_size = len(x)

        max_sequence_length = self.config.block_size

        full_batch_data = torch.zeros((batch_size, max_sequence_length, self.config.embed_dim), device=self.device)
        full_batch_target_data = torch.zeros((batch_size, max_sequence_length, self.config.image_token_dim), device=self.device)

        attention_mask = torch.ones((batch_size, max_sequence_length + 1, max_sequence_length + 1), device=self.device).tril()
        image_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool, device=self.device)

        text_list = []
        image_targets_list = []

        if times is None:
            times = torch.rand((batch_size, 1), device=self.device)

        for idx, batch in enumerate(x):
            offset = 0

            current_text_list = []

            for modality in batch:
                if modality.type == ModalityType.TEXT:
                    text_embeddings = self.wte(modality.data)

                    seq_len = text_embeddings.shape[0]

                    full_batch_data[idx, offset:(offset + seq_len), :] = text_embeddings

                    offset += seq_len
                    current_text_list.append(modality.data)

                elif modality.type == ModalityType.IMAGE:
                    image = modality.data.squeeze(dim=0)

                    noise = torch.randn_like(image)
                    noisy_image = (times[idx] * image) + ((1.0 - times[idx]) * noise)

                    targets = image - noise

                    noisy_embeddings = self.downsample_image(noisy_image)

                    seq_len = noisy_embeddings.shape[0]

                    full_batch_data[idx, offset:(offset + seq_len), :] = noisy_embeddings
                    full_batch_target_data[idx, offset:(offset + seq_len), :] = targets

                    image_mask[idx, offset:(offset + seq_len)] = True
                    attention_mask[idx, (offset + 1):(offset + 1 + seq_len), (offset + 1):(offset + 1 + seq_len)] = 1 # We shift by 1 because of the conditioning token
                    
                    offset += seq_len
                    image_targets_list.append(targets)

            text_list.append(torch.cat(current_text_list, dim=0))

        text_list = pad_sequence(text_list, batch_first=True, padding_value=self.config.padding_token)

        # plt.imshow(attention_mask[0].detach().cpu().numpy())
        # plt.show()

        transformer_output = self.transformer(full_batch_data, attention_mask, times)

        # Text

        text_embeddings = transformer_output[~image_mask].view((batch_size, -1, self.config.embed_dim))

        text_embeddings = text_embeddings[:, :-1]
        text_targets = text_list[:, 1:]

        text_logits = self.fc(text_embeddings)

        # Images

        image_embeddings = transformer_output[image_mask].view((-1, self.config.embed_dim))        
        
        image_drift_preds = self.upsample_image(image_embeddings)
        image_targets = full_batch_target_data[image_mask].view((-1, self.config.image_token_dim))

        if not compute_loss:
            return text_logits, image_drift_preds

        loss_text = F.cross_entropy(
            text_logits.view((-1, self.config.vocab_size)),
            text_targets.flatten(),
            
            ignore_index=self.config.padding_token
        )

        loss_images = F.mse_loss(image_drift_preds, image_targets)

        loss_images_weight = 1.0
        loss_text_weight = 1.0

        loss = (loss_text_weight * loss_text) + (loss_images_weight * loss_images)

        return loss


class MNISTDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mnist = torchvision.datasets.MNIST(
            root='./data', 
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )


    def __len__(self):
        return len(self.mnist)


    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        return image, label


def create_digit_sequence(label: int, config: TransfusionConfig) -> torch.Tensor:
    sot_token = config.vocab_size - 4
    eot_token = config.vocab_size - 3

    digit_token = label

    return torch.tensor([sot_token, digit_token, eot_token])


def classify_image(
    model: Transfusion,
    image: torch.Tensor,
    config: TransfusionConfig,
    device: torch.device,
) -> torch.Tensor:

    model.eval()
    
    image_modality = Modality(ModalityType.IMAGE, image)

    text_sequence = torch.tensor([config.vocab_size - 4], device=device, dtype=torch.int64)
    text_modality = Modality(ModalityType.TEXT, text_sequence)
    
    input_sequence = [[image_modality, text_modality]]
    
    logits, _ = model(input_sequence)
    digit = logits[0, 0].detach().cpu().argmax().item()

    return digit


def sample_image(
    model: Transfusion,
    digit: int,
    config: TransfusionConfig,
    device: torch.device,
    num_steps: int = 100
) -> torch.Tensor:

    model.eval()
    
    text_sequence = create_digit_sequence(digit, config)
    text_modality = Modality(ModalityType.TEXT, text_sequence.to(device))

    image_noise = torch.randn((28, 28), device=device)
    image_modality = Modality(ModalityType.IMAGE, image_noise)
    
    input_sequence = [[text_modality, image_modality]]
    
    with torch.no_grad():
        for step in range(num_steps):
            times = torch.ones((1, 1), device=device) * (step / num_steps)
            _, drift_pred = model(input_sequence, times=times)

            input_sequence[0][1].data += drift_pred * (1.0 / num_steps)
    
    final_image = input_sequence[0][1].data.unsqueeze(dim=0)

    return final_image


def train(
    model: Transfusion,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    config: TransfusionConfig,
    device: torch.device,
    test_every: int = 32
) -> List[float]:

    model.train()

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            if (batch_idx + 1) % test_every == 0:
                digits = [classify_image(model, digit, config, device) for digit in images]
                accuracy = (sum([digit == label for digit, label in zip(digits, labels.detach().cpu().tolist())])) / len(images)

                wandb.log({
                    'accuracy': accuracy
                })

                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {epoch_losses[-1]:.4f}, Accuracy: {(accuracy * 100):.2f}%')

                with torch.no_grad():
                    digit_images = [sample_image(model, idx, config, device) for idx in range(10)]

                    fig, axes = plt.subplots(1, 10, figsize=(15, 2))

                    for idx, img in enumerate(digit_images):
                        axes[idx].imshow(img.cpu().squeeze(), cmap='gray')

                        axes[idx].axis('off')
                        axes[idx].set_title(f'{idx}')

                    plt.savefig('digits.png')
                    plt.close()

                continue

            batch_sequences = []

            for idx in range(batch_size):
                text_sequence = create_digit_sequence(labels[idx].item(), config)
                
                image_modality = Modality(ModalityType.IMAGE, images[idx])
                text_modality = Modality(ModalityType.TEXT, text_sequence.to(device))

                batch_sequences.append([image_modality, text_modality])

            loss = model(batch_sequences, times=None, compute_loss=True)
            # loss = model(batch_sequences, times=torch.ones((len(images), 1, 28), device=device), compute_loss=True)

            optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(loss.item())

            wandb.log({
                'loss': epoch_losses[-1]
            })

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)

        print(f'Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}')

    return losses


def train_with_wandb():
    wandb.init()

    config = wandb.config

    transfusion_config = TransfusionConfig(
        embed_dim=config.embed_dim,
        ff_dim=config.ff_dim,
        num_heads=config.num_heads,
        layers_num=config.layers_num
    )

    model = Transfusion(transfusion_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    train_dataset = MNISTDataset(train=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=1
    )

    train(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        num_epochs=4,
        config=transfusion_config,
        device=device
    )


def _main():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-3
            },
            'weight_decay': {
                'min': 0.0,
                'max': 0.1
            },
            
            'embed_dim': {
                'values': [128, 256, 512, 1024]
            },
            'ff_dim': {
                'values': [128, 256, 512, 1024]
            },
            'num_heads': {
                'values': [4, 8, 16]
            },
            'layers_num': {
                'values': [4, 6, 8]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='transfusion_project')
    wandb.agent(sweep_id, function=train_with_wandb)


def main():

    transfusion_config = TransfusionConfig()

    model = Transfusion(transfusion_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=1e-4,
        fused=True
    )

    train_dataset = MNISTDataset(train=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=1
    )

    train(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        num_epochs=4,
        config=transfusion_config,
        device=device
    )


if __name__ == '__main__':
    main()
