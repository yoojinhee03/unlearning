"""
SISA 학습 루프.

각 shard를 독립적으로 slice 순서대로 incremental 학습하고
slice 체크포인트를 저장한다.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SISAConfig, SISADataset
from model import get_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def train_shard(
    shard_idx: int,
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "resnet18",
    num_classes: int = 10,
    epochs_per_slice: int = 5,
    lr: float = 0.01,
    batch_size: int = 128,
    checkpoint_dir: str = "checkpoints",
    use_wandb: bool = False,
) -> None:
    """단일 shard를 slice 순서대로 incremental 학습."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir) / f"shard_{shard_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(arch=arch, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if use_wandb:
        import wandb
        wandb.init(
            project="sisa-unlearning",
            name=f"shard_{shard_idx}",
            config={"arch": arch, "epochs_per_slice": epochs_per_slice, "lr": lr},
            reinit=True,
        )

    num_slices = config.num_slices
    for slice_idx in range(num_slices):
        # 이전 slice 체크포인트가 있으면 이어서 학습
        if slice_idx > 0:
            prev_ckpt = ckpt_dir / f"slice_{slice_idx - 1}.pt"
            if prev_ckpt.exists():
                model.load_state_dict(torch.load(prev_ckpt, map_location=device))

        slice_dataset = sisa_dataset.get_slice_dataset(shard_idx, slice_idx)
        loader = DataLoader(slice_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_slice
        )

        print(f"  [Shard {shard_idx}] Slice {slice_idx}/{num_slices - 1} "
              f"— {len(slice_dataset)} samples")

        for epoch in range(epochs_per_slice):
            loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if use_wandb:
                import wandb
                wandb.log({
                    f"shard_{shard_idx}/loss": loss,
                    f"shard_{shard_idx}/acc": acc,
                    "slice": slice_idx,
                    "epoch": slice_idx * epochs_per_slice + epoch,
                })

        ckpt_path = ckpt_dir / f"slice_{slice_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)

    if use_wandb:
        import wandb
        wandb.finish()


def train_all_shards(
    sisa_dataset: SISADataset,
    config: SISAConfig,
    **train_kwargs,
) -> None:
    """모든 shard를 순차적으로 학습. GPU가 충분하면 병렬화 가능."""
    for shard_idx in range(config.num_shards):
        print(f"\n=== Training Shard {shard_idx}/{config.num_shards - 1} ===")
        train_shard(shard_idx, sisa_dataset, config, **train_kwargs)
