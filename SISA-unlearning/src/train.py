"""
SISA 학습 루프.

각 shard를 독립적으로 slice 순서대로 incremental 학습하고
slice 체크포인트를 저장한다.

체크포인트 경로: checkpoints/shard{i}_slice{j}.pt
체크포인트 형식: {model_state_dict, shard_index, slice_index, epoch}
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SISAConfig, SISADataset, get_dataloader
from model import get_model, load_checkpoint, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """한 epoch 학습 후 (loss, accuracy)를 반환."""
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
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_shard(
    shard_index: int,
    sisa_dataset: SISADataset,
    config: SISAConfig,
    slice_range: tuple[int, int] | None = None,
    arch: str = "simple_cnn",
    num_classes: int = 10,
    epochs_per_slice: int = 10,
    lr: float = 0.05,
    batch_size: int = 128,
    checkpoint_dir: str = "checkpoints",
    use_wandb: bool = False,
) -> None:
    """단일 shard를 slice 범위에 따라 incremental 학습.

    Args:
        shard_index: 학습할 shard 번호.
        slice_range: (start, end) 양 끝 포함. None이면 전체 slice 학습.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_slice, end_slice = (
        slice_range if slice_range is not None else (0, config.num_slices - 1)
    )

    model = get_model(arch=arch, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if start_slice > 0:
        prev_ckpt = ckpt_dir / f"shard{shard_index}_slice{start_slice - 1}.pt"
        if prev_ckpt.exists():
            load_checkpoint(model, prev_ckpt, device)
            print(f"  Resumed from {prev_ckpt.name}")

    if use_wandb:
        import wandb
        wandb.init(
            project="sisa-unlearning",
            name=f"shard{shard_index}",
            config={
                "arch": arch,
                "epochs_per_slice": epochs_per_slice,
                "lr": lr,
                "slice_range": [start_slice, end_slice],
            },
            reinit=True,
        )

    for slice_idx in range(start_slice, end_slice + 1):
        slice_dataset = sisa_dataset.get_slice_dataset(shard_index, slice_idx)
        loader = get_dataloader(slice_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_slice
        )

        print(
            f"  [Shard {shard_index}] Slice {slice_idx}/{end_slice}"
            f" — {len(slice_dataset)} samples"
        )

        for epoch in range(epochs_per_slice):
            loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()

            if use_wandb:
                import wandb
                wandb.log({
                    f"shard{shard_index}/loss": loss,
                    f"shard{shard_index}/acc": acc,
                    "slice": slice_idx,
                    "epoch": slice_idx * epochs_per_slice + epoch,
                })

        save_checkpoint(
            model,
            ckpt_dir / f"shard{shard_index}_slice{slice_idx}.pt",
            shard_index=shard_index,
            slice_index=slice_idx,
            epoch=epochs_per_slice - 1,
        )

    if use_wandb:
        import wandb
        wandb.finish()


def train_all_shards(
    sisa_dataset: SISADataset,
    config: SISAConfig,
    **train_kwargs,
) -> None:
    """모든 shard를 순차적으로 학습."""
    for shard_index in range(config.num_shards):
        print(f"\n=== Training Shard {shard_index}/{config.num_shards - 1} ===")
        train_shard(shard_index, sisa_dataset, config, **train_kwargs)