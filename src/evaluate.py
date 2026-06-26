"""
SISA 모델 성능 평가.

각 shard의 최종 slice 체크포인트를 로드해 예측을 모아
majority voting으로 앙상블한다.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SISAConfig, SISADataset
from model import get_model


def evaluate_single_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total


def aggregate_predictions(
    shard_logits: list[torch.Tensor],
) -> torch.Tensor:
    """shard별 logit을 평균 내어 앙상블 예측 반환."""
    stacked = torch.stack(shard_logits, dim=0)  # (S, N, C)
    return stacked.mean(dim=0).argmax(dim=1)    # (N,)


def evaluate_sisa(
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "resnet18",
    num_classes: int = 10,
    batch_size: int = 256,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, float]:
    """모든 shard 앙상블 평가."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        sisa_dataset.test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for shard_idx in range(config.num_shards):
        ckpt_path = (
            Path(checkpoint_dir)
            / f"shard_{shard_idx}"
            / f"slice_{config.num_slices - 1}.pt"
        )
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path} — skipping shard {shard_idx}")
            continue

        model = get_model(arch=arch, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        shard_logits, shard_targets = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader,
                                        desc=f"Shard {shard_idx}", leave=False):
                inputs = inputs.to(device)
                shard_logits.append(model(inputs).cpu())
                shard_targets.append(targets)

        all_logits.append(torch.cat(shard_logits))
        if not all_targets:
            all_targets = [torch.cat(shard_targets)]

    if not all_logits:
        return {"error": "no checkpoints found"}

    preds = aggregate_predictions(all_logits)
    targets = all_targets[0]
    accuracy = preds.eq(targets).float().mean().item()

    print(f"\nSISA Ensemble Accuracy: {accuracy * 100:.2f}%")
    return {"accuracy": accuracy, "num_shards_used": len(all_logits)}


def evaluate_forget_efficacy(
    forget_indices: list[int],
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "resnet18",
    num_classes: int = 10,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, float]:
    """forget된 샘플에 대한 예측 신뢰도를 측정해 망각 효과를 검증."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torch.utils.data import Subset

    forget_dataset = Subset(sisa_dataset.train_dataset, forget_indices)
    loader = DataLoader(forget_dataset, batch_size=64, shuffle=False)

    shard_logits: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    for shard_idx in range(config.num_shards):
        ckpt_path = (
            Path(checkpoint_dir)
            / f"shard_{shard_idx}"
            / f"slice_{config.num_slices - 1}.pt"
        )
        if not ckpt_path.exists():
            continue

        model = get_model(arch=arch, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        batch_logits, batch_targets = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                batch_logits.append(model(inputs.to(device)).cpu())
                batch_targets.append(targets)

        shard_logits.append(torch.cat(batch_logits))
        if not targets_list:
            targets_list.append(torch.cat(batch_targets))

    if not shard_logits:
        return {}

    preds = aggregate_predictions(shard_logits)
    targets = targets_list[0]
    forget_acc = preds.eq(targets).float().mean().item()
    print(f"Accuracy on forgotten samples: {forget_acc * 100:.2f}% "
          f"(lower is better for unlearning)")
    return {"forget_accuracy": forget_acc}
