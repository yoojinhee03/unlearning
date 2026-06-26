"""
SISA 망각(unlearning) 요청 처리.

forget 요청이 들어오면:
1. 해당 샘플이 포함된 shard를 찾는다.
2. shard에서 해당 샘플을 제거한다.
3. 영향받은 shard만 slice 체크포인트부터 재학습한다.
"""

from pathlib import Path
from typing import Optional

import torch

from dataset import SISAConfig, SISADataset
from train import train_shard


def find_retraining_start(
    shard_idx: int,
    forget_indices: list[int],
    sisa_dataset: SISADataset,
) -> int:
    """forget 샘플이 처음 등장하는 slice 번호를 반환."""
    forget_set = set(forget_indices)
    for slice_idx, cumulative in enumerate(sisa_dataset.slice_indices[shard_idx]):
        if forget_set & set(cumulative):
            return slice_idx
    return len(sisa_dataset.slice_indices[shard_idx]) - 1


def retrain_shard_from_slice(
    shard_idx: int,
    start_slice: int,
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "resnet18",
    num_classes: int = 10,
    epochs_per_slice: int = 5,
    lr: float = 0.01,
    batch_size: int = 128,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """start_slice부터 마지막 slice까지 재학습."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir) / f"shard_{shard_idx}"

    from model import get_model
    model = get_model(arch=arch, num_classes=num_classes).to(device)

    # start_slice 직전 체크포인트에서 시작 (없으면 처음부터)
    if start_slice > 0:
        prev_ckpt = ckpt_dir / f"slice_{start_slice - 1}.pt"
        if prev_ckpt.exists():
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))
            print(f"  Resumed from slice {start_slice - 1} checkpoint.")

    import torch.nn as nn
    from torch.utils.data import DataLoader
    from train import train_one_epoch

    criterion = nn.CrossEntropyLoss()
    num_slices = config.num_slices

    for slice_idx in range(start_slice, num_slices):
        slice_dataset = sisa_dataset.get_slice_dataset(shard_idx, slice_idx)
        loader = DataLoader(slice_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_slice
        )

        print(f"  [Unlearn Shard {shard_idx}] Retraining slice {slice_idx} "
              f"— {len(slice_dataset)} samples")

        for _ in range(epochs_per_slice):
            train_one_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()

        ckpt_path = ckpt_dir / f"slice_{slice_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)


def process_forget_request(
    forget_indices: list[int],
    sisa_dataset: SISADataset,
    config: SISAConfig,
    **train_kwargs,
) -> dict[int, int]:
    """
    forget 요청 처리 파이프라인.

    Returns:
        {shard_idx: start_slice} — 재학습이 수행된 shard와 시작 slice 정보.
    """
    affected_shards = sisa_dataset.get_forget_shards(forget_indices)
    print(f"Forget request: {len(forget_indices)} samples "
          f"→ affects shards {affected_shards}")

    retrain_info: dict[int, int] = {}
    for shard_idx in affected_shards:
        start_slice = find_retraining_start(shard_idx, forget_indices, sisa_dataset)

        # shard 데이터에서 forget 샘플 제거
        sisa_dataset.remove_and_rebuild_shard(shard_idx, forget_indices)

        print(f"\n=== Retraining Shard {shard_idx} from Slice {start_slice} ===")
        retrain_shard_from_slice(
            shard_idx, start_slice, sisa_dataset, config, **train_kwargs
        )
        retrain_info[shard_idx] = start_slice

    return retrain_info
