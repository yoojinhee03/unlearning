"""
SISA 망각(unlearning) 요청 처리.

point_to_shard.json과 metadata.json을 읽어
해당 데이터가 포함된 shard만 재학습한다.
"""

import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import SVHNDataset, get_dataloader
from model import get_model, load_checkpoint, save_checkpoint
from train import train_one_epoch


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _load_shard_map(shards_dir: str = "shards") -> dict[int, int]:
    """point_to_shard.json 로드. 키를 str→int 변환해 반환."""
    path = Path(shards_dir) / "point_to_shard.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 train을 실행하세요.")
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def _load_metadata(shards_dir: str = "shards") -> dict:
    """metadata.json 로드."""
    path = Path(shards_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 train을 실행하세요.")
    return json.loads(path.read_text())


def _first_slice_with_data(
    forget_set: set[int],
    cumulative_slices: list[list[int]],
) -> int:
    """forget 샘플이 처음 등장하는 slice 번호 반환 (누적 slice 기준)."""
    if not cumulative_slices:
        return 0
    for slice_idx, indices in enumerate(cumulative_slices):
        if forget_set & set(indices):
            return slice_idx
    return len(cumulative_slices) - 1


def _make_cumulative_slices(
    indices: list[int], num_slices: int
) -> list[list[int]]:
    """인덱스 목록을 num_slices개의 누적 slice로 분할."""
    n = len(indices)
    if n == 0:
        return []
    slice_size = math.ceil(n / num_slices)
    cumulative, accumulated = [], []
    for i in range(num_slices):
        chunk = indices[i * slice_size: (i + 1) * slice_size]
        if not chunk:
            break
        accumulated = accumulated + chunk
        cumulative.append(list(accumulated))
    return cumulative


# ── Public API ───────────────────────────────────────────────────────────────

def unlearn_request(
    data_indices: list[int],
    shards_dir: str = "shards",
    checkpoint_dir: str = "checkpoints",
    arch: str = "simple_cnn",
    num_classes: int = 10,
    epochs_per_slice: int = 10,
    lr: float = 0.05,
    batch_size: int = 128,
) -> dict:
    """망각 요청 파이프라인.

    1. point_to_shard_map에서 해당 데이터가 속한 shard 확인
    2. 해당 데이터가 처음 등장하는 slice 이전 체크포인트 로드
    3. 해당 데이터를 제외하고 그 shard만 재학습
    4. 재학습 소요 시간 측정 및 반환

    Returns:
        {
            "unlearn_time_sec": float,
            "affected_shards": list[int],
            "start_slices": dict[str, int],   # shard_idx → start_slice
        }
    """
    t_start = time.perf_counter()

    # 1. 영향받는 shard 확인
    shard_map  = _load_shard_map(shards_dir)
    forget_set = set(data_indices)
    affected: set[int] = {shard_map[i] for i in data_indices if i in shard_map}

    if not affected:
        print("요청된 인덱스가 어느 shard에도 없습니다.")
        return {"unlearn_time_sec": 0.0, "affected_shards": [], "start_slices": {}}

    print(f"Forget request: {len(data_indices)} samples → affects shards {sorted(affected)}")

    # 메타데이터에서 shard/slice 구조 로드
    meta         = _load_metadata(shards_dir)
    num_slices   = meta["num_slices"]
    shard_indices = meta["shard_indices"]   # [shard][sample]
    slice_indices = meta["slice_indices"]   # [shard][slice][sample] cumulative

    # HF SVHN train split (재학습에 필요)
    from datasets import load_dataset as hf_load_dataset
    hf_train = hf_load_dataset("svhn", "cropped_digits", split="train")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    start_slices: dict[str, int] = {}

    # metadata 갱신용 복사본
    updated_shard_indices = list(shard_indices)
    updated_slice_indices = list(slice_indices)

    for shard_idx in sorted(affected):
        # 2. forget 샘플이 처음 등장하는 slice 탐색
        start_slice = _first_slice_with_data(forget_set, slice_indices[shard_idx])
        start_slices[str(shard_idx)] = start_slice

        # forget 샘플 제거 후 누적 slice 재구성
        updated = [i for i in shard_indices[shard_idx] if i not in forget_set]
        new_slices = _make_cumulative_slices(updated, num_slices)

        # metadata 갱신본에 반영
        updated_shard_indices[shard_idx] = updated
        updated_slice_indices[shard_idx] = new_slices

        ckpt_dir = Path(checkpoint_dir)

        # shard가 완전히 비었으면 기존 체크포인트를 새 빈 모델로 덮어씀
        if not new_slices:
            print(f"\n=== Shard {shard_idx}: 모든 샘플 제거됨 — 빈 모델로 초기화 ===")
            fresh_model = get_model(arch=arch, num_classes=num_classes).to(device)
            for sl_idx in range(start_slice, num_slices):
                save_checkpoint(
                    fresh_model,
                    ckpt_dir / f"shard{shard_idx}_slice{sl_idx}.pt",
                    shard_index=shard_idx,
                    slice_index=sl_idx,
                    epoch=0,
                )
            continue

        # 직전 체크포인트에서 모델 초기화
        model = get_model(arch=arch, num_classes=num_classes).to(device)
        if start_slice > 0:
            prev_ckpt = ckpt_dir / f"shard{shard_idx}_slice{start_slice - 1}.pt"
            if prev_ckpt.exists():
                load_checkpoint(model, prev_ckpt, device)
                print(f"  Resumed from {prev_ckpt.name}")

        print(f"\n=== Retraining Shard {shard_idx} from Slice {start_slice} ===")

        # 3. start_slice부터 마지막까지 재학습
        for slice_idx in range(start_slice, len(new_slices)):
            slice_ds = SVHNDataset(hf_train, new_slices[slice_idx])
            loader   = get_dataloader(slice_ds, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_per_slice
            )

            print(f"  Slice {slice_idx} — {len(slice_ds)} samples")
            for _ in range(epochs_per_slice):
                train_one_epoch(model, loader, optimizer, criterion, device)
                scheduler.step()

            save_checkpoint(
                model,
                ckpt_dir / f"shard{shard_idx}_slice{slice_idx}.pt",
                shard_index=shard_idx,
                slice_index=slice_idx,
                epoch=epochs_per_slice - 1,
            )

        # 재학습된 slice 수가 num_slices보다 적으면 마지막 체크포인트를
        # evaluate_sisa가 항상 찾는 위치(slice{num_slices-1})에 복사해 stale 파일을 덮어씀
        if len(new_slices) < num_slices:
            last_written = ckpt_dir / f"shard{shard_idx}_slice{len(new_slices) - 1}.pt"
            stale_final  = ckpt_dir / f"shard{shard_idx}_slice{num_slices - 1}.pt"
            shutil.copy2(last_written, stale_final)

    # metadata.json 갱신 — 다음 unlearn 호출이 최신 인덱스를 읽도록 보장
    meta_updated = {
        "num_shards":    meta["num_shards"],
        "num_slices":    meta["num_slices"],
        "shard_indices": updated_shard_indices,
        "slice_indices": updated_slice_indices,
    }
    with open(Path(shards_dir) / "metadata.json", "w") as f:
        json.dump(meta_updated, f)

    # 4. 소요 시간 측정
    unlearn_time = time.perf_counter() - t_start
    print(f"\nUnlearning complete: {unlearn_time:.1f}s")

    return {
        "unlearn_time_sec": round(unlearn_time, 2),
        "affected_shards":  sorted(affected),
        "start_slices":     start_slices,
    }


def compare_unlearn_time(
    unlearn_time_sec: float | None = None,
    timing_path: str = "results_timing.json",
) -> dict:
    """SISA 망각 시간과 전체 재학습 시간 비교.

    Args:
        unlearn_time_sec: 직접 전달. None이면 timing_path에서 읽음.
        timing_path: main.py가 저장한 타이밍 JSON 경로.

    Returns:
        {"full_retrain_time_sec": float, "sisa_unlearn_time_sec": float, "speedup": float}
    """
    timing_file = Path(timing_path)
    if timing_file.exists():
        timing = json.loads(timing_file.read_text())
        full_retrain = timing.get("full_retrain_time_sec", 0.0)
        sisa_unlearn = unlearn_time_sec or timing.get("sisa_unlearn_time_sec", 0.0)
    else:
        print(f"Warning: {timing_path} 없음. speedup을 추정할 수 없습니다.")
        full_retrain = 0.0
        sisa_unlearn = unlearn_time_sec or 0.0

    speedup = full_retrain / sisa_unlearn if sisa_unlearn > 0 else float("inf")

    result = {
        "full_retrain_time_sec": round(full_retrain, 2),
        "sisa_unlearn_time_sec": round(sisa_unlearn, 2),
        "speedup":               round(speedup, 2),
    }

    W = 42
    print(f"\n{'─'*W}")
    print(f"  Full retrain time : {full_retrain:>8.1f}s")
    print(f"  SISA unlearn time : {sisa_unlearn:>8.1f}s")
    print(f"  Speedup           : {speedup:>8.2f}x")
    print(f"{'─'*W}")

    return result


# ── 하위 호환 래퍼 ────────────────────────────────────────────────────────────

def process_forget_request(
    forget_indices: list[int],
    sisa_dataset,
    config,
    **train_kwargs,
) -> dict[int, int]:
    """SISADataset 기반 기존 API. unlearn_request()를 내부적으로 사용."""
    result = unlearn_request(
        forget_indices,
        shards_dir=train_kwargs.get(
            "shards_dir", getattr(config, "shards_dir", "shards")
        ),
        checkpoint_dir=train_kwargs.get("checkpoint_dir", "checkpoints"),
        arch=train_kwargs.get("arch", "simple_cnn"),
        num_classes=train_kwargs.get("num_classes", 10),
        epochs_per_slice=train_kwargs.get("epochs_per_slice", 10),
        lr=train_kwargs.get("lr", 0.05),
        batch_size=train_kwargs.get("batch_size", 128),
    )
    return {int(k): v for k, v in result["start_slices"].items()}