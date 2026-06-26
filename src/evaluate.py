"""
SISA 모델 성능 평가.

각 shard의 최종 slice 체크포인트를 로드해 예측을 모아
majority voting으로 앙상블한다.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import SISAConfig, SISADataset, get_dataloader
from model import get_model, load_checkpoint

# Bourtoule et al. (2021) Table 2 목표치
_PAPER_TARGETS = {
    "Purchase": {"speedup": 4.63, "max_drop_pp": 2.0},
    "SVHN":     {"speedup": 2.45, "max_drop_pp": 2.0},
}


def aggregate_predictions(shard_logits: list[torch.Tensor]) -> torch.Tensor:
    """Majority vote로 shard 예측을 집계.

    각 shard의 argmax 클래스에 투표하고, 가장 많이 나온 클래스를 최종 예측으로 선택한다.
    동수일 때는 torch.mode의 낮은 인덱스 우선 동작을 따른다.
    """
    shard_preds = torch.stack(
        [logits.argmax(dim=1) for logits in shard_logits], dim=0
    )  # (S, N)
    votes, _ = torch.mode(shard_preds, dim=0)  # (N,)
    return votes


def _collect_shard_logits(
    sisa_dataset: SISADataset,
    config: SISAConfig,
    loader: DataLoader,
    arch: str,
    num_classes: int,
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """모든 shard의 logit과 정답 레이블 수집."""
    all_logits: list[torch.Tensor] = []
    targets: torch.Tensor | None = None

    for shard_idx in range(config.num_shards):
        ckpt_path = (
            Path(checkpoint_dir) / f"shard{shard_idx}_slice{config.num_slices - 1}.pt"
        )
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path} — skipping shard {shard_idx}")
            continue

        model = get_model(arch=arch, num_classes=num_classes).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.eval()

        batch_logits, batch_targets = [], []
        with torch.no_grad():
            for inputs, tgts in tqdm(loader, desc=f"Shard {shard_idx}", leave=False):
                batch_logits.append(model(inputs.to(device)).cpu())
                batch_targets.append(tgts)

        all_logits.append(torch.cat(batch_logits))
        if targets is None:
            targets = torch.cat(batch_targets)

    return all_logits, targets


def evaluate_sisa(
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "simple_cnn",
    num_classes: int = 10,
    batch_size: int = 256,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, float]:
    """모든 shard 앙상블로 테스트셋 전체 정확도를 측정."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloader(sisa_dataset.test_dataset, batch_size=batch_size)

    all_logits, targets = _collect_shard_logits(
        sisa_dataset, config, test_loader, arch, num_classes, checkpoint_dir, device
    )

    if not all_logits:
        print("  No checkpoints found.")
        return {"accuracy": 0.0, "num_shards_used": 0}

    preds = aggregate_predictions(all_logits)
    accuracy = preds.eq(targets).float().mean().item()
    print(
        f"\nSISA Ensemble Accuracy: {accuracy * 100:.2f}%"
        f"  ({len(all_logits)}/{config.num_shards} shards)"
    )
    return {"accuracy": accuracy, "num_shards_used": len(all_logits)}


def evaluate_forget_efficacy(
    forget_indices: list[int],
    sisa_dataset: SISADataset,
    config: SISAConfig,
    arch: str = "simple_cnn",
    num_classes: int = 10,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, float]:
    """forget된 샘플에 대한 예측 정확도를 측정해 망각 효과를 검증."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forget_dataset = Subset(sisa_dataset.train_dataset, forget_indices)
    loader = DataLoader(forget_dataset, batch_size=64, shuffle=False)

    all_logits: list[torch.Tensor] = []
    targets: torch.Tensor | None = None

    for shard_idx in range(config.num_shards):
        ckpt_path = (
            Path(checkpoint_dir) / f"shard{shard_idx}_slice{config.num_slices - 1}.pt"
        )
        if not ckpt_path.exists():
            continue

        model = get_model(arch=arch, num_classes=num_classes).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.eval()

        batch_logits, batch_targets = [], []
        with torch.no_grad():
            for inputs, tgts in loader:
                batch_logits.append(model(inputs.to(device)).cpu())
                batch_targets.append(tgts)

        all_logits.append(torch.cat(batch_logits))
        if targets is None:
            targets = torch.cat(batch_targets)

    if not all_logits:
        return {}

    preds = aggregate_predictions(all_logits)
    forget_acc = preds.eq(targets).float().mean().item()
    print(
        f"Accuracy on forgotten samples: {forget_acc * 100:.2f}%"
        f"  (lower → better unlearning)"
    )
    return {"forget_accuracy": forget_acc}


def save_results(
    accuracy_before: float,
    accuracy_after: float,
    full_retrain_time: float,
    sisa_unlearn_time: float,
    forget_accuracy: float | None = None,
    save_path: str = "results.json",
) -> dict:
    """측정 결과를 results.json으로 저장하고 논문 목표치와 비교 표를 출력.

    Args:
        accuracy_before: 망각 전 테스트셋 정확도 (0~1).
        accuracy_after: 망각 후 테스트셋 정확도 (0~1).
        full_retrain_time: 전체 재학습 baseline 시간 (초).
        sisa_unlearn_time: SISA 망각 처리 시간 (초).
        forget_accuracy: forgotten 샘플 정확도 (0~1). None이면 생략.
        save_path: 저장할 JSON 경로.
    """
    speedup = (
        full_retrain_time / sisa_unlearn_time if sisa_unlearn_time > 0 else float("inf")
    )
    accuracy_drop_pp = (accuracy_before - accuracy_after) * 100

    results = {
        "full_retrain_time_sec": round(full_retrain_time, 2),
        "sisa_unlearn_time_sec": round(sisa_unlearn_time, 2),
        "speedup": round(speedup, 2),
        "accuracy_before_pct": round(accuracy_before * 100, 2),
        "accuracy_after_pct": round(accuracy_after * 100, 2),
        "accuracy_drop_pp": round(accuracy_drop_pp, 2),
    }
    if forget_accuracy is not None:
        results["forget_sample_accuracy_pct"] = round(forget_accuracy * 100, 2)

    Path(save_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved → {save_path}")

    _print_before_after(accuracy_before, accuracy_after)
    _print_paper_comparison(speedup, accuracy_drop_pp)
    return results


def _print_before_after(accuracy_before: float, accuracy_after: float) -> None:
    drop = (accuracy_before - accuracy_after) * 100
    arrow = "▼" if drop > 0 else ("▲" if drop < 0 else "─")
    print(f"\n{'─'*42}")
    print(f"  Accuracy before unlearning : {accuracy_before * 100:6.2f}%")
    print(f"  Accuracy after  unlearning : {accuracy_after * 100:6.2f}%")
    print(f"  Change                     : {arrow} {abs(drop):.2f}%p")
    print(f"{'─'*42}")


def _print_paper_comparison(speedup: float, accuracy_drop_pp: float) -> None:
    W = 74
    print(f"\n{'='*W}")
    print("  Paper Comparison — Bourtoule et al. (2021), SISA Training")
    print(f"{'='*W}")
    print(
        f"  {'Dataset':<12}"
        f"  {'Target Speedup':>14}  {'Achieved':>10}  {'':6}"
        f"  {'Target Drop':>12}  {'Achieved':>10}  {''}"
    )
    print(f"  {'─'*12}  {'─'*14}  {'─'*10}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*6}")

    for dataset, t in _PAPER_TARGETS.items():
        ok_speed = "PASS" if speedup >= t["speedup"] else "FAIL"
        ok_drop  = "PASS" if accuracy_drop_pp <= t["max_drop_pp"] else "FAIL"
        print(
            f"  {dataset:<12}"
            f"  {t['speedup']:>12.2f}x  {speedup:>8.2f}x  {ok_speed:>6}"
            f"  {t['max_drop_pp']:>10.1f}%p  {accuracy_drop_pp:>8.2f}%p  {ok_drop:>6}"
        )

    print(f"{'='*W}\n")