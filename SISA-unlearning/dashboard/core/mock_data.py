"""실제 학습 없이 대시보드를 시연할 때 사용할 Mock 데이터 생성."""

from __future__ import annotations

import math
import random


def generate_training_progress(
    num_shards: int = 5,
    num_slices: int = 4,
) -> dict:
    """shard별 학습 진행 상태 dict 반환.

    Returns:
        {
            shard_idx: {
                "status":   "done",          # idle|training|done|retraining
                "accuracy": float,           # 최종 slice 정확도 (0~1)
                "loss":     float,           # 최종 slice loss
                "slices": [
                    {"slice": int, "accuracy": float, "loss": float},
                    ...
                ],
            },
            ...
        }
    """
    rng = random.Random(42)
    result: dict = {}

    for s in range(num_shards):
        slices = []
        acc  = 0.10 + rng.uniform(-0.02, 0.02)
        loss = 2.30 + rng.uniform(-0.15, 0.15)

        for sl in range(num_slices):
            acc  = min(0.935, acc  + rng.uniform(0.08, 0.14))
            loss = max(0.22,  loss * (0.52 + rng.uniform(-0.04, 0.04)))
            slices.append({
                "slice":    sl,
                "accuracy": round(acc,  4),
                "loss":     round(loss, 4),
            })

        result[s] = {
            "status":   "done",
            "accuracy": slices[-1]["accuracy"],
            "loss":     slices[-1]["loss"],
            "slices":   slices,
        }

    return result


def generate_accuracy_curve(
    num_slices: int = 4,
    noise: float = 0.02,
) -> list[float]:
    """slice별 정확도 곡선 (89~91% 범위, 노이즈 포함).

    Returns:
        길이 num_slices의 정확도 리스트 (0~1 스케일).
    """
    rng = random.Random(7)
    target_min, target_max = 0.89, 0.91

    # 선형 증가 후 수렴
    curve = []
    for i in range(num_slices):
        t = i / max(num_slices - 1, 1)
        # 0 → 0.80, 1 → target 구간으로 수렴
        base = 0.80 + (target_min + (target_max - target_min) * 0.5 - 0.80) * (
            1 - math.exp(-3 * t)
        )
        val = base + rng.gauss(0, noise)
        curve.append(round(min(0.99, max(0.70, val)), 4))

    return curve


def generate_loss_curve(num_slices: int = 4) -> list[float]:
    """slice별 loss 곡선 (0.8 → 0.25 수렴).

    Returns:
        길이 num_slices의 loss 리스트.
    """
    rng = random.Random(13)
    start, end = 0.80, 0.25

    curve = []
    for i in range(num_slices):
        t = i / max(num_slices - 1, 1)
        base = end + (start - end) * math.exp(-3 * t)
        val  = base + rng.gauss(0, 0.012)
        curve.append(round(max(0.10, val), 4))

    return curve


def simulate_unlearn_time(
    affected_shards: int | list,
    baseline_time: float = 160.0,
) -> dict:
    """SISA 망각 시간과 전체 재학습 시간 시뮬레이션.

    Args:
        affected_shards: 영향받은 shard 수 또는 shard 인덱스 리스트.
        baseline_time:   전체 재학습 baseline 시간 (초).

    Returns:
        {"sisa_time": float, "baseline_time": float, "speedup": float}
    """
    n = affected_shards if isinstance(affected_shards, int) else len(affected_shards)
    # SISA 시간 = baseline / 총 shard 수 × 영향받은 shard 수 (+ 약간의 오버헤드)
    total_shards = max(n, 5)
    sisa_time = round(baseline_time * (n / total_shards) * 1.05, 2)
    speedup   = round(baseline_time / sisa_time, 2) if sisa_time > 0 else float("inf")

    return {
        "sisa_time":     sisa_time,
        "baseline_time": round(baseline_time, 2),
        "speedup":       speedup,
    }


def generate_paper_results() -> dict:
    """Bourtoule et al. (2021) Table 2 논문 수치 반환.

    Returns:
        {
            "purchase": {"speedup": float, "accuracy_drop": float},
            "svhn":     {"speedup": float, "accuracy_drop": float},
            "imagenet": {"speedup": float, "accuracy_drop": float},
        }
    """
    return {
        "purchase": {"speedup": 4.63, "accuracy_drop": 1.8},
        "svhn":     {"speedup": 2.45, "accuracy_drop": 1.2},
        "imagenet": {"speedup": 1.36, "accuracy_drop": 19.45},
    }