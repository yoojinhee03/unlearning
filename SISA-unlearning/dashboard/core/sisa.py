"""대시보드 ↔ src/ 브리지.

demo_mode=True  → mock_data.py 함수로 시뮬레이션
demo_mode=False → src/ 모듈을 직접 호출
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# src/ 경로를 sys.path에 추가 (src 모듈 import 가능)
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from . import mock_data as _mock


# ── 학습 상태 ─────────────────────────────────────────────────────────────────

def get_training_status(
    num_shards: int = 5,
    num_slices: int = 4,
    checkpoint_dir: str = "checkpoints",
    demo_mode: bool = True,
) -> dict:
    """현재 shard/slice 학습 상태 반환.

    Returns:
        {
            shard_idx: {
                "status":   "idle"|"training"|"done"|"retraining",
                "accuracy": float,
                "loss":     float,
                "slices":   [{"slice": int, "accuracy": float, "loss": float}, ...],
            },
            ...
        }
    """
    if demo_mode:
        return _mock.generate_training_progress(num_shards, num_slices)

    # 실제 모드: 체크포인트 존재 여부로 status 결정
    ckpt_dir = _ROOT / checkpoint_dir
    result: dict = {}

    for s in range(num_shards):
        slices = []
        for sl in range(num_slices):
            path = ckpt_dir / f"shard{s}_slice{sl}.pt"
            if path.exists():
                import torch
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                slices.append({
                    "slice":    sl,
                    "accuracy": float(ckpt.get("val_accuracy", 0.0)),
                    "loss":     float(ckpt.get("val_loss",     0.0)),
                })

        status   = "done" if len(slices) == num_slices else (
                   "training" if slices else "idle")
        result[s] = {
            "status":   status,
            "accuracy": slices[-1]["accuracy"] if slices else 0.0,
            "loss":     slices[-1]["loss"]     if slices else 0.0,
            "slices":   slices,
        }

    return result


# ── 망각 요청 ─────────────────────────────────────────────────────────────────

def run_unlearn(
    data_indices: list[int],
    checkpoint_dir: str = "checkpoints",
    demo_mode: bool = True,
    num_shards: int = 5,
    **kwargs,
) -> dict:
    """망각 요청 실행 및 결과 반환.

    Returns:
        demo_mode=True  → simulate_unlearn_time() 결과
        demo_mode=False → unlearn_request() 결과
            {"unlearn_time_sec", "affected_shards", "start_slices"}
    """
    if demo_mode:
        # 영향받을 shard 수를 인덱스 분포로 추정
        n_affected = max(1, len(set(i % num_shards for i in data_indices)))
        return _mock.simulate_unlearn_time(n_affected, num_shards=num_shards)

    from unlearn import unlearn_request
    return unlearn_request(
        data_indices=data_indices,
        checkpoint_dir=str(_ROOT / checkpoint_dir),
        **kwargs,
    )


# ── 결과 파일 ─────────────────────────────────────────────────────────────────

def get_results(
    results_path: str = "experiments/results.json",
    demo_mode: bool = True,
) -> dict | None:
    """experiments/results.json을 읽어 반환.

    Returns:
        결과 dict, 또는 파일이 없을 때 None (demo_mode=False) /
        mock 결과 (demo_mode=True).
    """
    if demo_mode:
        sim = _mock.simulate_unlearn_time(affected_shards=1, num_shards=5)
        paper = _mock.generate_paper_results()["svhn"]
        return {
            "speedup":               sim["speedup"],
            "accuracy_before":       90.12,
            "accuracy_after":        88.91,
            "accuracy_drop":         round(90.12 - 88.91, 2),
            "full_retrain_time_sec": sim["baseline_time"],
            "sisa_unlearn_time_sec": sim["sisa_time"],
            # 논문 참고치
            "paper_speedup":         paper["speedup"],
            "paper_accuracy_drop":   paper["accuracy_drop"],
        }

    path = _ROOT / results_path
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── shard 맵 ──────────────────────────────────────────────────────────────────

def get_point_to_shard_map(
    map_path: str = "shards/point_to_shard.json",
    demo_mode: bool = True,
    num_shards: int = 5,
    dataset_size: int = 73257,
) -> dict[int, int]:
    """data index → shard index 매핑 반환.

    Returns:
        demo_mode=True  → 균등 분할 mock 맵
        demo_mode=False → shards/point_to_shard.json 파일 내용
    """
    if demo_mode:
        shard_size = dataset_size // num_shards
        return {i: min(i // shard_size, num_shards - 1) for i in range(dataset_size)}

    path = _ROOT / map_path
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


# ── 하위 호환 (이전 코드가 참조하는 함수들) ──────────────────────────────────

def checkpoint_status(num_shards: int, num_slices: int,
                      checkpoint_dir: str = "checkpoints"):
    """shard×slice 완료 여부 DataFrame. (이전 API 호환)"""
    import pandas as pd
    ckpt_dir = _ROOT / checkpoint_dir
    matrix = [
        [1 if (ckpt_dir / f"shard{s}_slice{sl}.pt").exists() else 0
         for sl in range(num_slices)]
        for s in range(num_shards)
    ]
    return pd.DataFrame(
        matrix,
        index=[f"shard {i}" for i in range(num_shards)],
        columns=[f"slice {j}" for j in range(num_slices)],
    )


def any_checkpoint_exists(checkpoint_dir: str = "checkpoints") -> bool:
    return any((_ROOT / checkpoint_dir).glob("shard*.pt"))


def load_results(results_path: str = "experiments/results.json") -> dict | None:
    path = _ROOT / results_path
    return json.loads(path.read_text()) if path.exists() else None


def run_training(num_shards: int, num_slices: int,
                 epochs_per_slice: int = 10,
                 use_wandb: bool = False) -> subprocess.Popen:
    cmd = [
        sys.executable, str(_ROOT / "main.py"), "train",
        "--num-shards", str(num_shards),
        "--num-slices", str(num_slices),
    ]
    if use_wandb:
        cmd.append("--use-wandb")
    return subprocess.Popen(cmd, cwd=str(_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_unlearning(forget_indices: list[int], num_shards: int,
                   num_slices: int) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(_ROOT / "main.py"), "unlearn",
        "--num-shards", str(num_shards),
        "--num-slices", str(num_slices),
        "--forget-indices", *map(str, forget_indices),
    ]
    return subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)


def run_experiment(num_shards: int, num_slices: int,
                   forget_indices: list[int],
                   results_path: str = "experiments/results.json") -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_ROOT / "experiments" / "run_experiment.py"),
        "--num_shards", str(num_shards),
        "--num_slices", str(num_slices),
        "--unlearn_indices", *map(str, forget_indices),
        "--results_path", results_path,
    ]
    return subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)