"""
SISA Training 진입점.

사용법:
  # 전체 학습
  python main.py train

  # 망각 요청 처리 (인덱스 0~99 제거)
  python main.py unlearn --forget-indices 0 1 2 ... 99

  # 평가
  python main.py evaluate
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "src")

from dataset import SISAConfig, SISADataset
from evaluate import (
    evaluate_forget_efficacy,
    evaluate_sisa,
    save_results,
)
from train import train_all_shards
from unlearn import process_forget_request

TRAIN_KWARGS = dict(
    arch="simple_cnn",
    num_classes=10,
    epochs_per_slice=10,
    lr=0.05,
    batch_size=128,
    checkpoint_dir="checkpoints",
)

_TIMING_FILE = "results_timing.json"
_RESULTS_FILE = "results.json"

EVAL_KWARGS = {k: v for k, v in TRAIN_KWARGS.items() if k in ("arch", "num_classes", "checkpoint_dir")}
EVAL_KWARGS_BATCH = {**EVAL_KWARGS, "batch_size": TRAIN_KWARGS["batch_size"]}


def main():
    parser = argparse.ArgumentParser(description="SISA Training")
    parser.add_argument("mode", choices=["train", "unlearn", "evaluate"])
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--num-slices", type=int, default=4)
    parser.add_argument("--forget-indices", type=int, nargs="+",
                        help="unlearn 모드에서 제거할 샘플 인덱스")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    config = SISAConfig(
        num_shards=args.num_shards,
        num_slices=args.num_slices,
    )
    sisa_dataset = SISADataset(config)

    if args.mode == "train":
        print("Building shards...")
        sisa_dataset.build_shards()

        print("Starting SISA training...")
        t0 = time.perf_counter()
        train_all_shards(
            sisa_dataset, config,
            **{**TRAIN_KWARGS, "use_wandb": args.use_wandb},
        )
        full_retrain_time = time.perf_counter() - t0
        print(f"\nTotal training time: {full_retrain_time:.1f}s")

        result = evaluate_sisa(sisa_dataset, config, **EVAL_KWARGS_BATCH)

        # 타이밍과 망각 전 정확도를 저장해 unlearn 모드에서 참조
        with open(_TIMING_FILE, "w") as f:
            json.dump(
                {
                    "full_retrain_time_sec": round(full_retrain_time, 2),
                    "accuracy_before": result.get("accuracy", 0.0),
                },
                f,
            )

    elif args.mode == "unlearn":
        if not args.forget_indices:
            parser.error("--forget-indices가 필요합니다.")
        if not sisa_dataset.load_shard_metadata():
            parser.error("shards/metadata.json 없음. 먼저 train을 실행하세요.")

        # 망각 전 정확도/타이밍 로드
        timing = {}
        if (p := __import__("pathlib").Path(_TIMING_FILE)).exists():
            timing = json.loads(p.read_text())
        accuracy_before = timing.get("accuracy_before", 0.0)
        full_retrain_time = timing.get("full_retrain_time_sec", 0.0)

        t0 = time.perf_counter()
        process_forget_request(
            args.forget_indices, sisa_dataset, config, **TRAIN_KWARGS
        )
        sisa_unlearn_time = time.perf_counter() - t0
        print(f"\nUnlearning time: {sisa_unlearn_time:.1f}s")

        after_result = evaluate_sisa(sisa_dataset, config, **EVAL_KWARGS_BATCH)
        accuracy_after = after_result.get("accuracy", 0.0)

        forget_result = evaluate_forget_efficacy(
            args.forget_indices, sisa_dataset, config, **EVAL_KWARGS
        )

        save_results(
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            full_retrain_time=full_retrain_time,
            sisa_unlearn_time=sisa_unlearn_time,
            forget_accuracy=forget_result.get("forget_accuracy"),
            save_path=_RESULTS_FILE,
        )

    elif args.mode == "evaluate":
        if not sisa_dataset.load_shard_metadata():
            parser.error("shards/metadata.json 없음. 먼저 train을 실행하세요.")
        evaluate_sisa(sisa_dataset, config, **EVAL_KWARGS_BATCH)


if __name__ == "__main__":
    main()