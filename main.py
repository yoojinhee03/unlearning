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
import sys

sys.path.insert(0, "src")

from dataset import SISAConfig, SISADataset
from evaluate import evaluate_forget_efficacy, evaluate_sisa
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
        train_all_shards(
            sisa_dataset, config,
            **{**TRAIN_KWARGS, "use_wandb": args.use_wandb},
        )
        evaluate_sisa(sisa_dataset, config, **{
            k: v for k, v in TRAIN_KWARGS.items()
            if k in ("arch", "num_classes", "batch_size", "checkpoint_dir")
        })

    elif args.mode == "unlearn":
        if not args.forget_indices:
            parser.error("--forget-indices가 필요합니다.")
        if not sisa_dataset.load_shard_metadata():
            parser.error("shards/metadata.json 없음. 먼저 train을 실행하세요.")
        process_forget_request(
            args.forget_indices, sisa_dataset, config, **TRAIN_KWARGS
        )
        evaluate_forget_efficacy(
            args.forget_indices, sisa_dataset, config, **{
                k: v for k, v in TRAIN_KWARGS.items()
                if k in ("arch", "num_classes", "checkpoint_dir")
            }
        )

    elif args.mode == "evaluate":
        if not sisa_dataset.load_shard_metadata():
            parser.error("shards/metadata.json 없음. 먼저 train을 실행하세요.")
        evaluate_sisa(sisa_dataset, config, **{
            k: v for k, v in TRAIN_KWARGS.items()
            if k in ("arch", "num_classes", "batch_size", "checkpoint_dir")
        })


if __name__ == "__main__":
    main()
