"""
데이터 로드 및 SISA용 shard/slice 분할.

SISA는 전체 데이터셋을 S개의 shard로 나누고,
각 shard를 다시 R개의 slice로 나눠 incremental하게 학습한다.
"""

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class SISAConfig:
    num_shards: int = 5
    num_slices: int = 4
    dataset_name: str = "cifar10"
    data_dir: str = "data"
    shards_dir: str = "shards"
    seed: int = 42


class SISADataset:
    """전체 데이터셋을 shard × slice 구조로 관리."""

    def __init__(self, config: SISAConfig):
        self.config = config
        self.shards_dir = Path(config.shards_dir)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset, self.test_dataset = self._load_base_dataset()
        self.shard_indices: list[list[int]] = []
        self.slice_indices: list[list[list[int]]] = []

    def _load_base_dataset(self) -> tuple[Dataset, Dataset]:
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.config.dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            train = datasets.CIFAR10(str(data_dir), train=True,
                                     download=True, transform=transform_train)
            test = datasets.CIFAR10(str(data_dir), train=False,
                                    download=True, transform=transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")

        return train, test

    def build_shards(self, forget_indices: Optional[list[int]] = None) -> None:
        """전체 훈련 데이터를 S개 shard로 분할하고 각 shard를 R개 slice로 분할."""
        rng = np.random.default_rng(self.config.seed)
        n = len(self.train_dataset)
        all_indices = rng.permutation(n).tolist()

        shard_size = math.ceil(n / self.config.num_shards)
        self.shard_indices = [
            all_indices[i * shard_size: (i + 1) * shard_size]
            for i in range(self.config.num_shards)
        ]

        self.slice_indices = []
        for shard_idx, shard in enumerate(self.shard_indices):
            slices = self._split_into_slices(shard)
            self.slice_indices.append(slices)

        self._save_shard_metadata()

    def _split_into_slices(self, shard: list[int]) -> list[list[int]]:
        """단일 shard를 R개의 누적 slice로 분할."""
        slice_size = math.ceil(len(shard) / self.config.num_slices)
        raw_slices = [
            shard[i * slice_size: (i + 1) * slice_size]
            for i in range(self.config.num_slices)
        ]
        # slice r은 slice 0..r의 누적 데이터를 사용
        cumulative = []
        accumulated: list[int] = []
        for s in raw_slices:
            accumulated = accumulated + s
            cumulative.append(list(accumulated))
        return cumulative

    def get_slice_dataset(self, shard_idx: int, slice_idx: int) -> Subset:
        indices = self.slice_indices[shard_idx][slice_idx]
        return Subset(self.train_dataset, indices)

    def get_forget_shards(self, forget_indices: list[int]) -> list[int]:
        """forget 요청 인덱스가 속한 shard 번호 목록 반환."""
        forget_set = set(forget_indices)
        affected = []
        for shard_idx, shard in enumerate(self.shard_indices):
            if forget_set & set(shard):
                affected.append(shard_idx)
        return affected

    def remove_and_rebuild_shard(
        self, shard_idx: int, forget_indices: list[int]
    ) -> None:
        """특정 shard에서 forget 인덱스를 제거하고 slice를 재구성."""
        forget_set = set(forget_indices)
        self.shard_indices[shard_idx] = [
            idx for idx in self.shard_indices[shard_idx] if idx not in forget_set
        ]
        self.slice_indices[shard_idx] = self._split_into_slices(
            self.shard_indices[shard_idx]
        )
        self._save_shard_metadata()

    def _save_shard_metadata(self) -> None:
        meta = {
            "num_shards": self.config.num_shards,
            "num_slices": self.config.num_slices,
            "shard_indices": self.shard_indices,
            "slice_indices": self.slice_indices,
        }
        with open(self.shards_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

    def load_shard_metadata(self) -> bool:
        path = self.shards_dir / "metadata.json"
        if not path.exists():
            return False
        with open(path) as f:
            meta = json.load(f)
        self.shard_indices = meta["shard_indices"]
        self.slice_indices = meta["slice_indices"]
        return True
