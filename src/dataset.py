"""
SVHN 데이터셋 로드 및 SISA용 shard/slice 분할.

load_dataset → split_into_shards → split_into_slices 순으로 사용.
SISADataset은 이 저수준 함수들을 묶어 shard×slice 구조 전체를 관리한다.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_SVHN_MEAN = (0.4377, 0.4438, 0.4728)
_SVHN_STD = (0.1980, 0.2010, 0.1970)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_SVHN_MEAN, _SVHN_STD),
])


class SVHNDataset(Dataset):
    """HuggingFace SVHN을 PyTorch Dataset으로 래핑.

    indices는 HF 데이터셋 내 원본 인덱스를 직접 가리키므로,
    shard/slice 분할 시 데이터 복사 없이 인덱스 슬라이싱으로 뷰를 만든다.
    """

    def __init__(self, hf_dataset, indices: Optional[List[int]] = None):
        self._hf = hf_dataset
        self.indices = list(indices) if indices is not None else list(range(len(hf_dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        item = self._hf[self.indices[idx]]
        image = _transform(item["image"])
        return image, item["label"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(name: str = "svhn") -> SVHNDataset:
    """HuggingFace에서 SVHN 학습 split을 로드하여 PyTorch Dataset으로 반환.

    Args:
        name: 데이터셋 이름. 현재 "svhn"만 지원.

    Returns:
        전체 SVHN 학습 데이터를 감싼 SVHNDataset.
    """
    if name != "svhn":
        raise ValueError(f"지원하지 않는 데이터셋: {name!r}. 현재 'svhn'만 지원.")
    hf_ds = hf_load_dataset("svhn", "cropped_digits", split="train")
    return SVHNDataset(hf_ds)


def split_into_shards(
    dataset: SVHNDataset,
    num_shards: int = 20,
    seed: int = 42,
) -> List[SVHNDataset]:
    """전체 데이터셋을 num_shards개의 균등한 shard로 랜덤 분할.

    Args:
        dataset: 분할할 전체 SVHNDataset.
        num_shards: 분할할 shard 수 (기본값 20).
        seed: 셔플 재현성을 위한 시드.

    Returns:
        num_shards개의 SVHNDataset 리스트. 각 shard 크기는 최대 1씩 차이.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    permuted = rng.permutation(n).tolist()

    shard_size = math.ceil(n / num_shards)
    shards: List[SVHNDataset] = []
    for i in range(num_shards):
        chunk = permuted[i * shard_size: (i + 1) * shard_size]
        original_indices = [dataset.indices[j] for j in chunk]
        shards.append(SVHNDataset(dataset._hf, original_indices))
    return shards


def split_into_slices(
    shard: SVHNDataset,
    num_slices: int = 50,
) -> List[SVHNDataset]:
    """shard를 num_slices개의 slice로 순서대로 균등 분할.

    셔플 없이 shard 내 순서를 유지한 채 앞에서부터 잘라낸다.

    Args:
        shard: 분할할 SVHNDataset shard.
        num_slices: 분할할 slice 수 (기본값 50).

    Returns:
        num_slices개 이하의 SVHNDataset 리스트 (마지막이 더 작을 수 있음).
    """
    n = len(shard)
    slice_size = math.ceil(n / num_slices)
    slices: List[SVHNDataset] = []
    for i in range(num_slices):
        start = i * slice_size
        end = min(start + slice_size, n)
        if start >= n:
            break
        slices.append(SVHNDataset(shard._hf, shard.indices[start:end]))
    return slices


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """Dataset에 대한 DataLoader를 반환.

    Args:
        dataset: PyTorch Dataset.
        batch_size: 배치 크기 (기본값 128).
        shuffle: 배치 순서 셔플 여부.
        num_workers: 데이터 로딩 워커 수.

    Returns:
        설정된 DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@dataclass
class SISAConfig:
    num_shards: int = 5
    num_slices: int = 4
    seed: int = 42
    shards_dir: str = "shards"


class SISADataset:
    """전체 SVHN 데이터를 shard × slice 구조로 관리.

    slice_indices[shard][slice]는 누적 인덱스 리스트다:
    slice r은 raw slice 0..r의 샘플을 모두 포함한다.
    """

    def __init__(self, config: SISAConfig):
        self.config = config
        self.shards_dir = Path(config.shards_dir)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        hf_train = hf_load_dataset("svhn", "cropped_digits", split="train")
        hf_test = hf_load_dataset("svhn", "cropped_digits", split="test")
        self._hf_train = hf_train
        self.train_dataset = SVHNDataset(hf_train)
        self.test_dataset = SVHNDataset(hf_test)

        self.shard_indices: list[list[int]] = []
        self.slice_indices: list[list[list[int]]] = []

    def build_shards(self) -> None:
        """전체 훈련 데이터를 S개 shard로 분할하고 각 shard를 R개 누적 slice로 구성."""
        shards = split_into_shards(
            self.train_dataset, self.config.num_shards, self.config.seed
        )
        self.shard_indices = [s.indices for s in shards]
        self.slice_indices = [self._make_cumulative_slices(s) for s in shards]
        self._save_shard_metadata()
        build_point_to_shard_map(
            shards,
            save_path=str(self.shards_dir / "point_to_shard.json"),
        )

    def _make_cumulative_slices(self, shard: "SVHNDataset") -> list[list[int]]:
        raw_slices = split_into_slices(shard, self.config.num_slices)
        cumulative, accumulated = [], []
        for sl in raw_slices:
            accumulated = accumulated + sl.indices
            cumulative.append(list(accumulated))
        return cumulative

    def get_slice_dataset(self, shard_idx: int, slice_idx: int) -> "SVHNDataset":
        return SVHNDataset(self._hf_train, self.slice_indices[shard_idx][slice_idx])

    def get_forget_shards(self, forget_indices: list[int]) -> list[int]:
        forget_set = set(forget_indices)
        return [i for i, shard in enumerate(self.shard_indices) if forget_set & set(shard)]

    def remove_and_rebuild_shard(self, shard_idx: int, forget_indices: list[int]) -> None:
        forget_set = set(forget_indices)
        self.shard_indices[shard_idx] = [
            i for i in self.shard_indices[shard_idx] if i not in forget_set
        ]
        shard_ds = SVHNDataset(self._hf_train, self.shard_indices[shard_idx])
        self.slice_indices[shard_idx] = self._make_cumulative_slices(shard_ds)
        self._save_shard_metadata()

    def _save_shard_metadata(self) -> None:
        with open(self.shards_dir / "metadata.json", "w") as f:
            json.dump({
                "num_shards": self.config.num_shards,
                "num_slices": self.config.num_slices,
                "shard_indices": self.shard_indices,
                "slice_indices": self.slice_indices,
            }, f)

    def load_shard_metadata(self) -> bool:
        path = self.shards_dir / "metadata.json"
        if not path.exists():
            return False
        with open(path) as f:
            meta = json.load(f)
        self.shard_indices = meta["shard_indices"]
        self.slice_indices = meta["slice_indices"]
        return True


def build_point_to_shard_map(
    shards: List[SVHNDataset],
    save_path: str = "shards/point_to_shard.json",
) -> Dict[int, int]:
    """원본 데이터 인덱스 → shard 인덱스 매핑을 생성하고 JSON으로 저장.

    Args:
        shards: split_into_shards()가 반환한 shard 리스트.
        save_path: JSON 저장 경로.

    Returns:
        {data_index: shard_index} 딕셔너리.
    """
    mapping: Dict[int, int] = {}
    for shard_idx, shard in enumerate(shards):
        for data_idx in shard.indices:
            mapping[data_idx] = shard_idx

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON 키는 문자열만 허용하므로 int → str 변환 후 저장
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f)

    return mapping
