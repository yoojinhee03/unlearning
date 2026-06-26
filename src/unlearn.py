"""
SISA 망각(unlearning) 요청 처리.

forget 요청이 들어오면:
1. 해당 샘플이 포함된 shard를 찾는다.
2. shard에서 해당 샘플을 제거한다.
3. 영향받은 shard만 첫 등장 slice부터 재학습한다.
"""

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


def process_forget_request(
    forget_indices: list[int],
    sisa_dataset: SISADataset,
    config: SISAConfig,
    **train_kwargs,
) -> dict[int, int]:
    """forget 요청 처리 파이프라인.

    Returns:
        {shard_idx: start_slice} — 재학습이 수행된 shard와 시작 slice 정보.
    """
    affected_shards = sisa_dataset.get_forget_shards(forget_indices)
    print(f"Forget request: {len(forget_indices)} samples "
          f"→ affects shards {affected_shards}")

    retrain_info: dict[int, int] = {}
    for shard_idx in affected_shards:
        start_slice = find_retraining_start(shard_idx, forget_indices, sisa_dataset)
        sisa_dataset.remove_and_rebuild_shard(shard_idx, forget_indices)

        print(f"\n=== Retraining Shard {shard_idx} from Slice {start_slice} ===")
        train_shard(
            shard_idx,
            sisa_dataset,
            config,
            slice_range=(start_slice, config.num_slices - 1),
            **train_kwargs,
        )
        retrain_info[shard_idx] = start_slice

    return retrain_info