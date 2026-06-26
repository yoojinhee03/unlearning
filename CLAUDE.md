# SISA Unlearning

Machine unlearning 구현체. SVHN 데이터셋에 SISA(Sharded, Isolated, Sliced, Aggregated) 기법을 적용해 특정 학습 샘플을 효율적으로 "망각"시킨다.

## 핵심 개념

SISA는 전체 데이터를 **S개 shard**로 나누고, 각 shard를 **R개 slice**로 나눠 incremental하게 학습한다. Forget 요청이 오면 해당 샘플이 속한 shard만 재학습하면 되므로, 전체 재학습 대비 비용이 1/S로 줄어든다.

```
전체 데이터
  └─ shard 0 ─ slice 0 (누적) → ckpt: shard_0/slice_0.pt
  │            slice 1 (누적) → ckpt: shard_0/slice_1.pt
  │            ...
  └─ shard 1 ─ ...
```

Slice는 누적 구조: slice r은 raw slice 0..r의 샘플을 모두 포함한다.

평가 시에는 모든 shard의 마지막 slice 체크포인트를 로드해 logit을 평균 내어 앙상블한다.

## 실행 방법

```bash
# 의존성 설치 (uv 사용)
uv sync

# 전체 학습 (shard 5개, slice 4개)
python main.py train --num-shards 5 --num-slices 4

# 망각 요청 (인덱스 0~99 제거)
python main.py unlearn --forget-indices $(seq 0 99) --num-shards 5 --num-slices 4

# 평가만
python main.py evaluate

# wandb 로깅 활성화
python main.py train --use-wandb
```

## 아키텍처

```
main.py          진입점. argparse로 train/unlearn/evaluate 모드 분기
src/
  dataset.py     SISAConfig, SISADataset, SVHNDataset + 저수준 분할 함수
  model.py       get_model() — resnet18 또는 SimpleCNN
  train.py       train_shard(), train_all_shards()
  unlearn.py     process_forget_request(), retrain_shard_from_slice()
  evaluate.py    evaluate_sisa() (앙상블), evaluate_forget_efficacy()
shards/
  metadata.json  shard_indices, slice_indices (빌드 후 생성)
checkpoints/
  shard_N/
    slice_M.pt   각 shard의 slice별 모델 가중치
```

## 주요 클래스/함수

| 심볼 | 위치 | 역할 |
|------|------|------|
| `SISAConfig` | `dataset.py` | num_shards, num_slices, seed 등 설정 |
| `SISADataset` | `dataset.py` | shard×slice 구조 관리, 메타데이터 저장/로드 |
| `SVHNDataset` | `dataset.py` | HuggingFace SVHN 래퍼. 인덱스 슬라이싱으로 뷰 생성 |
| `split_into_shards()` | `dataset.py` | 랜덤 셔플 후 균등 분할 |
| `split_into_slices()` | `dataset.py` | 순서 유지 균등 분할 (비누적) |
| `get_model()` | `model.py` | `resnet18` 또는 `simple_cnn` |
| `train_shard()` | `train.py` | 단일 shard incremental 학습 |
| `process_forget_request()` | `unlearn.py` | forget 파이프라인 전체 조율 |
| `evaluate_sisa()` | `evaluate.py` | 앙상블 정확도 측정 |
| `evaluate_forget_efficacy()` | `evaluate.py` | forgotten 샘플에 대한 예측 신뢰도 |

## 데이터 흐름

```
HuggingFace SVHN (train split)
  → SVHNDataset (전체)
  → split_into_shards() → [SVHNDataset, ...] × S
  → split_into_slices() + 누적 변환 → slice_indices[shard][slice]
  → SISADataset.get_slice_dataset(shard_idx, slice_idx)
  → DataLoader → train_one_epoch()
  → checkpoints/shard_N/slice_M.pt
```

Forget 흐름:

```
forget_indices (원본 HF 인덱스)
  → SISADataset.get_forget_shards()  — 영향받은 shard 탐색
  → find_retraining_start()          — 첫 등장 slice 탐색
  → SISADataset.remove_and_rebuild_shard()  — 인덱스에서 제거 후 slice 재구성
  → retrain_shard_from_slice()       — 해당 slice부터 재학습
```

## 의존성

| 패키지 | 역할 |
|--------|------|
| `torch >= 2.2` | 학습/추론 엔진 |
| `torchvision >= 0.17` | ResNet18, 이미지 transforms |
| `datasets >= 2.18` | HuggingFace SVHN 로드 |
| `numpy >= 1.26` | shard 분할용 RNG |
| `tqdm >= 4.66` | 진행 표시 |
| `wandb >= 0.16` | 실험 로깅 (선택) |

Dev: `pytest`, `ruff`

## 주의사항

- `forget_indices`는 HuggingFace 원본 SVHN train split의 0-based 인덱스다.
- `slice_indices[shard][slice]`는 누적 구조이므로 unlearn 시 `find_retraining_start()`가 첫 등장 slice만 찾으면 된다.
- GPU 없이도 동작하지만 CPU에서는 매우 느리다. `simple_cnn` arch가 실험용으로 빠르다.
- `shards/metadata.json`이 없으면 `unlearn`/`evaluate` 모드가 즉시 종료된다. 먼저 `train`을 실행해야 한다.
- `train_all_shards()`는 현재 순차 실행이다. 멀티 GPU 환경에서는 병렬화 여지가 있다.

## 자주 하는 작업

**새 아키텍처 추가**
[src/model.py](src/model.py)의 `get_model()`에 `elif arch == "my_arch"` 분기 추가.
`main.py`의 `TRAIN_KWARGS["arch"]` 값을 바꾸면 즉시 적용.

**shard/slice 수 변경**
`main.py`의 `--num-shards`, `--num-slices` 인자로 제어.
변경 시 기존 `shards/metadata.json`과 `checkpoints/`를 삭제하고 처음부터 `train`해야 한다.

**특정 shard만 재학습**
`train.py`의 `train_shard(shard_idx, ...)` 직접 호출.

**망각 효과 확인**
```bash
python main.py unlearn --forget-indices 0 1 2
# → Accuracy on forgotten samples: X% (낮을수록 망각 성공)
```

**wandb 실험 추적**
```bash
python main.py train --use-wandb
# wandb.ai에서 shard별 loss/acc 확인
```

## 테스트/린트

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
```