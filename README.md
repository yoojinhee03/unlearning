# SISA Unlearning

SISA(Sharded, Isolated, Sliced, Aggregated) 기법으로 Machine Unlearning을 구현한 프로젝트.  
특정 학습 샘플을 "망각"시킬 때 전체 재학습 없이 해당 shard만 재학습해 비용을 ~1/S로 줄인다.

> Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021

---

## 설치

```bash
# 의존성 설치 (uv 권장)
uv sync

# 또는 pip
pip install -r requirements.txt
```

---

## 빠른 시작

### 전체 파이프라인 한 번에 실행

```bash
# 기본값: SVHN, shard 20개, slice 50개, 인덱스 0~4 망각
python experiments/run_experiment.py

# 옵션 지정
python experiments/run_experiment.py \
  --dataset svhn \
  --num_shards 10 \
  --num_slices 20 \
  --unlearn_indices 42 100 200 \
  --epochs_per_slice 5 \
  --results_path experiments/results.json
```

### wandb 로깅 활성화

```bash
python experiments/run_experiment.py \
  --num_shards 20 \
  --num_slices 50 \
  --wandb_project sisa-unlearning-svhn
```

---

## 단계별 실행 (`main.py`)

### 1. 학습

```bash
# shard 5개, slice 4개로 SVHN 학습
python main.py train --num-shards 5 --num-slices 4

# wandb 포함
python main.py train --num-shards 5 --num-slices 4 --use-wandb
```

### 2. 망각 요청

```bash
# 인덱스 0~9 제거
python main.py unlearn --forget-indices 0 1 2 3 4 5 6 7 8 9

# 범위를 셸 확장으로 지정
python main.py unlearn --forget-indices $(seq 0 99)
```

### 3. 평가

```bash
python main.py evaluate
```

---

## 출력 예시

```
[Step 3] 망각 요청 처리
  Forget indices (3 samples): [42, 100, 200]
  Forget request: 3 samples → affects shards [1]
  === Retraining Shard 1 from Slice 7 ===
  Unlearning time : 12.4s

[Step 4] 평가 및 결과 출력
  SISA Ensemble Accuracy: 87.21%  (20/20 shards)
  Accuracy on forgotten samples: 18.30%  (lower → better unlearning)

  Results saved → experiments/results.json

  ──────────────────────────────────────────
  Accuracy before unlearning :  87.34%
  Accuracy after  unlearning :  87.21%
  Change                     : ▼ 0.13%p
  ──────────────────────────────────────────

  ==========================================================================
    Paper Comparison — Bourtoule et al. (2021), SISA Training
  ==========================================================================
    Dataset       Target Speedup    Achieved        Target Drop    Achieved
    Purchase          4.63x         2.10x    FAIL      2.0%p      0.13%p    PASS
    SVHN              2.45x         2.10x    FAIL      2.0%p      0.13%p    PASS
```

### results.json

```json
{
  "full_retrain_time_sec": 142.30,
  "sisa_unlearn_time_sec": 12.40,
  "speedup": 2.10,
  "accuracy_before_pct": 87.34,
  "accuracy_after_pct": 87.21,
  "accuracy_drop_pp": 0.13,
  "forget_sample_accuracy_pct": 18.30
}
```

---

## 프로젝트 구조

```
sisa-unlearning/
  main.py                    단계별 CLI (train / unlearn / evaluate)
  experiments/
    run_experiment.py        전체 파이프라인 한 번에 실행
  src/
    dataset.py               SISAConfig, SISADataset, 데이터 로드·분할
    model.py                 SimpleCNN, save/load_checkpoint
    train.py                 train_shard(), train_all_shards()
    unlearn.py               process_forget_request()
    evaluate.py              evaluate_sisa(), save_results()
  shards/
    metadata.json            shard/slice 인덱스 (train 후 생성)
    point_to_shard.json      데이터 인덱스 → shard 매핑
  checkpoints/
    shard{i}_slice{j}.pt     체크포인트 (model_state_dict + 메타데이터)
  results.json               최종 측정 지표
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--num_shards` | 20 | 데이터를 나눌 shard 수. 클수록 망각 속도 ↑, 정확도 ↓ |
| `--num_slices` | 50 | 각 shard를 나눌 slice 수. 클수록 재학습 범위 ↓ |
| `--epochs_per_slice` | 10 | slice당 학습 epoch 수 |
| `--unlearn_indices` | `0 1 2 3 4` | 망각 요청할 데이터 인덱스 |
| `--wandb_project` | None | 지정 시 wandb 로깅 활성화 |

---

## 참고

- 논문 목표 (SVHN): **2.45x speedup**, **<2%p accuracy drop**
- CPU 환경에서는 `--num_shards 3 --num_slices 4 --epochs_per_slice 2` 로 빠르게 테스트 가능
- `shards/metadata.json` 없이 `unlearn`/`evaluate` 실행 시 오류 → 먼저 `train` 실행