# SISA Unlearning

SISA(Sharded, Isolated, Sliced, Aggregated) 기법으로 Machine Unlearning을 구현한 프로젝트.
특정 학습 샘플을 "망각"시킬 때 전체 재학습 없이 해당 shard만 재학습해 비용을 ~1/S로 줄인다.
Streamlit 대시보드로 학습 모니터링, 망각 요청, 결과 비교, 논문 재현을 시각적으로 수행할 수 있다.

> Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021 — [arXiv:1912.03817](https://arxiv.org/abs/1912.03817)

---

## 설치

```bash
# uv 권장
uv sync

# 또는 pip
pip install -r requirements.txt
```

---

## 실행

| 명령 | 설명 |
|------|------|
| `make demo`  | Mock 데이터로 대시보드 확인 (학습 불필요) |
| `make run`   | 실제 체크포인트와 연동된 대시보드 실행 |
| `make train` | SISA 학습 실행 (`python -m src.train`) |
| `make unlearn INDICES='0 1 2'` | 지정 인덱스 망각 처리 |
| `make evaluate` | 앙상블 정확도 측정 |

```bash
# 1. Mock 모드로 대시보드 바로 확인
make demo

# 2. 실제 학습 후 대시보드 연동
make train
make run

# 3. CLI로 전체 파이프라인
python experiments/run_experiment.py \
  --dataset svhn \
  --num_shards 10 \
  --num_slices 20 \
  --unlearn_indices 42 100 200
```

---

## 대시보드 탭

### 📊 모니터링
- Shard 상태 그리드 (done / training / retraining / idle)
- Slice별 loss·accuracy 학습 곡선
- 완료 shard 수, 현재 정확도, 경과 시간, 대기 망각 요청 수
- 자동 새로고침 (5초)

### 🗑️ 망각 요청
- 쉼표 구분 인덱스 입력 → 큐 관리 (추가/삭제)
- 영향받는 shard 자동 분석 및 막대 차트
- 예상 speedup 표시
- 망각 실행 및 이력 테이블

### 📈 결과 비교
- SISA vs 전체 재학습 수평 막대 그래프 + speedup annotation
- 망각 전후 정확도 3-column 비교 (손실 <2%p 초록/빨강)
- SISA 앙상블 vs 단일 모델 학습 곡선 비교
- 망각 이벤트 타임라인 (수직선 표시)

### 📄 논문 재현
- 데이터셋 선택 (Purchase / SVHN / ImageNet)
- 논문 목표 vs 내 실험 비교표 (✅/❌)
- Shard 수별 speedup scatter+line 차트
- 핵심 수식 (`st.latex`)
- 재현 체크리스트

---

## 폴더 구조

```
sisa-unlearning/
  main.py                    CLI 진입점 (train / unlearn / evaluate)
  Makefile                   make demo / run / train
  experiments/
    run_experiment.py        전체 파이프라인 한 번에 실행
  src/
    dataset.py               SISAConfig, SISADataset, 데이터 로드·분할
    model.py                 SimpleCNN, save/load_checkpoint, save/load_model
    train.py                 train_shard(), train_all_shards()
    unlearn.py               unlearn_request(), compare_unlearn_time()
    evaluate.py              evaluate_sisa(), save_results()
  dashboard/
    app.py                   홈 + 사이드바 (dataset, shards, slices, demo 모드)
    pages/
      1_모니터링.py            Shard 그리드, 학습 곡선, 자동 새로고침
      2_망각요청.py            인덱스 입력, 영향 분석, 망각 실행, 이력
      3_결과비교.py            속도·정확도·학습곡선·이벤트 타임라인
      4_논문재현.py            목표 비교표, shard별 speedup, 수식, 체크리스트
    core/
      state.py               init_state(), update_shard_status() 등
      mock_data.py           Demo 모드용 시뮬레이션 데이터
      sisa.py                대시보드 ↔ src/ 브리지 (demo_mode 분기)
  shards/
    metadata.json            shard/slice 인덱스 (train 후 생성)
    point_to_shard.json      데이터 인덱스 → shard 매핑
  checkpoints/
    shard{i}_slice{j}.pt     체크포인트 (model_state_dict + 메타데이터)
  experiments/
    results.json             최종 측정 지표
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

## 출력 예시

```
[Step 4] 평가 및 결과 출력
  SISA Ensemble Accuracy: 87.21%  (20/20 shards)
  Accuracy on forgotten samples: 18.30%  (낮을수록 망각 성공)

  Results saved → experiments/results.json

  ──────────────────────────────────────────
  Accuracy before unlearning :  87.34%
  Accuracy after  unlearning :  87.21%
  Change                     : ▼ 0.13%p
  ──────────────────────────────────────────

  Paper Comparison — Bourtoule et al. (2021)
  Dataset    Target Speedup  Achieved   Target Drop  Achieved
  SVHN           2.45x        2.10x FAIL   2.0%p    0.13%p PASS
```

```json
// experiments/results.json
{
  "speedup": 2.10,
  "accuracy_before": 87.34,
  "accuracy_after": 87.21,
  "accuracy_drop": 0.13,
  "full_retrain_time_sec": 142.30,
  "sisa_unlearn_time_sec": 12.40,
  "forget_accuracy": 18.30
}
```

---

## 참고

- 논문 목표 (SVHN): **2.45x speedup**, **<2%p accuracy drop**
- CPU 환경: `--num_shards 3 --num_slices 4 --epochs_per_slice 2` 로 빠르게 테스트
- `shards/metadata.json` 없이 `unlearn`/`evaluate` 실행 시 오류 → 먼저 `train` 실행
- 논문 원문: [arXiv:1912.03817](https://arxiv.org/abs/1912.03817)