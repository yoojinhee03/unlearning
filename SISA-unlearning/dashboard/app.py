"""SISA Unlearning Dashboard — 홈 페이지."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from core.state import init_state, log_experiment

st.set_page_config(
    page_title="SISA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SISA Training")
    st.caption(
        "Sharded · Isolated · Sliced · Aggregated  \n"
        "[Bourtoule et al., IEEE S&P 2021]"
        "(https://arxiv.org/abs/1912.03817)"
    )
    st.divider()

    cfg = st.session_state.config

    _DATASETS = ["SVHN", "Purchase", "ImageNet"]
    _current  = cfg.get("dataset", "SVHN").upper()
    _ds_map   = {d.upper(): d for d in _DATASETS}
    _default  = _ds_map.get(_current, "SVHN")
    cfg["dataset"] = st.selectbox(
        "데이터셋", _DATASETS, index=_DATASETS.index(_default)
    )
    cfg["num_shards"] = st.slider(
        "Shard 수 (S)", min_value=5, max_value=20,
        value=cfg.get("num_shards", 5), step=1,
    )
    cfg["num_slices"] = st.slider(
        "Slice 수 (R)", min_value=10, max_value=50,
        value=cfg.get("num_slices", 10), step=5,
    )
    cfg["batch_size"] = st.number_input(
        "Batch size", min_value=16, max_value=512,
        value=cfg.get("batch_size", 128), step=16,
    )
    cfg["demo_mode"] = st.toggle("Demo 모드", value=cfg.get("demo_mode", True))
    st.session_state.config = cfg

    st.divider()

    if st.button("▶️ 실험 시작", type="primary", use_container_width=True):
        log_experiment({
            "event":      "experiment_start",
            "num_shards": cfg["num_shards"],
            "num_slices": cfg["num_slices"],
            "dataset":    cfg["dataset"],
        })
        st.success("실험이 큐에 추가되었습니다. 모니터링 페이지를 확인하세요.")

    if cfg["demo_mode"]:
        st.info("📦 Demo 모드  \n실제 학습 없이 시뮬레이션합니다.")
    else:
        st.warning("⚡ 실제 모드  \n학습/망각 시 실제 연산이 수행됩니다.")

# ── 메인 ──────────────────────────────────────────────────────────────────────
st.title("🧠 SISA Unlearning Dashboard")
st.caption(
    "Sharded · Isolated · Sliced · Aggregated Training — "
    "Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021"
)

st.divider()

# 소개 카드
with st.container(border=True):
    st.markdown(
        """
        **SISA**는 머신 러닝 모델에서 특정 학습 데이터를 효율적으로 제거하는 기법입니다.

        | 개념 | 설명 |
        |------|------|
        | **Sharding** | 전체 데이터를 S개의 독립 그룹으로 분할 |
        | **Slicing** | 각 shard를 R개의 누적 slice로 재분할 |
        | **Isolation** | 각 shard는 별도 모델로 학습 |
        | **Aggregation** | Majority vote로 예측 앙상블 |

        망각 요청 시 해당 데이터가 속한 shard만 재학습하므로, 전체 재학습 대비 **최대 1/S** 비용으로 망각이 가능합니다.
        """
    )

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    with st.container(border=True):
        st.markdown("### 📊 모니터링")
        st.caption("학습 진행 상황, shard 상태, loss/accuracy 곡선을 확인합니다.")
with col2:
    with st.container(border=True):
        st.markdown("### 🗑️ 망각 요청")
        st.caption("삭제할 인덱스를 입력하고 SISA 망각 파이프라인을 실행합니다.")
with col3:
    with st.container(border=True):
        st.markdown("### 📈 결과 비교")
        st.caption("망각 전후 정확도, 속도 향상, 학습 곡선을 비교합니다.")
with col4:
    with st.container(border=True):
        st.markdown("### 📄 논문 재현")
        st.caption("Bourtoule 2021 목표치와 현재 결과를 비교합니다.")

st.divider()

# 현재 설정 요약
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dataset",   cfg["dataset"])
c2.metric("Shards (S)", cfg["num_shards"])
c3.metric("Slices (R)", cfg["num_slices"])
c4.metric("Batch size", cfg["batch_size"])
c5.metric("Mode", "Demo" if cfg["demo_mode"] else "Real")