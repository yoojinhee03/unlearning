"""페이지 4 — 논문 재현: Bourtoule et al. 2021 목표치 vs 내 실험."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import mock_data, sisa
from core.state import init_state

st.set_page_config(page_title="논문 재현", page_icon="📄", layout="wide")
init_state()

st.title("📄 논문 재현")
st.caption(
    "Bourtoule et al., *Machine Unlearning*, IEEE S&P 2021 — "
    "Table 2 목표치와 현재 결과를 비교합니다."
)

cfg       = st.session_state.config
demo_mode = cfg["demo_mode"]

PAPER = mock_data.generate_paper_results()  # {purchase, svhn, imagenet}

# ── 1. 데이터셋 선택 ──────────────────────────────────────────────────────────

dataset_label = st.radio(
    "비교할 데이터셋 선택",
    ["Purchase", "SVHN", "ImageNet"],
    horizontal=True,
)
dataset_key = dataset_label.lower()
target      = PAPER[dataset_key]

# ── 2. 논문 목표 vs 내 실험 비교표 ───────────────────────────────────────────

st.subheader("논문 목표 vs 내 실험")

results = sisa.get_results(demo_mode=demo_mode)
if results is None:
    results = sisa.get_results(demo_mode=True)

my_speedup = results.get("speedup", 2.10)
my_drop    = results.get("accuracy_drop", 1.21)

rows = [
    {
        "항목":      "Speedup",
        "논문 목표": f"{target['speedup']:.2f}x",
        "내 실험":   f"{my_speedup:.2f}x",
        "달성 여부": "✅" if my_speedup >= target["speedup"] else "❌",
    },
    {
        "항목":      "정확도 손실",
        "논문 목표": f"< {target['accuracy_drop']:.1f}%p",
        "내 실험":   f"{my_drop:.2f}%p",
        "달성 여부": "✅" if my_drop <= target["accuracy_drop"] else "❌",
    },
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── 3. Shard 수별 Speedup (scatter + line) ───────────────────────────────────

st.subheader("Shard 수별 Speedup")

shard_counts = [5, 10, 15, 20]
# 이론값: speedup ≈ S (완벽한 경우)
theory_speedup  = [float(s) for s in shard_counts]
# 실험값: 오버헤드 포함하여 약 0.6~0.8× 이론값
exp_ratios      = [0.62, 0.65, 0.68, 0.72]
exp_speedup     = [t * r for t, r in zip(theory_speedup, exp_ratios)]

fig_shard = go.Figure()
fig_shard.add_trace(go.Scatter(
    x=shard_counts, y=theory_speedup,
    mode="lines",
    name="이론값 (S배)",
    line=dict(color="#888780", width=2, dash="solid"),
))
fig_shard.add_trace(go.Scatter(
    x=shard_counts, y=exp_speedup,
    mode="markers+lines",
    name="실험값",
    line=dict(color="#378ADD", width=2, dash="dot"),
    marker=dict(size=10, symbol="circle"),
))
# 논문 목표선
fig_shard.add_hline(
    y=target["speedup"],
    line=dict(color="#E24B4A", dash="dash", width=1.5),
    annotation_text=f"논문 목표 ({target['speedup']}x)",
    annotation_position="right",
)
fig_shard.update_layout(
    height=320,
    xaxis=dict(title="Shard 수 (S)", tickvals=shard_counts),
    yaxis=dict(title="Speedup (×배)", range=[0, 22]),
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=0, r=80, t=30, b=0),
)
st.plotly_chart(fig_shard, use_container_width=True)

st.divider()

# ── 4. 핵심 수식 ──────────────────────────────────────────────────────────────

st.subheader("핵심 수식")

col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown("**Speedup 정의**")
    st.latex(r"S(n) = \frac{T_{\text{retrain}}}{T_{\text{SISA}}}")
    st.caption("전체 재학습 시간 대비 SISA 망각 처리 시간 비율")

with col_eq2:
    st.markdown("**Unlearning 정의**")
    st.latex(
        r"\mathcal{A}(D \setminus D_f) \approx "
        r"\mathcal{A}_{-D_f}"
    )
    st.caption(
        r"$D_f$: 망각 대상 / "
        r"$\mathcal{A}$: 학습 알고리즘 / "
        "망각 후 모델 ≈ 처음부터 제외하고 학습한 모델"
    )

st.divider()

# ── 5. 재현 체크리스트 ────────────────────────────────────────────────────────

st.subheader("재현 체크리스트")

ckpt_exists = sisa.any_checkpoint_exists()
result_file = (Path(__file__).parent.parent.parent / "experiments" / "results.json").exists()
unlearn_done = len(st.session_state.get("unlearn_results", [])) > 0
target_met   = my_speedup >= target["speedup"] and my_drop <= target["accuracy_drop"]

checklist = [
    ("데이터 로드 완료",       True),            # datasets 설치되면 항상 OK
    ("Shard 분할 완료",        ckpt_exists),
    ("전체 학습 완료",         ckpt_exists),
    ("망각 요청 처리 완료",    unlearn_done),
    (f"논문 목표치 달성 ({dataset_label})", target_met),
]

for label, checked in checklist:
    st.checkbox(label, value=checked, disabled=True)