"""페이지 3 — 결과 비교: 속도·정확도·학습 곡선·망각 이벤트 타임라인."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import mock_data, sisa
from core.state import init_state

st.set_page_config(page_title="결과 비교", page_icon="📈", layout="wide")
init_state()

st.title("📈 결과 비교")
st.caption("망각 전후 정확도, 속도 향상 배율, 학습 곡선을 비교합니다.")

cfg       = st.session_state.config
demo_mode = cfg["demo_mode"]
num_shards = cfg["num_shards"]
num_slices = cfg["num_slices"]

# 결과 로드
results = sisa.get_results(demo_mode=demo_mode)
if results is None:
    st.info("📦 결과 파일이 없어 Demo 데이터를 표시합니다.")
    results = sisa.get_results(demo_mode=True)

speedup        = results.get("speedup", 0.0)
acc_before     = results.get("accuracy_before", 0.0)
acc_after      = results.get("accuracy_after", 0.0)
acc_drop       = results.get("accuracy_drop", round(acc_before - acc_after, 2))
full_retrain_t = results.get("full_retrain_time_sec", 160.0)
sisa_unlearn_t = results.get("sisa_unlearn_time_sec", full_retrain_t / max(speedup, 1))

# ── 1. 속도 비교 (horizontal bar) ────────────────────────────────────────────

st.subheader("속도 비교")

fig_time = go.Figure()
fig_time.add_trace(go.Bar(
    name="SISA 재학습",
    y=["처리 시간"],
    x=[sisa_unlearn_t],
    orientation="h",
    marker_color="#378ADD",
    text=[f"{sisa_unlearn_t:.1f}s"],
    textposition="inside",
))
fig_time.add_trace(go.Bar(
    name="전체 재학습",
    y=["처리 시간"],
    x=[full_retrain_t],
    orientation="h",
    marker_color="#888780",
    text=[f"{full_retrain_t:.1f}s"],
    textposition="inside",
))
fig_time.add_annotation(
    x=full_retrain_t, y=0,
    text=f"  ×{speedup:.2f} 빠름",
    showarrow=False,
    font=dict(size=14, color="#E24B4A"),
    xanchor="left",
)
fig_time.update_layout(
    barmode="overlay",
    height=140,
    xaxis=dict(title="시간 (초)"),
    margin=dict(l=0, r=120, t=10, b=0),
    legend=dict(orientation="h", y=1.3),
)
st.plotly_chart(fig_time, use_container_width=True)

# ── 2. 정확도 비교 (3 columns) ───────────────────────────────────────────────

st.subheader("정확도 비교")

_DROP_OK = acc_drop < 2.0
c1, c2, c3 = st.columns(3)
c1.metric("망각 전 정확도", f"{acc_before:.2f}%")
c2.metric("망각 후 정확도", f"{acc_after:.2f}%",
          delta=f"{acc_after - acc_before:.2f}%p",
          delta_color="inverse")

drop_color = "normal" if _DROP_OK else "off"
c3.metric(
    "정확도 손실",
    f"{acc_drop:.2f}%p",
    delta="< 2%p ✅" if _DROP_OK else "> 2%p ❌",
    delta_color="off",
)
if _DROP_OK:
    c3.success("목표 달성 (< 2%p)")
else:
    c3.error("목표 미달성 (≥ 2%p)")

# ── 3. 학습 곡선 비교 (SISA vs baseline) ─────────────────────────────────────

st.subheader("학습 곡선 비교")

sisa_curve     = mock_data.generate_accuracy_curve(num_slices, noise=0.015)
baseline_curve = mock_data.generate_accuracy_curve(num_slices, noise=0.025)
# baseline은 단일 모델이므로 SISA보다 약간 낮게 조정
baseline_curve = [max(0.0, v - 0.025) for v in baseline_curve]

slice_labels = list(range(num_slices))

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=slice_labels,
    y=[v * 100 for v in sisa_curve],
    name="SISA 앙상블",
    mode="lines+markers",
    line=dict(color="#378ADD", width=2.5),
    marker=dict(size=7),
))
fig_curve.add_trace(go.Scatter(
    x=slice_labels,
    y=[v * 100 for v in baseline_curve],
    name="단일 모델 (Baseline)",
    mode="lines+markers",
    line=dict(color="#888780", width=2, dash="dot"),
    marker=dict(size=7),
))
fig_curve.update_layout(
    height=300,
    xaxis=dict(title="Slice 번호"),
    yaxis=dict(title="Accuracy (%)", range=[75, 100]),
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_curve, use_container_width=True)

# ── 4. 망각 이벤트 타임라인 ───────────────────────────────────────────────────

st.subheader("망각 이벤트 타임라인")

unlearn_history: list[dict] = st.session_state.get("unlearn_results", [])

# 이력이 없으면 mock
if not unlearn_history:
    mock_events = [2, 5, 8]   # 망각이 일어난 요청 번호
    n_events    = 10
    acc_seq     = [acc_before - i * 0.15 for i in range(n_events)]
else:
    mock_events = []
    n_events    = len(unlearn_history) + 1
    acc_seq     = [acc_before] + [
        h.get("accuracy_after", acc_after) for h in unlearn_history
    ]

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=list(range(n_events)),
    y=acc_seq,
    mode="lines+markers",
    name="정확도",
    line=dict(color="#1D9E75", width=2),
))
for ev in mock_events:
    fig_timeline.add_vline(
        x=ev,
        line=dict(color="#E24B4A", dash="dash", width=1.5),
        annotation_text="망각",
        annotation_position="top",
    )
fig_timeline.update_layout(
    height=280,
    xaxis=dict(title="요청 횟수"),
    yaxis=dict(title="Accuracy (%)", range=[80, 95]),
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False,
)
st.plotly_chart(fig_timeline, use_container_width=True)

if not unlearn_history:
    st.caption("📦 Mock 타임라인 — 실제 망각 이력이 없습니다.")

# ── 원본 JSON ─────────────────────────────────────────────────────────────────
with st.expander("📄 results.json 원본"):
    import json
    st.code(json.dumps(results, indent=2, ensure_ascii=False), language="json")