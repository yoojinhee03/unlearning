"""페이지 1 — 학습 모니터링."""

import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import mock_data, sisa
from core.state import init_state

st.set_page_config(page_title="모니터링", page_icon="📊", layout="wide")
init_state()

st.title("📊 학습 모니터링")

cfg        = st.session_state.config
num_shards = cfg["num_shards"]
num_slices = cfg["num_slices"]
demo_mode  = cfg["demo_mode"]

# ── 상단 메트릭 카드 ─────────────────────────────────────────────────────────

status_data = sisa.get_training_status(
    num_shards=num_shards, num_slices=num_slices, demo_mode=demo_mode
)
done_shards   = sum(1 for v in status_data.values() if v["status"] == "done")
avg_accuracy  = (
    sum(v["accuracy"] for v in status_data.values()) / num_shards * 100
    if status_data else 0.0
)
elapsed_min   = st.session_state.get("elapsed_time", 0.0) / 60
unlearn_queue = st.session_state.get("unlearn_queue", [])

m1, m2, m3, m4 = st.columns(4)
m1.metric("완료된 Shard",    f"{done_shards} / {num_shards}")
m2.metric("현재 정확도",      f"{avg_accuracy:.2f}%")
m3.metric("경과 시간",        f"{elapsed_min:.1f} 분")
m4.metric("대기 중인 망각 요청", f"{len(unlearn_queue)} 건")

st.divider()

# ── Shard 상태 그리드 (plotly scatter) ───────────────────────────────────────

st.subheader("Shard 상태 그리드")

STATUS_COLOR = {
    "done":       "#1D9E75",
    "training":   "#378ADD",
    "retraining": "#E24B4A",
    "idle":       "#888780",
}

COLS, ROWS = 5, 4   # 4행 5열 = 20 shard

xs, ys, colors, texts, hovers = [], [], [], [], []
for idx in range(num_shards):
    row = idx // COLS
    col = idx % COLS
    sv  = status_data.get(idx, {})
    st_ = sv.get("status", "idle")
    acc = sv.get("accuracy", 0.0) * 100

    xs.append(col)
    ys.append(ROWS - 1 - row)
    colors.append(STATUS_COLOR.get(st_, STATUS_COLOR["idle"]))
    texts.append(str(idx))
    hovers.append(f"Shard {idx}<br>Status: {st_}<br>Acc: {acc:.1f}%")

# 빈 shard 자리 (num_shards < COLS*ROWS) 는 그냥 비워 둠
fig_grid = go.Figure()
fig_grid.add_trace(go.Scatter(
    x=xs, y=ys,
    mode="markers+text",
    marker=dict(size=52, color=colors, symbol="square"),
    text=texts,
    textfont=dict(color="white", size=14),
    textposition="middle center",
    hovertext=hovers,
    hoverinfo="text",
))
# 범례 (더미 트레이스)
for label, color in STATUS_COLOR.items():
    fig_grid.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=color, symbol="square"),
        name=label,
    ))
fig_grid.update_layout(
    height=260,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               range=[-0.6, COLS - 0.4]),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               range=[-0.6, ROWS - 0.4]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=-0.05),
)
st.plotly_chart(fig_grid, use_container_width=True)

st.divider()

# ── 현재 Shard 상세 ──────────────────────────────────────────────────────────

st.subheader("현재 Shard 상세")

sel_shard = st.selectbox("Shard 선택", list(range(num_shards)),
                         format_func=lambda i: f"Shard {i}")
shard_info = status_data.get(sel_shard, {})
slices     = shard_info.get("slices", [])

left, right = st.columns(2)

with left:
    # Slice 진행률
    n_done = len(slices)
    st.progress(n_done / max(num_slices, 1),
                text=f"Slice 진행: {n_done} / {num_slices}")

    # Loss & Accuracy 라인 차트
    if slices:
        slice_nums = [s["slice"] for s in slices]
        accs  = [s["accuracy"] * 100 for s in slices]
        losses = [s["loss"] for s in slices]
    else:
        slice_nums = list(range(num_slices))
        accs   = mock_data.generate_accuracy_curve(num_slices)
        accs   = [a * 100 for a in accs]
        losses = mock_data.generate_loss_curve(num_slices)

    fig_detail = go.Figure()
    fig_detail.add_trace(go.Scatter(
        x=slice_nums, y=losses, name="Loss",
        mode="lines+markers", yaxis="y1",
        line=dict(color="#E24B4A", width=2),
    ))
    fig_detail.add_trace(go.Scatter(
        x=slice_nums, y=accs, name="Accuracy (%)",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#1D9E75", width=2),
    ))
    fig_detail.update_layout(
        height=280,
        xaxis=dict(title="Slice"),
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Accuracy (%)", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_detail, use_container_width=True)

with right:
    st.markdown("**단계별 타임라인**")
    st.markdown(
        """
        <style>
        .timeline { list-style: none; padding: 0; margin: 0; }
        .timeline li { position: relative; padding-left: 28px; margin-bottom: 12px; }
        .timeline li::before {
            content: "●"; position: absolute; left: 0;
            color: #1D9E75; font-size: 1.1rem; line-height: 1.4;
        }
        .timeline li::after {
            content: ""; position: absolute;
            left: 7px; top: 22px; bottom: -14px; width: 2px;
            background: #ccc;
        }
        .timeline li:last-child::after { display: none; }
        .tl-label { font-weight: 600; }
        .tl-sub   { font-size: 0.82rem; color: #666; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    timeline_steps = [
        ("데이터 로드", f"SVHN train split → {num_shards} shard"),
        ("Shard 분할",  f"{num_shards} shards, seed=42"),
        ("Slice 분할",  f"{num_slices} slices per shard (누적)"),
        ("모델 초기화", "SimpleCNN (3-Conv + FC)"),
        ("Incremental 학습", f"Slice 0 → {num_slices - 1} 순차 학습"),
        ("체크포인트 저장", "shard{i}_slice{j}.pt"),
    ]

    items_html = "\n".join(
        f'<li><span class="tl-label">{title}</span>'
        f'<br><span class="tl-sub">{detail}</span></li>'
        for title, detail in timeline_steps
    )
    st.markdown(f'<ul class="timeline">{items_html}</ul>', unsafe_allow_html=True)

st.divider()

# ── 새로고침 ──────────────────────────────────────────────────────────────────
col_btn, col_toggle = st.columns([1, 3])

with col_btn:
    if st.button("🔄 새로고침"):
        st.rerun()

with col_toggle:
    auto = st.toggle("자동 새로고침 (5초)", value=False)

if auto:
    time.sleep(5)
    st.rerun()