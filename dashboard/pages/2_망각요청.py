"""페이지 2 — 망각 요청."""

import sys
import time
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import sisa
from core.state import add_unlearn_request, init_state, log_experiment

st.set_page_config(page_title="망각 요청", page_icon="🗑️", layout="wide")
init_state()

st.title("🗑️ 망각 요청")
st.caption("삭제할 데이터 인덱스를 지정하고 SISA 망각 파이프라인을 실행합니다.")

cfg       = st.session_state.config
num_shards = cfg["num_shards"]
demo_mode  = cfg["demo_mode"]

# ── 1. 인덱스 입력 ────────────────────────────────────────────────────────────

st.subheader("1. 망각할 인덱스 입력")

raw_input = st.text_input(
    "인덱스 (쉼표 구분)",
    placeholder="예: 0, 5, 12, 100",
)

col_add, col_clear = st.columns([1, 5])
with col_add:
    add_clicked = st.button("추가", type="primary")

if add_clicked and raw_input.strip():
    try:
        new_indices = [int(x.strip()) for x in raw_input.split(",") if x.strip()]
        add_unlearn_request(new_indices)
        st.success(f"{len(new_indices)}개 인덱스가 큐에 추가되었습니다.")
    except ValueError:
        st.error("숫자만 입력해 주세요.")

# 현재 큐를 multiselect로 표시 (삭제 가능)
queue: list[int] = st.session_state.get("unlearn_queue", [])
if queue:
    remaining = st.multiselect(
        "큐에 있는 인덱스 (제거하려면 선택 해제)",
        options=queue,
        default=queue,
        label_visibility="visible",
    )
    if set(remaining) != set(queue):
        st.session_state["unlearn_queue"] = sorted(remaining)
        st.rerun()
else:
    st.info("아직 큐에 인덱스가 없습니다.")

# ── 2. 영향 범위 자동 분석 ────────────────────────────────────────────────────

st.subheader("2. 영향 범위 분석")

current_queue: list[int] = st.session_state.get("unlearn_queue", [])

if current_queue:
    shard_map = sisa.get_point_to_shard_map(
        demo_mode=demo_mode, num_shards=num_shards
    )
    affected_shards = sorted({
        shard_map[i] for i in current_queue if i in shard_map
    })

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("선택된 인덱스 수", len(current_queue))
    col_info2.metric("영향받는 Shard", len(affected_shards))
    sim = sisa.run_unlearn(
        current_queue, demo_mode=True, num_shards=num_shards
    ) if demo_mode else {"speedup": "N/A", "sisa_time": "실행 후 측정"}
    speedup_str = (f"{sim['speedup']:.2f}x" if isinstance(sim.get("speedup"), float)
                   else str(sim.get("speedup", "N/A")))
    col_info3.metric("예상 Speedup", speedup_str)

    # 영향받는 shard 막대 그래프
    colors = ["#E24B4A" if i in affected_shards else "#BBBBBB"
              for i in range(num_shards)]
    labels = ["재학습 필요" if i in affected_shards else "정상"
              for i in range(num_shards)]
    fig_impact = go.Figure(go.Bar(
        x=[f"Shard {i}" for i in range(num_shards)],
        y=[1] * num_shards,
        marker_color=colors,
        text=labels,
        textposition="inside",
    ))
    fig_impact.update_layout(
        height=180,
        yaxis=dict(showticklabels=False, range=[0, 1.5]),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    # 각 shard의 재학습 시작 slice (mock: slice 0)
    if affected_shards:
        st.markdown("**Shard별 재학습 시작 Slice**")
        cols = st.columns(min(len(affected_shards), 5))
        for ci, si in enumerate(affected_shards[:5]):
            cols[ci].metric(f"Shard {si}", "Slice 0")

# ── 3. 망각 실행 ──────────────────────────────────────────────────────────────

st.subheader("3. 망각 실행")

if not current_queue:
    st.warning("인덱스를 먼저 추가하세요.")
else:
    st.warning(
        f"⚠️ {len(current_queue)}개 샘플을 망각합니다. "
        "영향받은 shard의 체크포인트가 갱신됩니다."
    )

    if st.button("🚀 망각 실행", type="primary", disabled=not current_queue):
        prog = st.progress(0, text="망각 처리 중...")
        started = time.perf_counter()

        if demo_mode:
            for pct in range(0, 101, 20):
                time.sleep(0.3)
                prog.progress(pct, text=f"망각 처리 중... {pct}%")
            result = sisa.run_unlearn(
                current_queue, demo_mode=True, num_shards=num_shards
            )
            elapsed = time.perf_counter() - started
            result["unlearn_time_sec"] = round(elapsed, 2)
            success = True
        else:
            proc = sisa.run_unlearning(
                current_queue, num_shards=num_shards,
                num_slices=cfg["num_slices"],
            )
            prog.progress(100)
            success = proc.returncode == 0
            result  = {"unlearn_time_sec": round(time.perf_counter() - started, 2)}
            if not success:
                st.error("❌ 망각 실패")
                st.code(proc.stderr or proc.stdout)

        if success:
            prog.progress(100, text="완료!")
            st.success(
                f"✅ 망각 완료!  "
                f"소요 시간: **{result.get('unlearn_time_sec', 0):.1f}초**"
            )
            entry = {
                "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "indices":        list(current_queue),
                "affected_shards": affected_shards if current_queue else [],
                "elapsed_sec":    result.get("unlearn_time_sec", 0),
                "speedup":        result.get("speedup", result.get("sisa_time", "N/A")),
            }
            st.session_state["unlearn_results"].append(entry)
            log_experiment({"event": "unlearn_complete", **entry})
            st.session_state["unlearn_queue"] = []

# ── 4. 망각 이력 테이블 ───────────────────────────────────────────────────────

st.subheader("4. 망각 이력")

history: list[dict] = st.session_state.get("unlearn_results", [])
if history:
    import pandas as pd
    rows = []
    for h in history:
        rows.append({
            "요청 시각":    h.get("timestamp", "-"),
            "삭제 인덱스":  str(h.get("indices", []))[:60],
            "영향 Shard":   str(h.get("affected_shards", [])),
            "소요 시간 (s)": h.get("elapsed_sec", "-"),
            "Speedup":      h.get("speedup", "-"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("아직 망각 이력이 없습니다.")