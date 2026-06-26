"""Streamlit 세션 상태 초기화 및 접근 헬퍼."""

from __future__ import annotations

import streamlit as st

_DEFAULTS: dict = {
    # SISA 설정
    "config": {
        "dataset":    "svhn",
        "num_shards": 5,
        "num_slices": 4,
        "batch_size": 128,
        "demo_mode":  True,
    },
    # shard별 상태 (init_state에서 num_shards 크기로 재설정)
    "shard_status":   [],   # List[str]  "idle"|"training"|"done"|"retraining"
    "shard_accuracy": [],   # List[float]
    "shard_loss":     [],   # List[float]
    # 현재 진행 위치
    "current_shard": 0,
    "current_slice": 0,
    "elapsed_time":  0.0,
    # 망각 큐 & 결과
    "unlearn_queue":   [],  # List[int]   — data_indices 누적
    "unlearn_results": [],  # List[dict]  — unlearn_request() 반환값 누적
    # 실험 로그
    "experiment_log": [],   # List[dict]  — {event, shard, slice, time, ...}
}


def _num_shards() -> int:
    cfg = st.session_state.get("config", _DEFAULTS["config"])
    return cfg.get("num_shards", 5)


def init_state() -> None:
    """누락된 세션 키를 기본값으로 초기화. 각 페이지 최상단에서 한 번 호출."""
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            # mutable 기본값은 복사해서 넣는다
            st.session_state[k] = v.copy() if isinstance(v, (list, dict)) else v

    # shard 목록 크기를 현재 config와 맞춤
    n = _num_shards()
    for key, default in (
        ("shard_status",   "idle"),
        ("shard_accuracy", 0.0),
        ("shard_loss",     0.0),
    ):
        lst = st.session_state[key]
        if len(lst) != n:
            st.session_state[key] = [default] * n


def update_shard_status(shard_idx: int, status: str) -> None:
    """shard_idx번 shard의 상태를 갱신.

    Args:
        shard_idx: 0-based shard 번호.
        status: "idle" | "training" | "done" | "retraining"
    """
    lst = st.session_state.get("shard_status", [])
    if shard_idx < len(lst):
        lst[shard_idx] = status
        st.session_state["shard_status"] = lst


def add_unlearn_request(data_indices: list[int]) -> None:
    """data_indices를 unlearn_queue에 추가.

    중복 인덱스는 제거하고 정렬해 저장한다.
    """
    queue: list[int] = st.session_state.get("unlearn_queue", [])
    merged = sorted(set(queue) | set(data_indices))
    st.session_state["unlearn_queue"] = merged


def log_experiment(event: dict) -> None:
    """experiment_log에 이벤트를 추가.

    Args:
        event: {"event": str, ...} 형태의 임의 dict.
               "event" 키가 없으면 "unknown"으로 채운다.
    """
    entry = {"event": "unknown", **event}
    log: list[dict] = st.session_state.get("experiment_log", [])
    log.append(entry)
    st.session_state["experiment_log"] = log