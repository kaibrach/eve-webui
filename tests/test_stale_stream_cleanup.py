from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUTES_SRC = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
SESSIONS_SRC = (REPO / "static" / "sessions.js").read_text(encoding="utf-8")
SW_SRC = (REPO / "static" / "sw.js").read_text(encoding="utf-8")


def test_stale_stream_cleanup_helper_exists():
    assert "def _clear_stale_stream_state(session)" in ROUTES_SRC
    assert "stream_id in STREAMS" in ROUTES_SRC
    assert "session.active_stream_id = None" in ROUTES_SRC
    assert "session.pending_user_message = None" in ROUTES_SRC
    assert "session.pending_attachments = []" in ROUTES_SRC
    assert "session.pending_started_at = None" in ROUTES_SRC
    assert "session.save()" in ROUTES_SRC


def test_session_load_clears_stale_stream_before_response():
    load_pos = ROUTES_SRC.index("s = get_session(sid, metadata_only=(not load_messages))")
    cleanup_pos = ROUTES_SRC.index("_clear_stale_stream_state(s)", load_pos)
    response_pos = ROUTES_SRC.index('"active_stream_id": getattr(s, "active_stream_id", None)', cleanup_pos)
    assert load_pos < cleanup_pos < response_pos


def test_chat_start_clears_stale_pending_state_not_only_active_id():
    stale_comment_pos = ROUTES_SRC.index("# Stale stream id from a previous run; clear and continue.")
    cleanup_pos = ROUTES_SRC.index("_clear_stale_stream_state(s)", stale_comment_pos)
    stream_id_pos = ROUTES_SRC.index("stream_id = uuid.uuid4().hex", cleanup_pos)
    assert stale_comment_pos < cleanup_pos < stream_id_pos


def test_frontend_drops_inflight_cache_when_server_session_is_idle():
    marker = "If the server says the session is idle, discard any browser-side inflight"
    marker_pos = SESSIONS_SRC.index(marker)
    window = SESSIONS_SRC[marker_pos:marker_pos + 500]
    assert "if(!activeStreamId&&INFLIGHT[sid])" in window
    assert "delete INFLIGHT[sid]" in window
    assert "clearInflightState" in window
    assert "S.busy=false" in window


def test_service_worker_cache_bumped_for_frontend_fix_delivery():
    """The SW CACHE_NAME must be keyed on the WEBUI_VERSION placeholder so
    every release naturally invalidates the previous shell cache and delivers
    the frontend half of the stale-stream cleanup fix to existing browsers.

    Originally pinned a manual `-stale-stream-cleanup1` suffix on
    `CACHE_NAME` (PR #1525 author shipped that to force-bump existing
    SWs). During the v0.50.279 stage build that suffix collided with the
    independent #1517 placeholder rename (`__CACHE_VERSION__` →
    `__WEBUI_VERSION__`), so the maintainer dropped the manual suffix in
    favor of the canonical version-token path. The natural bump still
    invalidates the old cache via `keys.filter((k) => k !== CACHE_NAME)`
    in the activate handler — same delivery guarantee, less churn.
    """
    # CACHE_NAME must include the WEBUI_VERSION placeholder so each release
    # produces a different cache name. The activate handler then deletes any
    # cache whose key != current CACHE_NAME, so the old shell is reaped on
    # every upgrade and the new sessions.js (with the INFLIGHT[sid] clear)
    # ships to existing browsers.
    assert "CACHE_NAME = 'hermes-shell-__WEBUI_VERSION__'" in SW_SRC, (
        "SW CACHE_NAME must include __WEBUI_VERSION__ so each release "
        "invalidates the previous cache and delivers frontend changes."
    )
