"""Regressions for first-turn sessions appearing in the sidebar immediately."""

import pathlib

REPO = pathlib.Path(__file__).parent.parent


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


class TestSidebarFirstTurnVisibility:
    def test_messages_send_optimistically_upserts_active_sidebar_row(self):
        src = read("static/messages.js")
        assert "upsertActiveSessionForLocalTurn" in src, (
            "send() must optimistically upsert the active session into the sidebar "
            "as soon as the local user message is pushed."
        )
        push_idx = src.index("S.messages.push(userMsg);renderMessages();appendThinking();setBusy(true);")
        helper_idx = src.index("upsertActiveSessionForLocalTurn", push_idx)
        start_idx = src.index("api('/api/chat/start'", push_idx)
        assert helper_idx < start_idx, (
            "The sidebar row must be rendered before /api/chat/start returns so "
            "tool calls are reachable while the first agent turn is still running."
        )
        pre_start = src[helper_idx:start_idx]
        assert "renderSessionList();" not in pre_start, (
            "Do not re-fetch /api/sessions before /api/chat/start saves pending state; "
            "that race can overwrite the optimistic first-turn row with an empty list."
        )

    def test_sessions_js_has_local_turn_upsert_helper(self):
        src = read("static/sessions.js")
        assert "function upsertActiveSessionForLocalTurn" in src
        start = src.index("function upsertActiveSessionForLocalTurn")
        end = src.index("function renderSessionListFromCache", start)
        body = src[start:end]
        assert "_allSessions.unshift" in body or "_allSessions.splice" in body, (
            "Helper must add a missing active session to the cached sidebar list."
        )
        assert "S.session.message_count" in body and "S.messages.length" in body, (
            "Helper must treat the locally pushed user message as a real sidebar message."
        )
        assert "is_streaming:true" in body.replace(" ", ""), (
            "Optimistic row should render as streaming until the backend reconciles."
        )

    def test_backend_compact_counts_pending_first_turn_as_visible(self):
        src = read("api/models.py")
        compact = src[src.index("def compact"):src.index("def _get_profile_home")]
        assert "has_pending_user_message" in compact and "pending_user_message" in compact, (
            "Session.compact() must account for pending_user_message in sidebar metadata."
        )
        assert "message_count = max(message_count, 1)" in compact, (
            "Pending first user turn should make message_count non-zero for /api/sessions."
        )
        assert "pending_started_at" in compact and "last_message_at" in compact, (
            "Pending first user turn should sort by pending_started_at in the sidebar."
        )

    def test_backend_index_filter_keeps_pending_first_turn_sessions(self):
        src = read("api/models.py")
        index_filter_start = src.index("# Hide empty Untitled sessions from the UI entirely")
        index_filter_end = src.index("result = [s for s in result if not _hide_from_default_sidebar", index_filter_start)
        index_filter = src[index_filter_start:index_filter_end]
        assert "has_pending_user_message" in index_filter, (
            "The index-path empty-session filter must exempt pending first-turn sessions, "
            "matching the full-scan fallback."
        )
