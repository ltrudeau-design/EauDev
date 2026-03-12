"""Session persistence for EauDev (plain JSON, rovodev session_context.json format)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from eaudev.constants import SESSION_DIR


class Session:
    """A single conversation session."""

    def __init__(
        self,
        session_id: str | None = None,
        title: str = "Untitled Session",
        workspace_path: str | None = None,
        message_history: list[dict] | None = None,
        timestamp: float | None = None,
        initial_prompt: str | None = None,
        last_saved: str | None = None,
        path: Path | None = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title
        self.workspace_path = workspace_path or str(Path.cwd())
        self.message_history: list[dict] = message_history or []
        self.timestamp = timestamp or datetime.now().timestamp()
        self.initial_prompt = initial_prompt or ""
        self.last_saved = last_saved  # Human-readable mtime string
        self.path = path  # Path to session directory on disk

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, persistence_dir: Path | str = SESSION_DIR) -> Path:
        persistence_dir = Path(persistence_dir).expanduser()
        session_dir = persistence_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self.path = session_dir

        ctx = {
            "id": self.session_id,
            "title": self.title,
            "workspace_path": self.workspace_path,
            "timestamp": self.timestamp,
            "initial_prompt": self.initial_prompt,
            "message_history": self.message_history,
        }
        ctx_path = session_dir / "session_context.json"
        tmp_path = ctx_path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(ctx, indent=2))
        os.replace(tmp_path, ctx_path)  # atomic on POSIX

        # Update last_saved from actual mtime
        self.last_saved = datetime.fromtimestamp(
            ctx_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        meta = {
            "title": self.title,
            "workspace_path": self.workspace_path,
            "timestamp": self.timestamp,
        }
        (session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return session_dir

    @classmethod
    def load(cls, session_id: str, persistence_dir: Path | str = SESSION_DIR) -> "Session":
        persistence_dir = Path(persistence_dir).expanduser()
        session_dir = persistence_dir / session_id
        ctx_path = session_dir / "session_context.json"
        if not ctx_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        data = json.loads(ctx_path.read_text())

        # last_saved = mtime of session_context.json
        last_saved = datetime.fromtimestamp(
            ctx_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Try to enrich title from metadata.json
        title = data.get("title", "Untitled Session")
        meta_path = session_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                title = meta.get("title") or title
            except Exception:
                pass

        return cls(
            session_id=data.get("id", session_id),
            title=title,
            workspace_path=data.get("workspace_path"),
            message_history=data.get("message_history", []),
            timestamp=data.get("timestamp"),
            initial_prompt=data.get("initial_prompt", ""),
            last_saved=last_saved,
            path=session_dir,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_title_from_first_message(self) -> bool:
        """Set title from first user message. Returns True if successful."""
        for m in self.message_history:
            if m.get("role") == "user":
                self.title = m["content"][:60].replace("\n", " ")
                return True
        return False

    @property
    def num_messages(self) -> int:
        return len(self.message_history)

    @property
    def created(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def _sort_key(self) -> str:
        """Sort key: prefer last_saved (mtime), fall back to created (timestamp)."""
        return self.last_saved or self.created or ""


# ── Collection helpers ────────────────────────────────────────────────────────

def get_sessions(
    persistence_dir: Path | str = SESSION_DIR,
    workspace_path: Path | None = None,
) -> dict[str, Session]:
    """Load all sessions, optionally filtered to a specific workspace.

    Args:
        persistence_dir: Directory containing session subdirectories.
        workspace_path: If provided, only return sessions whose workspace_path
                        matches this path (resolved). Legacy sessions without a
                        workspace_path are always included for backward compat.

    Returns:
        Dict of session_id → Session, sorted by last_saved descending (most recent first).
    """
    persistence_dir = Path(persistence_dir).expanduser()
    sessions: dict[str, Session] = {}

    if not persistence_dir.exists():
        return sessions

    for ctx_path in persistence_dir.glob("*/session_context.json"):
        session_dir = ctx_path.parent
        try:
            s = Session.load(session_dir.name, persistence_dir)
        except Exception:
            continue

        # Workspace filtering
        if workspace_path is not None and s.workspace_path:
            try:
                if Path(s.workspace_path).resolve() != workspace_path.resolve():
                    continue
            except Exception:
                pass  # Legacy session with unresolvable path — include it

        sessions[s.session_id] = s

    # Sort by last_saved descending (most recent first)
    return dict(
        sorted(sessions.items(), key=lambda item: item[1]._sort_key(), reverse=True)
    )


def get_most_recent_session(
    persistence_dir: Path | str = SESSION_DIR,
    workspace_path: Path | None = None,
) -> Session | None:
    """Return the most recently saved session, or None if no sessions exist."""
    sessions = get_sessions(persistence_dir, workspace_path=workspace_path)
    if not sessions:
        return None
    return next(iter(sessions.values()))
