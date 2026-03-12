"""EauDev session menu panel — arrow-key browser, no nemo/pydantic_ai deps."""

from __future__ import annotations

import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING

from humanize import naturaltime

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output import create_output
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from eaudev.constants import DEFAULT_PANEL_WIDTH
from eaudev.ui.components.user_menu_panel import Choice, user_menu_panel

if TYPE_CHECKING:
    from eaudev.modules.sessions import Session

console = Console()

MAX_VISIBLE_ITEMS = 15
TITLE_WIDTH = 40
USAGE_WIDTH = 10
TITLE_TEMPLATE = "{title:<34} {message_cnt:>9} {last_activity:>16}"
MENU_TEMPLATE  = "{title:<40} {message_cnt:>3} {last_activity:>16}"
DETAILS_WIDTH  = 39
DETAILS_TEMPLATE = """
[bold]{title}[/bold]
[bright_black]{session_id}[/bright_black]
Created [bright_black]{created}[/bright_black]

🔎 Message Analysis
User: [bright_black]{user_cnt}[/bright_black]  Assistant: [bright_black]{assistant_cnt}[/bright_black]  Total: [bright_black]{message_cnt}[/bright_black]

💬 Conversation
Initial Prompt: [bright_black]{initial_prompt}[/bright_black]
Latest Message: [bright_black]{latest_message}[/bright_black]
"""
FOOTER = (
    "[bright_black]↑ ↓: Navigate | n: New | f: Fork | d: Delete | Enter: Select | q: Quit[/bright_black]"
)


async def session_menu_panel(
    sessions: dict[str, "Session"],
    current_session_id: str,
    persistence_dir: Path,
) -> tuple["Session | None", bool]:
    """Display arrow-key session browser. Returns (selected_session | None, is_new)."""
    if not sessions:
        console.print("  No saved sessions.")
        return None, True

    # Loop for deletion/fork re-entry — avoids recursion on repeated operations
    while True:
        session_ids = list(sessions)
        if not session_ids:
            return None, True

        buffer = Buffer()
        kb = KeyBindings()

        # Default to current session if present, else first
        try:
            start_idx = session_ids.index(current_session_id)
        except ValueError:
            start_idx = 0

        info: dict = {
            "index": start_idx,
            "choice": sessions[session_ids[start_idx]],
            "is_new": False,
            "deleted_session": None,
        }

        @kb.add("c-c")
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=SystemExit(0))

        @kb.add("down")
        def _(event: KeyPressEvent) -> None:
            info["index"] = (info["index"] + 1) % len(session_ids)
            info["choice"] = sessions[session_ids[info["index"]]]

        @kb.add("up")
        def _(event: KeyPressEvent) -> None:
            info["index"] = (info["index"] - 1) % len(session_ids)
            info["choice"] = sessions[session_ids[info["index"]]]

        @kb.add("enter")
        def _(event: KeyPressEvent) -> None:
            event.app.exit()

        @kb.add("q")
        def _(event: KeyPressEvent) -> None:
            info["choice"] = None
            event.app.exit()

        @kb.add("n")
        def _(event: KeyPressEvent) -> None:
            info["choice"] = None
            info["is_new"] = True
            event.app.exit()

        @kb.add("d")
        def _(event: KeyPressEvent) -> None:
            if not info["choice"]:
                return
            info["deleted_session"] = info["choice"].session_id
            event.app.exit()

        @kb.add("f")
        def _(event: KeyPressEvent) -> None:
            """Fork the selected session — copy its message history into a new session."""
            session = info["choice"]
            if not session:
                return
            fork_id = str(uuid.uuid4())
            fork_path = persistence_dir / fork_id
            fork_path.mkdir(parents=True, exist_ok=True)
            # Copy session file if it exists
            src = Path(session.path) if session.path else None
            if src and src.exists():
                shutil.copytree(src, fork_path, dirs_exist_ok=True)
            # Build a minimal forked session and write it
            from eaudev.modules.sessions import Session as _Session
            forked = _Session(
                session_id=fork_id,
                title=f"Fork of {session.title}",
                workspace_path=session.workspace_path,
                initial_prompt=session.initial_prompt,
                message_history=list(session.message_history),
            )
            forked.path = str(fork_path)
            forked.save(fork_path)
            sessions[fork_id] = forked
            info["choice"] = forked
            info["is_new"] = True
            event.app.exit()

        layout = Layout(Window(BufferControl(buffer=buffer)))
        app = Application(
            output=create_output(),
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
            paste_mode=True,
        )

        def _create_panel() -> Group:
            header_line = (
                "[bold]  "
                + TITLE_TEMPLATE.format(
                    title="Title", message_cnt="Messages", last_activity="Last Activity"
                )
                + "[/bold]"
            )
            menu_lines = [header_line]

            first_vis = max(0, info["index"] - MAX_VISIBLE_ITEMS // 2)
            last_vis  = min(len(session_ids), first_vis + MAX_VISIBLE_ITEMS)
            first_vis = max(0, last_vis - MAX_VISIBLE_ITEMS)
            visible   = session_ids[first_vis:last_vis]

            menu_lines.append("[bright_black]  ...[/bright_black]" if first_vis > 0 else "")

            details_text = ""
            for sid in visible:
                s = sessions[sid]
                entry = MENU_TEMPLATE.format(
                    title=(s.title[:TITLE_WIDTH] if len(s.title) <= TITLE_WIDTH
                           else s.title[: TITLE_WIDTH - 3] + "..."),
                    message_cnt=s.num_messages,
                    last_activity=_fmt_dt(getattr(s, "last_saved", None) or s.created),
                )
                if info["choice"] and sid == info["choice"].session_id:
                    details_text = _fmt_details(s)
                    menu_lines.append(f"[blue bold]> {entry}[/blue bold]")
                else:
                    menu_lines.append(f"[bright_black]  {entry}[/bright_black]")

            menu_lines.append("[bright_black]  ...[/bright_black]" if last_vis < len(session_ids) else "")

            panel_content = Columns(
                ["\n" + "\n".join(menu_lines), details_text], padding=(0, 2)
            )
            return Group(
                "",
                Panel(panel_content, width=DEFAULT_PANEL_WIDTH, title="Session management", title_align="left"),
                FOOTER,
            )

        with Live(_create_panel(), auto_refresh=False, transient=True) as live:
            def before_render(_: Application) -> None:
                live.update(_create_panel())
                live.refresh()
            app.before_render += before_render
            await app.run_async()

        # ── Handle deletion ──────────────────────────────────────────────────────
        if info["deleted_session"]:
            if len(sessions) == 1:
                console.print("[red]Cannot delete the only remaining session.[/red]")
                current_session_id = next(iter(sessions), current_session_id)
                continue  # restart loop with remaining sessions
            
            to_delete = sessions[info["deleted_session"]]
            confirmed = await user_menu_panel(
                choices=[Choice(name="Yes", value=True), Choice(name="No", value=False)],
                message=f"Delete session '{to_delete.title}'?",
            )
            if confirmed:
                if to_delete.path and Path(to_delete.path).exists():
                    shutil.rmtree(to_delete.path)
                del sessions[info["deleted_session"]]
                console.print(f"[green]Deleted session: {to_delete.session_id}[/green]")
                current_session_id = (
                    current_session_id
                    if current_session_id != to_delete.session_id
                    else next(iter(sessions), current_session_id)
                )
            else:
                current_session_id = current_session_id
            if sessions:
                continue  # restart loop instead of recursing
        
        # Normal exit — no deletion/re-entry needed
        break

    time.sleep(0.1)
    return info["choice"], info["is_new"]


def session_menu_panel_sync(
    sessions: dict[str, "Session"],
    current_session_id: str,
    persistence_dir: Path,
) -> tuple["Session | None", bool]:
    """Synchronous wrapper for session_menu_panel."""
    import asyncio
    return asyncio.run(session_menu_panel(sessions, current_session_id, persistence_dir))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_dt(dt_str: str | None) -> str:
    """Format a datetime string using humanize.naturaltime."""
    if not dt_str:
        return "-"
    # Try multiple formats: plain datetime, ISO 8601 with T, ISO with timezone
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(dt_str[:26], fmt[:len(fmt)])
            return naturaltime(dt)
        except ValueError:
            continue
    # ISO format with timezone via fromisoformat (Python 3.11+)
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return naturaltime(dt)
    except ValueError:
        return dt_str[:16]  # Show at least YYYY-MM-DD HH:MM


def _fmt_details(session: "Session") -> str:
    """Format the right-hand details panel for a session."""
    # Count messages by role (exclude system)
    history = session.message_history or []
    user_cnt = sum(1 for m in history if m.get("role") == "user")
    assistant_cnt = sum(1 for m in history if m.get("role") == "assistant")
    total_cnt = user_cnt + assistant_cnt

    # Initial prompt snippet
    raw_prompt = (session.initial_prompt or "-")
    lines = wrap(raw_prompt[:200], DETAILS_WIDTH) or ["-"]
    prompt_wrapped = "\n".join(lines[:2])

    # Latest non-system message snippet
    latest = "-"
    for m in reversed(history):
        if m.get("role") in ("user", "assistant"):
            content = str(m.get("content", ""))
            # Skip tool results
            if content.startswith("["):
                continue
            latest = content[:80].replace("\n", " ")
            if len(content) > 80:
                latest += "..."
            break

    return DETAILS_TEMPLATE.format(
        title=session.title[:DETAILS_WIDTH],
        session_id=session.session_id,
        created=session.created,
        user_cnt=user_cnt,
        assistant_cnt=assistant_cnt,
        message_cnt=total_cnt,
        initial_prompt=prompt_wrapped,
        latest_message=latest,
    )
