"""Editor detection and file opening — ported from rovodev commands/config/command.py."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import rich


def detect_editor() -> str | None:
    """Detect the best editor based on the current terminal environment."""
    # Cursor terminal
    if (
        os.environ.get("__CFBundleIdentifier") == "com.todesktop.230313mzl4w4u92"
        or os.environ.get("CURSOR_TRACE_ID")
    ):
        return "cursor"

    # VSCode terminal
    if (
        os.environ.get("__CFBundleIdentifier") == "com.microsoft.VSCode"
        or os.environ.get("VSCODE_PROFILE_INITIALIZED")
        or os.environ.get("VSCODE_INJECTION")
    ):
        return "code"

    # JetBrains terminal
    if os.environ.get("__CFBundleIdentifier") == "com.jetbrains.intellij":
        return "idea"

    # Windsurf terminal
    if os.environ.get("WINDSURF_TRACE_ID"):
        return "windsurf"

    return None


def open_file_in_editor(file_path: str, create_if_missing: bool = True) -> None:
    """Open a file in the best available editor.

    Priority: IDE detection → $EDITOR env var → nano → print path.
    """
    path = Path(file_path)
    if create_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    editor = detect_editor()
    if not editor or not shutil.which(editor):
        editor = os.environ.get("EDITOR")

    if not editor or not shutil.which(editor):
        # Last resort fallbacks
        for fallback in ("nano", "vim", "vi"):
            if shutil.which(fallback):
                editor = fallback
                break

    if not editor:
        rich.print(
            f"[yellow]Could not detect editor. Please set $EDITOR or open manually:[/yellow]\n  {path}"
        )
        return

    try:
        # Intentionally blocking — editor takes over the terminal until closed.
        # If moving to async context, wrap with asyncio.to_thread().
        subprocess.run([editor, str(path)], check=True)
    except subprocess.CalledProcessError as e:
        rich.print(f"[yellow]Failed to open '{path}' in '{editor}': {e}[/yellow]")
    except FileNotFoundError:
        rich.print(f"[yellow]Editor '{editor}' not found. Please set $EDITOR.[/yellow]")
