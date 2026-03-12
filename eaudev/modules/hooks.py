"""
EauDev hooks system — pre/post tool call lifecycle hooks.

Protocol (identical to Claude Code hooks):
  - Hook receives JSON on stdin: {"tool_name": str, "tool_input": dict, "session_id": str}
  - Hook communicates via exit code:
      0 = allow / proceed (PreToolUse) or acknowledged (PostToolUse)
      1 = warn  — stderr shown to user, tool still runs (PreToolUse only)
      2 = block — stderr shown to agent as tool result, tool does NOT run (PreToolUse only)
  - PostToolUse hooks additionally receive {"tool_result": str} in stdin JSON
  - Hook stderr on exit 1/2 is returned as the tool result message

Hooks are registered in ~/.eaudev/config.yml under:
  hooks:
    PreToolUse:
      - matcher: "run_bash"          # tool name or "*" for all tools
        command: "python3 ~/.eaudev/hooks/bash_guard.py"
    PostToolUse:
      - matcher: "write_file"
        command: "python3 ~/.eaudev/hooks/auto_git_add.py"
"""

from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path
from typing import Optional

from eaudev.common.config_model import HookEntry, HooksConfig


# ── Hook result ───────────────────────────────────────────────────────────────

class HookResult:
    """Result from running a set of hooks."""
    def __init__(
        self,
        allowed: bool = True,
        message: Optional[str] = None,
        exit_code: int = 0,
        additional_context: Optional[str] = None,
    ):
        self.allowed  = allowed   # False means block the tool call
        self.message  = message   # stderr from the hook, if any
        self.exit_code = exit_code
        self.additional_context = additional_context  # JSON additionalContext from stdout

    def __repr__(self) -> str:
        return f"HookResult(allowed={self.allowed}, exit_code={self.exit_code}, message={self.message!r})"


# ── Main entry points ─────────────────────────────────────────────────────────

def run_pre_tool_hooks(
    tool_name: str,
    tool_input: dict,
    hooks_cfg: HooksConfig,
    session_id: str = "",
) -> HookResult:
    """
    Run all matching PreToolUse hooks for a tool call.
    Returns HookResult — if allowed=False, the tool should NOT be dispatched.
    """
    if not hooks_cfg.enabled:
        return HookResult(allowed=True)

    matching = _find_matching_hooks(tool_name, hooks_cfg.PreToolUse)
    if not matching:
        return HookResult(allowed=True)

    payload = json.dumps({
        "tool_name":  tool_name,
        "tool_input": tool_input,
        "session_id": session_id,
    })

    contexts = []
    for hook in matching:
        result = _run_hook(hook.command, payload)
        if result.additional_context:
            contexts.append(result.additional_context)
        if result.exit_code == 2:
            # Block — return immediately, don't run remaining hooks
            return HookResult(
                allowed=False,
                message=result.message or f"[hook blocked: {hook.command}]",
                exit_code=2,
            )
        if result.exit_code == 1:
            # Warn — log stderr but continue
            if result.message:
                _print_hook_warning(hook.command, result.message)

    return HookResult(allowed=True, additional_context="\n".join(contexts) if contexts else None)


def run_post_tool_hooks(
    tool_name: str,
    tool_input: dict,
    tool_result: str,
    hooks_cfg: HooksConfig,
    session_id: str = "",
) -> Optional[str]:
    """
    Run all matching PostToolUse hooks for a completed tool call.
    Exit codes are informational only — tool already ran.
    Returns any additionalContext injected by hooks.
    """
    if not hooks_cfg.enabled:
        return None

    matching = _find_matching_hooks(tool_name, hooks_cfg.PostToolUse)
    if not matching:
        return None

    payload = json.dumps({
        "tool_name":   tool_name,
        "tool_input":  tool_input,
        "tool_result": tool_result,
        "session_id":  session_id,
    })

    contexts = []
    for hook in matching:
        result = _run_hook(hook.command, payload)
        if result.additional_context:
            contexts.append(result.additional_context)
        if result.exit_code != 0 and result.message:
            _print_hook_warning(hook.command, result.message)

    return "\n".join(contexts) if contexts else None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_matching_hooks(tool_name: str, hooks: list[HookEntry]) -> list[HookEntry]:
    """Return hooks whose matcher matches the tool name."""
    return [
        h for h in hooks
        if h.matcher == "*" or h.matcher == tool_name
    ]


def _run_hook(command: str, stdin_payload: str) -> HookResult:
    """Run a single hook command with JSON payload on stdin."""
    try:
        proc = subprocess.run(
            command, shell=True, input=stdin_payload,
            capture_output=True, text=True, timeout=10, env={**os.environ},
        )
        stderr = proc.stderr.strip() if proc.stderr else None
        stdout = proc.stdout.strip() if proc.stdout else None

        # Parse stdout JSON for additionalContext
        additional_context = None
        if stdout:
            try:
                out_data = json.loads(stdout)
                additional_context = out_data.get("additionalContext") or out_data.get("additional_context")
            except (json.JSONDecodeError, AttributeError):
                pass

        return HookResult(
            allowed=proc.returncode != 2,
            message=stderr or None,
            exit_code=proc.returncode,
            additional_context=additional_context,
        )
    except subprocess.TimeoutExpired:
        return HookResult(allowed=True)
    except Exception as exc:
        return HookResult(allowed=True, message=str(exc))


def run_session_start_hooks(
    session_id: str,
    hooks_cfg: HooksConfig,
    context: dict = None,
) -> Optional[str]:
    """
    Run all SessionStart hooks at the beginning of a session.
    Returns any additionalContext injected by hooks.
    """
    if not hooks_cfg.enabled:
        return None

    matching = _find_matching_hooks("*", hooks_cfg.SessionStart)
    if not matching:
        return None

    payload = json.dumps({
        "event": "SessionStart",
        "session_id": session_id,
        "context": context or {},
    })

    contexts = []
    for hook in matching:
        result = _run_hook(hook.command, payload)
        if result.additional_context:
            contexts.append(result.additional_context)

    return "\n".join(contexts) if contexts else None


def _print_hook_warning(command: str, message: str) -> None:
    """Print hook warning to console."""
    from rich.console import Console
    _console = Console(stderr=True)
    _console.print(f"[yellow]hook warning ({command}):[/yellow] {message}")
