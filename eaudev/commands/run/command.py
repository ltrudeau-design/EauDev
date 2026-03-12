"""EauDev main run loop — ported from rovodev, local inference only."""

from __future__ import annotations

import asyncio
import atexit
import difflib
import json
import os
import re
import signal as _signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from eaudev.commands.run.command_registry import registry
from eaudev.common.banner import BANNER
from eaudev.common.config import load_config
from eaudev.common.config_model import EauDevConfig
from eaudev.common.editor import open_file_in_editor
from eaudev.common.exceptions import EauDevError, RequestTooLargeError, ServerError
from eaudev.constants import CONFIG_PATH, SESSION_DIR, VERSION, DEFAULT_EXIT_COMMANDS
from eaudev.modules.instructions import handle_instructions_command
from eaudev.modules.logging import setup_logging
from eaudev.modules.memory import handle_memory_command, handle_memory_note, get_memory_instructions
from eaudev.modules.sessions import Session, get_most_recent_session, get_sessions
from eaudev.ui.components.session_menu_panel import session_menu_panel_sync
from eaudev.ui.components.token_display import display_token_usage, DEFAULT_CONTEXT_LIMIT
from eaudev.modules.mcp_client import get_mcp_manager
from eaudev.modules.voice_io import VoiceIO, VoiceIOConfig, get_voice_io, check_dependencies
from eaudev.modules.memory_store import get_memory_store
from eaudev.modules.tool_permissions import ToolPermissionManager
from eaudev.modules.hooks import run_pre_tool_hooks, run_post_tool_hooks, run_session_start_hooks
from eaudev.modules.tool_call_parsers import get_parser
from eaudev.ui.prompt_session import FilteredFileHistory, PromptSession

console = Console()

BANNER_TEMPLATE = """\
{banner}

• Ask EauDev anything — "explain this repo", "add unit tests", "refactor this function".
• Type [bold]/[/bold] at any time to see available commands.
• Use [bold]Ctrl+C[/bold] to interrupt the agent during generation.
• Use [bold]/exit[/bold] to quit.

Working in [blue bold]{workdir}[/blue bold]
"""

HOME_DIR_WARNING = """\
[yellow]WARNING: You are running EauDev in your home directory.
This directory may be too large to process efficiently.
Consider running in a more specific project directory.[/yellow]\
"""

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are EauDev, a local AI coding agent running on the user's Mac (Apple Silicon).

Environment:
  Working directory : {workdir}
  Home directory    : {home}
  OS                : macOS Darwin arm64
  Shell             : /bin/zsh

# Rules
- One tool call per response. Wait for the result before calling another.
- Always use absolute paths.
- State your plan in 1-2 sentences before the first tool call.
- After completing a task, give a brief summary of what changed.
- Be concise — no filler, no apologies.
- Use tools ONLY for filesystem and shell operations. For knowledge questions, explanations, or conversation — respond directly without any tool call.
- run_bash has a 60 second timeout — long-running commands will be killed.

# Persistent Memory & LoRA Training
- Your conversation is recorded in a 5-layer SQLite memory system (observations, episodes, facts, FTS5 search, knowledge graph).
- At session end, your turns are compressed and exported to JSONL for LoRA fine-tuning.
- Over time, you become personalized to this user's workflow, preferences, and codebase through LoRA adapters.
- Session adapters are stored in ~/.cluster/adapters/ — they make you smarter about this specific workspace.
{thinking_directive}"""

# ── Built-in tool schemas (OpenAI function format) ─────────────────────────────

_EAUDEV_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given absolute path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, overwriting it if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command in the working directory. Timeout: 60 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given absolute path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the directory."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with the given content. Fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path for the new file."},
                    "content": {"type": "string", "description": "Content for the new file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Absolute source path."},
                    "destination": {"type": "string", "description": "Absolute destination path."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file at the given absolute path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["path"],
            },
        },
    },
]

# ── Tool implementations ───────────────────────────────────────────────────────

MAX_READ_CHARS = 16_000  # ~4K tokens — keeps context manageable on local inference

def _read_file(path: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1
        total_chars = len(content)
        if total_chars > MAX_READ_CHARS:
            truncated = content[:MAX_READ_CHARS]
            shown_lines = truncated.count("\n") + 1
            omitted = total_chars - MAX_READ_CHARS
            return (
                f"[read_file: {path} — {lines} lines total, showing first {shown_lines} "
                f"({omitted:,} chars omitted — use /summarize {path} for full analysis)]\n{truncated}"
            )
        return f"[read_file: {path} — {lines} lines]\n{content}"
    except Exception as e:
        return f"[read_file error: {e}]"


def _write_file(path: str, content: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        # Capture old content for diff display
        old_content = ""
        is_new = not p.exists()
        if not is_new:
            try:
                old_content = p.read_text(encoding="utf-8")
            except Exception:
                is_new = True
        p.write_text(content, encoding="utf-8")
        # Show diff if modifying existing file
        if not is_new and old_content != content:
            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{p.name}",
                tofile=f"b/{p.name}",
                lineterm="",
            ))
            if diff:
                diff_text = "".join(diff)
                console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))
        action = "created" if is_new else "updated"
        return f"[write_file: {action} {path} — {len(content)} bytes]"
    except Exception as e:
        return f"[write_file error: {e}]"


BASH_TIMEOUT = 60  # seconds — kill any bash command that runs longer than this

def _run_bash(command: str, workdir: str) -> str:
    """Run a shell command with streaming output to terminal. Returns full output for context."""
    try:
        cwd = os.path.expanduser(workdir)
        env = {**os.environ, "TERM": "xterm-256color"}
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=cwd, env=env, bufsize=1,
        )
        output_lines: list[str] = []
        console.print()
        timed_out = False

        # Drain stdout in a thread so we can enforce a wall-clock timeout
        def _drain():
            assert proc.stdout is not None
            for line in proc.stdout:
                console.print(f"  [bright_black]{line.rstrip()}[/bright_black]")
                output_lines.append(line)

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        drain_thread.join(timeout=BASH_TIMEOUT)

        if drain_thread.is_alive():
            # Timeout — kill the process
            timed_out = True
            proc.kill()
            drain_thread.join(timeout=2)
            console.print(
                f"  [yellow][run_bash: killed after {BASH_TIMEOUT}s timeout][/yellow]"
            )

        proc.wait()
        output = "".join(output_lines).strip()

        if timed_out:
            return (
                f"[run_bash error: timed out after {BASH_TIMEOUT}s — {command}]\n"
                f"{output or '(no output before timeout)'}"
            )
        return f"[run_bash exit={proc.returncode}: {command}]\n{output or '(no output)'}"
    except Exception as e:
        return f"[run_bash error: {e}]"


def _list_directory(path: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"  📁 {e.name}/")
            else:
                lines.append(f"  📄 {e.name}  ({e.stat().st_size:,} bytes)")
        return f"[list_directory: {path} — {len(entries)} entries]\n" + "\n".join(lines)
    except Exception as e:
        return f"[list_directory error: {e}]"


def _create_file(path: str, content: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        if p.exists():
            return f"[create_file: file already exists — {path}. Use write_file to overwrite.]"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[create_file: {path} — {len(content)} bytes]"
    except Exception as e:
        return f"[create_file error: {e}]"


def _move_file(source: str, destination: str) -> str:
    try:
        src = Path(os.path.expanduser(source))
        dst = Path(os.path.expanduser(destination))
        if not src.exists():
            return f"[move_file: source does not exist — {source}]"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"[move_file: {source} → {destination}]"
    except Exception as e:
        return f"[move_file error: {e}]"


def _delete_file(path: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        if not p.exists():
            return f"[delete_file: file does not exist — {path}]"
        if p.is_dir():
            return f"[delete_file: {path} is a directory — use run_bash with rm -rf if intended]"
        p.unlink()
        return f"[delete_file: {path} removed]"
    except Exception as e:
        return f"[delete_file error: {e}]"


# ── Tool dispatch ──────────────────────────────────────────────────────────────

def _dispatch_tool(tool_call: dict, perms: ToolPermissionManager, workdir: str, config: "EauDevConfig | None" = None, session_id: str = "") -> str:
    name = tool_call.get("name", "")
    # Flatten "arguments" wrapper if present (nous format: {"name":..., "arguments":{...}})
    if "arguments" in tool_call and isinstance(tool_call["arguments"], dict):
        tool_call = {"name": name, **tool_call["arguments"]}
    tool_input = {k: v for k, v in tool_call.items() if k != "name"}

    # ── PreToolUse hooks ──────────────────────────────────────────────────────
    if config is not None and config.hooks.enabled:
        hook_result = run_pre_tool_hooks(name, tool_input, config.hooks, session_id)
        if not hook_result.allowed:
            return hook_result.message or f"[hook blocked tool: {name}]"

    if name == "read_file":
        return _read_file(tool_call.get("path", ""))

    if name == "list_directory":
        return _list_directory(tool_call.get("path", ""))

    if name == "write_file":
        allowed, msg = perms.check_tool(tool_call)
        if not allowed:
            return f"[write_file: {msg}]"
        return _write_file(tool_call.get("path", ""), tool_call.get("content", ""))

    if name == "run_bash":
        allowed, msg = perms.check_tool(tool_call)
        if not allowed:
            return f"[run_bash: {msg}]"
        return _run_bash(tool_call.get("command", ""), workdir)

    if name == "create_file":
        allowed, msg = perms.check_tool(tool_call)
        if not allowed:
            return f"[create_file: {msg}]"
        return _create_file(tool_call.get("path", ""), tool_call.get("content", ""))

    if name == "move_file":
        allowed, msg = perms.check_tool(tool_call)
        if not allowed:
            return f"[move_file: {msg}]"
        return _move_file(tool_call.get("source", ""), tool_call.get("destination", ""))

    if name == "delete_file":
        allowed, msg = perms.check_tool(tool_call)
        if not allowed:
            return f"[delete_file: {msg}]"
        return _delete_file(tool_call.get("path", ""))

    # ── MCP tool call ─────────────────────────────────────────────────────────
    # MCP tools are named "{server}__{tool}" e.g. "archive__search_archive_tool"
    if "__" in name:
        mcp = get_mcp_manager()
        if mcp.has_tool(name):
            allowed, msg = perms.check_tool(tool_call)
            if not allowed:
                return f"[mcp:{name}: {msg}]"
            # Strip the dispatcher keys; pass remaining as arguments
            arguments = {k: v for k, v in tool_call.items() if k != "name"}
            return mcp.call_tool(name, arguments)

    return f"[unknown tool: {name}]"


# ── Tool call parsing ──────────────────────────────────────────────────────────

_NATIVE_TOOLS = frozenset((
    "write_file", "read_file", "run_bash", "list_directory",
    "create_file", "move_file", "delete_file",
))

def _is_known_tool(name: str) -> bool:
    """Return True if name is a native tool or an MCP tool (contains __)."""
    return name in _NATIVE_TOOLS or "__" in str(name)


def _extract_tool_call(text: str, model_name: str = "") -> tuple[dict | None, str]:
    """
    Extract a tool call from raw model output using the parser registry.

    The parser is selected based on model_name — Qwen3.5, GLM4, InternLM each
    have model-specific parsers that handle their native tool call formats.
    Falls back to StandardParser for unknown models or canonical <tool> format.

    Returns:
        (tool_call_dict, preamble_str) if a tool call was found
        (None, original_text)          if no tool call was found
    """
    parser = get_parser(model_name)
    tool_call, preamble = parser.parse(text)

    if tool_call is None:
        return None, text

    # Sanity check — ensure name is present and recognised
    name = tool_call.get("name", "")
    if not name or not _is_known_tool(name):
        raw = text[:120] + ("..." if len(text) > 120 else "")
        console.print(
            f"\n[bright_black][EauDev: model emitted unrecognised tool '{name}' — skipping: {raw}][/bright_black]\n"
        )
        return None, text

    return tool_call, preamble


# ── LLM calls ──────────────────────────────────────────────────────────────────

def _chat_no_stream(messages: list[dict], config: EauDevConfig, tools: list[dict] | None = None) -> tuple[str, int, dict | None]:
    """Non-streaming chat completion. Returns (response_text, total_tokens, tool_call_data|None)."""
    inf = config.agent.inference
    payload_dict: dict = {
        "model":           inf.model,
        "messages":        messages,
        "stream":          False,
        "temperature":     inf.temperature,
        "top_p":           inf.top_p,
        "top_k":           inf.top_k,
        "min_p":           inf.min_p,
        "max_tokens":      inf.max_tokens,
        "enable_thinking": inf.enable_thinking,
    }
    if tools:
        payload_dict["tools"] = tools
    payload = json.dumps(payload_dict).encode()

    req = urllib.request.Request(
        inf.endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )

    with Live(_make_status_bar(), console=console, refresh_per_second=4, transient=True):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8", errors="replace")
            if e.code in (400, 413) and (
                "context" in body_err.lower() or "too large" in body_err.lower() or e.code == 413
            ):
                raise RequestTooLargeError(
                    "The context window is full. Use /prune or /compact to reduce context size."
                )
            raise ServerError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            err = ServerError(str(e))
            console.print(f"\n[red][bold]{err.title}[/bold] {err.message}[/red]")
            return "", 0, None

    try:
        obj = json.loads(body)
        msg = obj["choices"][0]["message"]
        text = msg.get("content") or ""
        total_tokens = (obj.get("usage") or {}).get("total_tokens", 0)
        tc_list = msg.get("tool_calls") or []
        if tc_list:
            tc = tc_list[0]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            tc_data = {
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "arguments": args,
                "raw": tc_list,
            }
            return text, total_tokens, tc_data
        return text, total_tokens, None
    except (json.JSONDecodeError, KeyError) as exc:
        raise ServerError(f"Unexpected response format: {exc}") from exc


def _chat_stream(messages: list[dict], config: EauDevConfig, tools: list[dict] | None = None) -> tuple[str, int, dict | None]:
    """Stream a chat completion. Returns (response_text, total_tokens_used, tool_call_data|None).

    Shows 'EauDev is thinking..' status bar via Rich Live until the first
    token arrives, then switches to raw streaming output.
    """
    inf = config.agent.inference
    payload_dict: dict = {
        "model":           inf.model,
        "messages":        messages,
        "stream":          True,
        "temperature":     inf.temperature,
        "top_p":           inf.top_p,
        "top_k":           inf.top_k,
        "min_p":           inf.min_p,
        "max_tokens":      inf.max_tokens,
        "enable_thinking": inf.enable_thinking,
    }
    if tools:
        payload_dict["tools"] = tools
    payload = json.dumps(payload_dict).encode()

    req = urllib.request.Request(
        inf.endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )

    chunks: list[str] = []
    total_tokens = 0
    in_think = False
    first_token_received = threading.Event()
    # Native tool_calls accumulator
    _tc_id: str = ""
    _tc_name: str = ""
    _tc_args: str = ""
    _tc_raw: list = []

    # Show status bar until the first token arrives
    def _show_status():
        with Live(
            _make_status_bar(), console=console,
            refresh_per_second=4, transient=True
        ):
            first_token_received.wait(timeout=120)

    status_thread = threading.Thread(target=_show_status, daemon=True)
    status_thread.start()

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    d = obj["choices"][0]["delta"]

                    # ── Native tool_calls (mlx_lm.server intercepts <tool_call> text) ──
                    tc_list = d.get("tool_calls") or []
                    if tc_list:
                        first_token_received.set()
                        for tc in tc_list:
                            if tc.get("id"):
                                _tc_id = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                _tc_name = fn["name"]
                            if fn.get("arguments"):
                                _tc_args += fn["arguments"]
                        if not _tc_raw:
                            _tc_raw = tc_list
                        usage = obj.get("usage") or {}
                        if usage.get("total_tokens"):
                            total_tokens = usage["total_tokens"]
                        continue

                    delta = d.get("content") or ""
                    if delta:
                        # Signal status bar to close on first real token
                        if not first_token_received.is_set():
                            first_token_received.set()
                            status_thread.join(timeout=0.5)

                        chunks.append(delta)

                        if not in_think and "<think>" in delta:
                            in_think = True
                            sys.stdout.write("\r\033[2m[thinking...]\033[0m")
                            sys.stdout.flush()
                        if in_think:
                            if "</think>" in delta:
                                in_think = False
                                sys.stdout.write("\r\033[K")
                                sys.stdout.flush()
                            else:
                                sys.stdout.write("\r\033[2m[thinking...]\033[0m")
                                sys.stdout.flush()
                        else:
                            # Strip any complete think blocks in this chunk before printing
                            printable = re.sub(r"<think>.*?</think>", "", delta, flags=re.DOTALL)
                            if printable:
                                print(printable, end="", flush=True)

                    usage = obj.get("usage") or {}
                    if usage.get("total_tokens"):
                        total_tokens = usage["total_tokens"]
                except (json.JSONDecodeError, KeyError):
                    continue
    except urllib.error.HTTPError as e:
        first_token_received.set()
        body = e.read().decode("utf-8", errors="replace")
        if e.code in (400, 413) and (
            "context" in body.lower() or "too large" in body.lower() or e.code == 413
        ):
            raise RequestTooLargeError(
                "The context window is full. Use /prune or /compact to reduce context size."
            )
        raise ServerError(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        first_token_received.set()
        err = ServerError(str(e))
        console.print(f"\n[red][bold]{err.title}[/bold] {err.message}[/red]")

    first_token_received.set()
    status_thread.join(timeout=1)

    if _tc_name:
        try:
            args_obj = json.loads(_tc_args) if _tc_args.strip() else {}
        except json.JSONDecodeError:
            args_obj = {}
        tc_data = {
            "id": _tc_id,
            "name": _tc_name,
            "arguments": args_obj,
            "raw": _tc_raw,
        }
        return "", total_tokens, tc_data

    print()
    return "".join(chunks), total_tokens, None


def _chat_complete(messages: list[dict], config: EauDevConfig, tools: list[dict] | None = None) -> tuple[str, int, dict | None]:
    """Dispatch to streaming or non-streaming completion based on config.agent.streaming."""
    if config.agent.streaming:
        return _chat_stream(messages, config, tools=tools)
    return _chat_no_stream(messages, config, tools=tools)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _make_status_bar(message: str = "EauDev is thinking..") -> Table:
    """Create a full-width status bar matching rovodev's bottom bar style."""
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(justify="right")
    table.add_row(
        f"[green]{message}[/green]",
        "[bright_black]ctrl+c to interrupt[/bright_black]",
    )
    return table


def _make_progress_status(step: str = "thinking") -> Table:
    """
    Enhanced progress status with step indicator.

    Steps: thinking → planning → tool → waiting → responding

    Inspired by modern AI coding assistants (Qwen Code, Cursor, etc.)
    Shows user exactly what stage the agent is in.
    """
    step_labels = {
        "thinking": "Analyzing request...",
        "planning": "Planning approach...",
        "tool": "Executing tool...",
        "waiting": "Waiting for result...",
        "responding": "Generating response...",
    }

    step_icons = {
        "thinking": "🧠",
        "planning": "📋",
        "tool": "🔧",
        "waiting": "⏳",
        "responding": "✍️",
    }

    # Check if terminal supports emoji
    icon = step_icons.get(step, "🧠")
    label = step_labels.get(step, "Working...")
    message = f"{icon}  {label}"

    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(justify="right")

    # Note: Spinner can't be used with Text.assemble() - would need Live display
    table.add_row(
        Text(f"⠋  {message}", style="green"),
        "[bright_black]ctrl+c to interrupt[/bright_black]",
    )
    return table


def _colorize_tool_arg(key: str, value: str) -> Text:
    """Return a Rich Text object with colourized tool argument values."""
    t = Text()
    t.append(f"      ● ", style="green")
    t.append(f"{key}: ", style="default")
    # Paths → cyan; short values → green; long content → dim truncation
    if key in ("path", "file_path") or (isinstance(value, str) and value.startswith("/")):
        t.append(value, style="cyan")
    elif key == "command":
        t.append(value, style="bold yellow")
    elif len(value) > 120:
        t.append(value[:117] + "...", style="bright_black")
    else:
        t.append(repr(value), style="green")
    return t


def _render_tool_result(result: str) -> None:
    """Render tool result intelligently — diffs in syntax panel, plain text dim."""
    first_line = result.split("\n")[0] if result else ""

    # Diff output
    if result.startswith("---") or result.startswith("+++") or "\n+++ " in result or "\n--- " in result:
        console.print(Syntax(result, "diff", theme="monokai", line_numbers=False))
        return

    # Bash output — already streamed live during execution, so only show the
    # header line (exit code + command) here. Never re-print the body — it was
    # already visible as it ran.
    if first_line.startswith("[run_bash"):
        console.print(f"  [bright_black]{first_line}[/bright_black]")
        return

    # File reads — show header + snippet
    if first_line.startswith("[read_file"):
        lines = result.split("\n", 1)
        console.print(f"  [bright_black]{lines[0]}[/bright_black]")
        return

    # Directory listing — show header + all entries
    if first_line.startswith("[list_directory"):
        console.print(f"  [bright_black]{result}[/bright_black]")
        return

    # File ops — create/move/delete/write — show single-line result dim
    if (
        first_line.startswith("[create_file")
        or first_line.startswith("[move_file")
        or first_line.startswith("[delete_file")
        or first_line.startswith("[write_file")
    ):
        console.print(f"  [bright_black]{first_line}[/bright_black]")
        return

    # MCP tool results — show first line of JSON or full result if short
    if first_line.startswith("{") or first_line.startswith("["):
        if len(result) <= 300:
            console.print(f"  [bright_black]{result}[/bright_black]")
        else:
            console.print(f"  [bright_black]{result[:200]}...[/bright_black]")
        return

    # Default — just show first line dim
    snippet = first_line[:100] + ("..." if len(first_line) > 100 else "")
    console.print(f"  [bright_black]{snippet}[/bright_black]")


def _get_context_limit(endpoint: str) -> int:
    """Query the inference server for the model's context window size.

    Tries llama.cpp-specific endpoints first, then falls back to 32768.
    mlx_lm.server doesn't expose context size via API — use config value instead.

    Sources tried in order:
      1. /props  → default_generation_settings.n_ctx  (llama.cpp >= 0.0.1)
      2. /props  → n_ctx  (some older builds)
      3. /slots  → n_ctx  (slot-level context size)
    Falls back to 32768 if all fail (mlx_lm.server case).
    """
    try:
        base = endpoint.rsplit("/v1", 1)[0]
        # Source 1 + 2: /props
        try:
            req = urllib.request.Request(f"{base}/props")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            # Preferred location in recent llama.cpp builds
            n_ctx = (
                data.get("default_generation_settings", {}).get("n_ctx")
                or data.get("n_ctx")
            )
            if n_ctx:
                return int(n_ctx)
        except Exception as e:
            logger.debug(f"Context limit: /props failed: {e}")
        # Source 3: /slots
        try:
            req = urllib.request.Request(f"{base}/slots", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                slots = json.loads(resp.read())
            if slots and isinstance(slots, list):
                n_ctx = slots[0].get("n_ctx")
                if n_ctx:
                    return int(n_ctx)
        except Exception as e:
            logger.debug(f"Context limit: /slots failed: {e}")
    except Exception as e:
        logger.debug(f"Context limit query failed for {endpoint}: {e}")
    return 32768


# ── Session title generation ───────────────────────────────────────────────────

def _generate_session_title(messages: list[dict], config: EauDevConfig) -> str:
    """Generate a short session title via a single non-streaming LLM request."""
    first_user_msg = ""
    for m in messages:
        if m.get("role") == "user":
            first_user_msg = str(m.get("content", ""))
            break

    fallback = first_user_msg[:50]

    if not first_user_msg:
        return fallback

    inf = config.agent.inference
    prompt = (
        f"Generate a 4-6 word title for this conversation. "
        f"Reply with ONLY the title, no quotes, no punctuation at the end. "
        f"First message: {first_user_msg}"
    )
    payload = json.dumps({
        "model": inf.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 20,
    }).encode()

    req = urllib.request.Request(
        inf.endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            title = data["choices"][0]["message"]["content"].strip()
            return title if title else fallback
    except Exception:
        return fallback


# ── Agentic loop ───────────────────────────────────────────────────────────────

def _write_context_metrics(session_id: str, used_tokens: int, total_tokens: int) -> None:
    """Write context metrics to /tmp for the context_monitor hook to read."""
    if not session_id or total_tokens <= 0:
        return
    metrics = {
        "used_tokens":    used_tokens,
        "total_tokens":   total_tokens,
        "remaining_pct":  round(100.0 * (1 - used_tokens / total_tokens), 1) if total_tokens > 0 else 100.0,
        "timestamp":      time.time(),
    }
    path = Path(tempfile.gettempdir()) / f"eaudev-ctx-{session_id}.json"
    try:
        path.write_text(json.dumps(metrics))
    except Exception:
        pass


def _get_n_past(endpoint: str) -> int:
    """Query llama.cpp /slots for actual KV cache usage (n_past tokens in context)."""
    try:
        base = endpoint.rsplit("/v1", 1)[0]
        req = urllib.request.Request(f"{base}/slots", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1) as resp:
            slots = json.loads(resp.read())
            if slots and isinstance(slots, list):
                return slots[0].get("n_past", 0)
    except Exception as e:
        logger.debug(f"Failed to get n_past from /slots: {e}")
    return 0


def _fallback_to_smaller_model(config: EauDevConfig) -> bool:
    """Switch to the next smaller model in the registry. Returns True if switched."""
    try:
        from eaudev.modules.model_registry import load_model_registry, switch_model
        models = load_model_registry()
        if not models:
            return False
        # Sort by size, find current, pick next smaller
        sorted_models = sorted(models, key=lambda m: m.size_gb)
        current_endpoint = config.agent.inference.endpoint
        current_idx = next(
            (i for i, m in enumerate(sorted_models) if m.endpoint == current_endpoint), -1
        )
        if current_idx <= 0:
            return False  # already on smallest model
        target = sorted_models[current_idx - 1]
        console.print(
            f"[yellow]Context too large for current model. "
            f"Falling back to {target.name} ({target.size_display})...[/yellow]"
        )
        success = switch_model(target)
        if success:
            console.print(f"[green]Switched to {target.name}. Retrying...[/green]")
        return success
    except Exception as e:
        logger.warning(f"Fallback failed: {e}")
        return False


def _run_agent(session: Session, perms: ToolPermissionManager, config: EauDevConfig, workdir: str, context_limit: int = 32768) -> int:
    """Run the agentic tool loop. Returns the n_past token count from llama.cpp /slots."""
    max_iterations = 20
    last_tokens = 0
    resp_tokens = 0

    # Fire SessionStart hooks ONCE at the start of _run_agent (first user turn)
    # Check if already fired by looking for the marker message
    _session_start_fired = any(
        m.get("content") and m.get("content", "").startswith("[Session context from hooks]")
        for m in session.message_history
    )
    if config and config.hooks and not _session_start_fired:
        model_id = config.agent.inference.model if config.agent else ""
        session_ctx = run_session_start_hooks(
            session_id=session.session_id,
            hooks_cfg=config.hooks,
            context={"workdir": workdir, "model": model_id},
        )
        if session_ctx:
            # Inject AFTER the system prompt (index 1), not before it
            messages = session.message_history
            insert_idx = 1 if (messages and messages[0].get("role") == "system") else 0
            messages.insert(insert_idx, {"role": "user", "content": f"[Session context from hooks]\n{session_ctx}"})
            messages.insert(insert_idx + 1, {"role": "assistant", "content": "Understood, I have the session context."})

    # Build the full tools list: 7 built-ins + MCP tools (excluding hidden servers)
    _HIDDEN_MCP_SERVERS = frozenset({"memory"})
    all_tools = _EAUDEV_TOOLS + get_mcp_manager().build_openai_tools(exclude_servers=_HIDDEN_MCP_SERVERS)

    for i in range(max_iterations):
        # ── Response header (only on first turn, not every tool iteration) ───────────
        # Rule removed — response now shown in Panel below

        try:
            response, resp_tokens, tc_data = _chat_complete(session.message_history, config, tools=all_tools)
        except RequestTooLargeError:
            if _fallback_to_smaller_model(config):
                try:
                    response, resp_tokens, tc_data = _chat_complete(session.message_history, config, tools=all_tools)
                except RequestTooLargeError:
                    console.print(
                        "[red]Context still too large after fallback. Use /prune or /compact.[/red]"
                    )
                    return last_tokens
            else:
                console.print(
                    "[red]Context too large and no smaller model available. Use /prune or /compact.[/red]"
                )
                console.print(
                    "  [dim]Tip: /prune removes tool results, /compact summarizes middle turns.[/dim]"
                )
                return last_tokens

        # Use token count from response (mlx_lm returns this via standard OpenAI usage field)
        if resp_tokens > 0:
            last_tokens = resp_tokens

        # Write context metrics for hooks (context_monitor.py reads this)
        _write_context_metrics(
            session_id=getattr(session, "id", ""),
            used_tokens=last_tokens,
            total_tokens=context_limit,
        )

        get_memory_store().record_turn("assistant", response)

        # Determine if we have a native tool call or a prose response
        if tc_data:
            # Native tool call from mlx_lm.server — store with tool_calls array
            session.message_history.append({
                "role": "assistant",
                "content": response or None,
                "tool_calls": tc_data["raw"],
            })
            tool_call = {"name": tc_data["name"], **tc_data["arguments"]}
        else:
            # Prose response (or fallback text-format tool call)
            session.message_history.append({"role": "assistant", "content": response})
            tool_call, preamble = _extract_tool_call(response, model_name=config.agent.inference.model if config else "")

        if tool_call is None:
            # Non-streaming: response was not printed during generation — print it now.
            if not config.agent.streaming:
                fmt = config.console.output_format
                response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                if fmt == "markdown":
                    console.print(Panel(
                        Markdown(response_clean),
                        title="[green]Response[/green]",
                        border_style="green",
                        padding=(1, 2),
                    ))
                else:
                    console.print(Panel(
                        response_clean,
                        title="[green]Response[/green]",
                        border_style="green",
                        padding=(1, 2),
                    ))

            # Final prose response — check context fill and auto-compact if critical
            if last_tokens > 0 and context_limit > 0:
                fill_pct = last_tokens / context_limit
                if fill_pct >= 0.90:
                    console.print(
                        f"\n[red]⚠ Context {fill_pct:.0%} full ({last_tokens:,}/{context_limit:,} tokens). "
                        f"Auto-compacting…[/red]"
                    )
                    _compact_cmd(_session=session, _config=config, _message_list=[])
                elif fill_pct >= 0.75:
                    console.print(
                        f"\n[yellow]Context {fill_pct:.0%} full ({last_tokens:,}/{context_limit:,} tokens). "
                        f"Use [bold]/compact[/bold] to free space.[/yellow]"
                    )
            break

        # ── Tool call display (rovodev style) ──────────────────────────────────
        tool_name = tool_call.get("name", "unknown")
        tool_args = {k: v for k, v in tool_call.items() if k != "name"}

        if config.console.show_tool_results:
            console.print(f"\n  [green]●[/green] Called [bold green]{tool_name}:[/bold green]")
            for k, v in tool_args.items():
                console.print(_colorize_tool_arg(k, str(v)))

        # Show progress indicator — use enhanced step-based status
        # Note: We don't actually show the indicator here because permission dialogs
        # need a clean terminal. The _make_progress_status function is kept for future use.
        result = _dispatch_tool(tool_call, perms, workdir, config=config, session_id=session.session_id)
        # PostToolUse hooks
        post_ctx = None
        if config.hooks.enabled:
            post_ctx = run_post_tool_hooks(tool_name, tool_args, result, config.hooks, session_id=session.session_id)

        if config.console.show_tool_results:
            _render_tool_result(result)

        # Inject tool result — use role:tool with tool_call_id for native calls
        if tc_data:
            session.message_history.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "name": tool_name,
                "content": result,
            })
        else:
            session.message_history.append({"role": "user", "content": result})

        if post_ctx:
            session.message_history.append({"role": "user", "content": f"[Hook context]\n{post_ctx}"})
            session.message_history.append({"role": "assistant", "content": "Noted."})
    else:
        console.print("[bright_black][EauDev: reached max tool iterations — stopping][/bright_black]")

    return last_tokens


# ── Slash commands ─────────────────────────────────────────────────────────────

@registry.register("/sessions", None, "View and manage sessions.")
def _sessions_cmd(*args, **kwargs):
    session: Session = kwargs["_session"]
    persistence_dir: Path = kwargs["_persistence_dir"]
    workspace_path = Path(session.workspace_path) if session.workspace_path else Path.cwd()
    sessions = get_sessions(persistence_dir, workspace_path=workspace_path)
    selected, is_new = session_menu_panel_sync(sessions, session.session_id, persistence_dir)
    if is_new or (selected is None):
        # Start fresh session
        return {"continue": True, "new_session": True}
    if selected and selected.session_id != session.session_id:
        # Switch to selected session
        return {"continue": True, "switch_session": selected}
    return {"continue": True}


@registry.register("/clear", None, "Clear the current session's message history.")
def _clear_cmd(*args, **kwargs):
    logger.bind(role="info").info("Session history cleared")
    return {"continue": True, "message_history": None}


@registry.register(
    "/prune", None,
    "Reduce session token size by stripping tool results from history."
)
def _prune_cmd(*args, **kwargs):
    session: Session = kwargs["_session"]
    history = session.message_history
    original_len = len(history)

    # Keep system prompt (index 0) and all non-tool-result messages.
    # Tool results are user-role messages whose content starts with "[" (our tool result format).
    pruned = []
    for i, msg in enumerate(history):
        if i == 0:
            pruned.append(msg)  # always keep system prompt
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Skip native tool result messages (role:tool)
        if role == "tool":
            continue
        # Skip assistant messages that are pure tool calls (no prose content)
        if role == "assistant" and msg.get("tool_calls") and not content:
            continue
        # Skip legacy user messages that are tool results (produced by _dispatch_tool)
        # Matches: native tool results ("[read_file..."), MCP results ("[mcp:..."), and
        # tool_response wrappers ("<tool_response>")
        if role == "user" and isinstance(content, str) and (
            content.startswith("[read_file")
            or content.startswith("[write_file")
            or content.startswith("[run_bash")
            or content.startswith("[list_directory")
            or content.startswith("[create_file")
            or content.startswith("[move_file")
            or content.startswith("[delete_file")
            or content.startswith("[unknown tool")
            or content.startswith("[mcp:")
            or content.startswith("<tool_response>")
        ):
            continue
        pruned.append(msg)

    removed = original_len - len(pruned)
    session.message_history = pruned
    console.print(
        f"  [bright_black]Pruned {removed} tool result message(s). "
        f"History: {original_len} → {len(pruned)} messages.[/bright_black]"
    )
    # Signal the main loop to re-query /slots for an accurate token count.
    # We cannot call _get_n_past here because we don't have the endpoint, but
    # the main loop does — it will refresh last_tokens on the next iteration.
    return {"continue": True, "refresh_tokens": True}


@registry.register("/compact", None, "Summarise and truncate context to reduce token usage.")
def _compact_cmd(*args, **kwargs):
    """
    Compress context using a head+tail protection strategy (Hermes-style):
      - Always keep: system prompt + first PROTECT_HEAD user/assistant turns
      - Always keep: last PROTECT_TAIL user/assistant turns (recent work)
      - Summarise: everything in between using the local model
      - Fallback: mechanical bullet-point summary if LLM call fails

    This preserves the most important context (initial task framing + recent state)
    while aggressively compressing stale middle turns.
    """
    PROTECT_HEAD = 2   # turns (user+assistant pairs) to keep from start
    PROTECT_TAIL = 3   # turns (user+assistant pairs) to keep from end

    session: Session = kwargs["_session"]
    config: EauDevConfig = kwargs["_config"]
    history = session.message_history

    # Separate system prompt from conversation turns
    system_msg = history[0] if history and history[0].get("role") == "system" else None
    turns = [m for m in history if m.get("role") != "system"]

    # Need enough turns to make compression worthwhile
    min_turns = PROTECT_HEAD * 2 + PROTECT_TAIL * 2 + 2
    if len(turns) < min_turns:
        console.print("  [bright_black]Nothing to compact — context is already minimal.[/bright_black]")
        return {"continue": True}

    before_tokens = sum(len(str(m.get("content", ""))) for m in history) // 4

    # Slice into head / middle / tail
    head_turns = turns[: PROTECT_HEAD * 2]
    tail_turns = turns[-(PROTECT_TAIL * 2):]
    middle_turns = turns[PROTECT_HEAD * 2 : -(PROTECT_TAIL * 2)]

    if not middle_turns:
        console.print("  [bright_black]Nothing to compact — no middle turns to summarise.[/bright_black]")
        return {"continue": True}

    # Build summarisation prompt for the middle turns
    def _format_turns_for_summary(msgs: list[dict]) -> str:
        parts = []
        for m in msgs:
            role = m.get("role", "?").upper()
            content = str(m.get("content", ""))
            if len(content) > 1500:
                content = content[:800] + "\n...[truncated]...\n" + content[-400:]
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    summary: str | None = None
    try:
        inf = config.agent.inference
        summarise_messages = [
            {
                "role": "user",
                "content": (
                    "Summarise the following conversation turns concisely.\n"
                    "Write from a neutral perspective. Cover:\n"
                    "- What the user asked or requested\n"
                    "- What actions were taken (files created/modified, commands run)\n"
                    "- Key decisions or findings\n"
                    "- Any errors encountered and how they were resolved\n"
                    "Be specific and factual. Target 150–250 words.\n\n"
                    "TURNS TO SUMMARISE:\n\n"
                    + _format_turns_for_summary(middle_turns)
                ),
            }
        ]
        payload = json.dumps({
            "model": inf.model,
            "messages": summarise_messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 600,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            inf.endpoint, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            raw = result["choices"][0]["message"].get("content", "") or \
                  result["choices"][0]["message"].get("reasoning_content", "")
            summary = raw.strip()
    except Exception as exc:
        logger.debug(f"/compact LLM summarisation failed: {exc}")

    # Build compressed summary message
    if summary:
        summary_block = f"[Context compressed — {len(middle_turns)} turns summarised]\n\n{summary}"
    else:
        # Mechanical fallback — bullet points from middle turns
        lines = ["[Context compressed — mechanical summary]"]
        for m in middle_turns:
            role = m.get("role", "?")
            content = str(m.get("content", ""))
            if role == "user" and content.startswith("["):
                continue  # skip injected summaries / system messages
            snippet = content[:250].replace("\n", " ").strip()
            if snippet:
                label = "User" if role == "user" else "Assistant"
                lines.append(f"- {label}: {snippet}{'…' if len(content) > 250 else ''}")
        summary_block = "\n".join(lines)

    # Reassemble: system → head → summary injection → tail
    new_history: list[dict] = []
    if system_msg:
        new_history.append(system_msg)
    new_history.extend(head_turns)
    new_history.append({"role": "user", "content": summary_block})
    new_history.append({"role": "assistant", "content": "Understood — I have the summary of prior work. Continuing."})
    new_history.extend(tail_turns)

    session.message_history = new_history
    after_tokens = sum(len(str(m.get("content", ""))) for m in new_history) // 4

    mode = "LLM summary" if summary else "mechanical fallback"
    console.print(
        f"  [bright_black]Compacted ({mode}): "
        f"~{before_tokens:,} → ~{after_tokens:,} tokens "
        f"({len(middle_turns)} middle turns compressed, "
        f"{len(head_turns)} head + {len(tail_turns)} tail preserved)[/bright_black]"
    )

    # ── Bridge to Memory MCP episodic layer ───────────────────────────────────
    # Write the compression summary as an episodic event so future sessions
    # can recall what happened in this one, even after context is compacted.
    if summary:
        try:
            store = get_memory_store()
            if store.available and store._episodic is not None:
                session_id = getattr(session, "session_id", None) or getattr(session, "id", "unknown")
                user_count = sum(1 for m in middle_turns if m.get("role") == "user")
                asst_count = sum(1 for m in middle_turns if m.get("role") == "assistant")
                store._episodic.store_episode(
                    session_id=str(session_id),
                    summary=f"[Mid-session compact — {len(middle_turns)} turns]\n\n{summary}",
                    keywords=["compact", "context-compression", "mid-session"],
                    turn_count=len(middle_turns),
                    user_turns=user_count,
                    assistant_turns=asst_count,
                )
        except Exception:
            pass  # Memory MCP bridge is always silent — never blocks the run loop

    return {"continue": True, "reset_tokens": after_tokens}


@registry.register("/shadow", None, "Ask the model a question without affecting the main context.")
def _shadow_cmd(*args, **kwargs):
    """Run a shadow query — model sees current context snapshot but response is discarded."""
    message_list: list[str] = kwargs.get("_message_list", [])
    session: Session = kwargs["_session"]
    config: EauDevConfig = kwargs["_config"]

    prompt = " ".join(message_list).strip()
    if not prompt:
        console.print("  [bright_black]Usage: /shadow <question>[/bright_black]")
        console.print("  [bright_black]Asks the model a question using a snapshot of the current context.[/bright_black]")
        console.print("  [bright_black]The response is shown but never added to the session.[/bright_black]")
        return {"continue": True}

    # Fork the context — snapshot only, never mutate the real history
    shadow_messages = [m.copy() for m in session.message_history]
    shadow_messages.append({"role": "user", "content": prompt})

    inf = config.agent.inference
    payload = json.dumps({
        "model": inf.model,
        "messages": shadow_messages,
        "stream": False,
        "temperature": inf.temperature,
        "max_tokens": inf.max_tokens,
    }).encode()

    req = urllib.request.Request(
        inf.endpoint, data=payload,
        headers={"Content-Type": "application/json"},
    )

    console.print()
    console.print(Rule(" ◌ shadow ", style="dim"))

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            response = result["choices"][0]["message"]["content"].strip()

        # Strip <think>...</think> blocks for clean shadow output
        response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        console.print(
            Panel(
                Markdown(response_clean) if response_clean else "[dim](empty response)[/dim]",
                border_style="dim",
                padding=(1, 2),
                subtitle="[dim]◌ not saved to context[/dim]",
            )
        )
    except urllib.error.URLError as e:
        console.print(f"  [red]shadow: connection error — {e}[/red]")
    except Exception as e:
        console.print(f"  [red]shadow: error — {e}[/red]")

    console.print(Rule(style="dim"))
    console.print()

    # Return without adding anything to session history
    return {"continue": True}


@registry.register("#", None, "Add a quick note to workspace .agent.md.")
def _memory_note_cmd(*args, **kwargs):
    note = " ".join(kwargs.get("_message_list", []))
    handle_memory_note(note)
    return {"continue": True}


@registry.register("#!", None, "Remove a note from workspace .agent.md.")
def _memory_note_remove_cmd(*args, **kwargs):
    note = " ".join(kwargs.get("_message_list", []))
    handle_memory_note("!" + note)
    return {"continue": True}


@registry.register("/instructions", None, "Run saved instructions (stub).")
def _instructions_cmd(*args, **kwargs):
    instruction_args = " ".join(kwargs.get("_message_list", []))
    message = handle_instructions_command(instruction_args or None)
    return {"message": message, "continue": bool(message)}


@registry.register("/memory", None, "Edit workspace .agent.md in $EDITOR.")
def _memory_cmd(*args, **kwargs):
    message = handle_memory_command(None)
    return {"message": message}


@registry.register("/memory", "user", "Edit ~/.eaudev/.agent.md in $EDITOR.")
def _memory_user_cmd(*args, **kwargs):
    message = handle_memory_command("user")
    return {"message": message}


@registry.register("/memory", "init", "Prompt agent to create/update .agent.md.")
def _memory_init_cmd(*args, **kwargs):
    message = handle_memory_command("init")
    return {"message": message}


@registry.register("/memory", "stats", "Show persistent memory store diagnostics.")
def _memory_stats_cmd(*args, **kwargs):
    """Display live stats from all EauDevMemoryStore layers."""
    store = get_memory_store()

    if not store.available:
        console.print("[yellow]Memory store unavailable — Memory MCP layers not found.[/yellow]")
        return {"continue": True}

    stats = store.get_stats()

    obs      = stats.get("observations", {})
    episodic = stats.get("episodic", {})
    facts    = stats.get("facts", {})
    fts5     = stats.get("fts5", {})
    graph    = stats.get("graph", {})

    # Abbreviate UUID scope labels for readability
    scope_display = obs.get('scope', '?')
    if isinstance(scope_display, str) and len(scope_display) == 36 and scope_display.count('-') == 4:
        scope_display = f"session ({scope_display[:8]}...)"

    console.print()
    console.print("[bold]Memory Store[/bold]  [dim](~/.eaudev/)[/dim]")
    console.print()
    console.print(f"  [cyan]Observations[/cyan]   {obs.get('turn_count', 0):>6} turns     "
                  f"scope: [dim]{scope_display}[/dim]   "
                  f"max: [dim]{obs.get('max_turns', '?')}[/dim]")
    console.print(f"  [cyan]Episodic[/cyan]       {episodic.get('total_episodes', 0):>6} episodes  "
                  f"avg turns: [dim]{episodic.get('avg_turn_count', 0)}[/dim]   "
                  f"size: [dim]{episodic.get('db_size_bytes', 0) // 1024}KB[/dim]")
    console.print(f"  [cyan]Facts[/cyan]          {facts.get('total_facts', 0):>6} facts     "
                  f"categories: [dim]{facts.get('unique_categories', 0)}[/dim]   "
                  f"size: [dim]{facts.get('db_size_bytes', 0) // 1024}KB[/dim]")
    console.print(f"  [cyan]FTS5 Index[/cyan]     {fts5.get('total_documents', 0):>6} docs      "
                  f"size: [dim]{fts5.get('db_size_bytes', 0) // 1024}KB[/dim]")
    console.print(f"  [cyan]Graph[/cyan]          {graph.get('entities', 0):>6} entities  "
                  f"relationships: [dim]{graph.get('relationships', 0)}[/dim]   "
                  f"status: [dim]{graph.get('status', '?')}[/dim]")
    console.print()

    return {"continue": True}


@registry.register("/think", None, "Toggle or set Qwen3 thinking mode. Subcommands: on, off (default: toggle).")
def _think_cmd(*args, **kwargs):
    """Toggle extended reasoning (<think> blocks) on or off.

    /think         → toggle
    /think on      → enable thinking mode
    /think off     → disable thinking mode (faster, less verbose)
    """
    message_list: list[str] = kwargs.get("_message_list", [])
    config: EauDevConfig = kwargs["_config"]
    session: Session = kwargs["_session"]

    sub = message_list[0].lower() if message_list else "toggle"

    if sub == "on":
        new_state = True
    elif sub == "off":
        new_state = False
    else:  # "toggle"
        new_state = not config.agent.inference.enable_thinking

    config.agent.inference.enable_thinking = new_state
    label = "[green]ON[/green]" if new_state else "[dim]off[/dim]"
    directive = "" if new_state else "/no_think"
    console.print(f"  Thinking mode: {label}")

    # Update the system prompt in the current session to reflect the new directive
    if session.message_history and session.message_history[0].get("role") == "system":
        current_sys = session.message_history[0]["content"]
        # Replace existing thinking directive line (use word boundary, not end-of-string)
        current_sys = re.sub(r"\n/no_think\b", "", current_sys)
        current_sys = re.sub(r"\n$", "", current_sys)
        if directive:
            current_sys = current_sys + "\n" + directive
        session.message_history[0]["content"] = current_sys

    return {"continue": True}


@registry.register("/memory", "graph", "Query the knowledge graph for an entity and its relationships.")
def _memory_graph_cmd(*args, **kwargs):
    """Query graph.db for an entity and show its 1-2 level relationships."""
    message_list: list[str] = kwargs.get("_message_list", [])
    entity_query = " ".join(message_list).strip()

    if not entity_query:
        console.print("  [bright_black]Usage: /memory graph <entity name>[/bright_black]")
        console.print("  [bright_black]Example: /memory graph biquad filter[/bright_black]")
        return {"continue": True}

    store = get_memory_store()
    if not store.available or store._graph is None:
        console.print("[yellow]Knowledge graph unavailable.[/yellow]")
        return {"continue": True}

    try:
        results = store._graph.get_relationships(entity_query, direction="both")
        if not results:
            # Also try search_entities in case the name is a partial match
            matches = store._graph.search_entities(entity_query, limit=5)
            if matches:
                # Found entities but no relationships — show them
                console.print()
                console.print(f"[bold]Graph: {entity_query}[/bold]")
                console.print()
                for m in matches:
                    console.print(f"  [cyan]{m['name']}[/cyan] [dim]({m['type']})[/dim]  — no relationships")
                console.print()
            else:
                console.print(f"  [bright_black]No graph entries found for: {entity_query}[/bright_black]")
            return {"continue": True}

        console.print()
        console.print(f"[bold]Graph: {entity_query}[/bold]")
        console.print()
        for r in results:
            src = r.get("source", "?")
            rel = r.get("relation_type", "?")
            tgt = r.get("target", "?")
            direction = r.get("direction", "outbound")
            arrow = "─→" if direction == "outbound" else "←─"
            console.print(f"  [cyan]{src}[/cyan] [dim]{arrow}{rel}→[/dim] [green]{tgt}[/green]")
        console.print()
    except Exception as e:
        console.print(f"  [red]Graph query error: {e}[/red]")

    return {"continue": True}


@registry.register("/server", None, "Show inference server status.")
def _server_cmd(*args, **kwargs):
    config: EauDevConfig = kwargs["_config"]
    endpoint = config.agent.inference.endpoint
    base = endpoint.rsplit("/v1/", 1)[0]
    table = Table.grid(expand=False, padding=(0, 2))
    table.add_column(style="bright_black")
    table.add_column()
    try:
        req = urllib.request.Request(f"{base}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            health = json.loads(resp.read())
        table.add_row("status", f"[green]{health.get('status', 'ok')}[/green]")
    except Exception as e:
        table.add_row("status", f"[red]unreachable — {e}[/red]")
        console.print(Panel(table, title="[bold]Inference Server[/bold]", border_style="bright_black"))
        return {"continue": True}

    table.add_row("endpoint", endpoint)
    table.add_row("model", config.agent.inference.model)
    table.add_row("thinking", "[green]on[/green]" if config.agent.inference.enable_thinking else "off")

    # Try llama.cpp /props (only present on llama-server, not mlx_lm.server)
    try:
        req2 = urllib.request.Request(f"{base}/props")
        with urllib.request.urlopen(req2, timeout=2) as resp2:
            props = json.loads(resp2.read())
        table.add_row("n_ctx", str(props.get("n_ctx", "?")))
        table.add_row("slots", str(props.get("total_slots", 1)))
        table.add_row("backend", "llama-server")
    except Exception:
        table.add_row("backend", "mlx_lm.server")

    # /v1/models — works on both mlx_lm.server and llama-server
    try:
        req3 = urllib.request.Request(f"{base}/v1/models")
        with urllib.request.urlopen(req3, timeout=3) as resp3:
            models_data = json.loads(resp3.read())
            model_ids = [m.get("id", "?") for m in models_data.get("data", [])]
            table.add_row("loaded models", ", ".join(model_ids) or "?")
    except Exception:
        pass

    console.print(Panel(table, title="[bold]Inference Server[/bold]", border_style="bright_black"))
    return {"continue": True}


@registry.register("/model", None, "Switch inference model (stops/starts llama-server).")
def _model_cmd(*args, **kwargs):
    from eaudev.modules.model_registry import load_model_registry, switch_model
    from eaudev.ui.components.user_menu_panel import Choice, user_menu_panel_sync

    config: EauDevConfig = kwargs["_config"]
    models = load_model_registry()

    if not models:
        console.print(
            "[yellow]No models found in ~/.eaudev/models.yml[/yellow]\n"
            "[bright_black]Create or edit it with /config to add models.[/bright_black]"
        )
        return {"continue": True}

    current = config.agent.inference.model
    choices = [
        Choice(
            name=f"{'[green]●[/green] ' if m.name == current else '  '}{m.menu_label}",
            value=m.name,
        )
        for m in models
    ]

    # Find current selection index
    try:
        start_idx = next(i for i, m in enumerate(models) if m.name == current)
    except StopIteration:
        start_idx = 0

    selected_name = user_menu_panel_sync(
        choices=choices,
        title="Select Model",
        message=f"Current: [bold green]{current}[/bold green]  |  ↑↓ navigate  Enter select  q cancel",
        selection=start_idx,
        escape_return_value=current,
        action_name="Switch",
    )

    if not selected_name or selected_name == current:
        console.print("[bright_black]No change.[/bright_black]")
        return {"continue": True}

    # Find the selected config
    selected_cfg = next((m for m in models if m.name == selected_name), None)
    if not selected_cfg:
        return {"continue": True}

    # Warn about large models
    if selected_cfg.size_gb >= 30:
        console.print(
            f"[yellow]⚠  {selected_cfg.display} is {selected_cfg.size_display} "
            f"— uses ~{selected_cfg.size_gb / 64 * 100:.0f}% of your 64GB RAM.[/yellow]"
        )

    # Stop current server, start new one, wait for ready
    ok = switch_model(selected_cfg)
    if ok:
        config.agent.inference.model = selected_cfg.api_model_name
        config.agent.inference.endpoint = selected_cfg.endpoint
        # Update context limit for the token bar
        return {"continue": True, "new_context_limit": selected_cfg.context}

    return {"continue": True}


@registry.register("/summarize", None, "Summarize a file or directory without filling context.")
def _summarize_cmd(*args, **kwargs):
    config: EauDevConfig = kwargs["_config"]
    session: Session = kwargs["_session"]
    target = " ".join(args).strip() if args else ""
    if not target:
        console.print("  [bright_black]Usage: /summarize <path>[/bright_black]")
        return {"continue": True}
    target_path = Path(os.path.expanduser(target))
    if not target_path.exists():
        console.print(f"  [red]Path not found: {target}[/red]")
        return {"continue": True}

    # Read content (full, bypassing truncation for summarization)
    if target_path.is_dir():
        try:
            entries = list(target_path.iterdir())
            file_list = "\n".join(
                f"{'[dir]' if e.is_dir() else f'[{e.stat().st_size:,}B]':>12}  {e.name}"
                for e in sorted(entries)
            )
            raw_content = f"Directory listing of {target}:\n{file_list}"
        except Exception as e:
            console.print(f"  [red]Error reading directory: {e}[/red]")
            return {"continue": True}
    else:
        try:
            full = target_path.read_text(encoding="utf-8", errors="replace")
            # Still cap at 64K for the summarization request itself
            if len(full) > 64_000:
                full = full[:64_000] + f"\n[... {len(full)-64_000:,} chars omitted for summarization]"
            raw_content = full
        except Exception as e:
            console.print(f"  [red]Error reading file: {e}[/red]")
            return {"continue": True}

    console.print(f"  [bright_black]Summarizing {target}...[/bright_black]")
    prompt = (
        f"Summarize the following in 150-200 words. Focus on: purpose, key components, "
        f"technologies/languages used, and anything notable. Be concise and factual.\n\n"
        f"File: {target}\n\n{raw_content}"
    )

    messages = [
        {"role": "system", "content": "You are a concise technical summarizer. Respond with a single focused paragraph."},
        {"role": "user", "content": prompt},
    ]

    inf = config.agent.inference
    payload = json.dumps({
        "model": inf.model,
        "messages": messages,
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 400,
    }).encode()

    try:
        req = urllib.request.Request(
            inf.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            summary = data["choices"][0]["message"]["content"].strip()

        console.print(f"\n[bold]Summary of {target_path.name}:[/bold]")
        console.print(summary)
        console.print()

        # Inject summary into session context instead of raw content
        session.message_history.append({
            "role": "user",
            "content": f"[/summarize {target}]\n{summary}"
        })
        session.message_history.append({
            "role": "assistant",
            "content": f"I've summarized {target_path.name} and added it to our context."
        })
    except Exception as e:
        console.print(f"  [red]Summarization failed: {e}[/red]")

    return {"continue": True}


@registry.register("/config", None, "Open EauDev config file in $EDITOR.")
def _config_cmd(*args, **kwargs):
    open_file_in_editor(str(CONFIG_PATH), create_if_missing=True)
    return {"continue": True}


@registry.register("/log", None, "Open EauDev log file in $EDITOR.")
def _log_cmd(*args, **kwargs):
    config: EauDevConfig = kwargs.get("_config")
    log_path = config.logging.path if config else str(Path.home() / ".eaudev" / "eaudev.log")
    open_file_in_editor(log_path, create_if_missing=True)
    return {"continue": True}


@registry.register("/voice", None, "Toggle voice I/O mode. Subcommands: on, off, status, config.")
def _voice_cmd(*args, **kwargs):
    """Activate / deactivate VoiceIO mode or show status."""
    message_list: list[str] = kwargs.get("_message_list", [])
    sub = message_list[0].lower() if message_list else "on"
    voice = get_voice_io()
    config = kwargs.get("_config", None)

    if sub == "off":
        if voice.active:
            voice.stop()
            console.print("[dim]VoiceIO deactivated.[/dim]")
        else:
            console.print("[dim]VoiceIO is not active.[/dim]")
        return {"continue": True}

    elif sub == "status":
        if voice.active:
            cfg = voice.config
            console.print(f"[green]VoiceIO active[/green] — whisper:{cfg.whisper_model}  piper:{cfg.piper_model or 'not set'}")
        else:
            console.print("[dim]VoiceIO inactive.[/dim]")
            missing = check_dependencies()
            if missing:
                console.print("[yellow]Missing dependencies:[/yellow]")
                for m in missing:
                    console.print(f"  [dim]• {m}[/dim]")
        return {"continue": True}

    elif sub == "config":
        open_file_in_editor(str(Path.home() / ".eaudev" / "config.yml"), create_if_missing=False)
        return {"continue": True}

    else:  # "on" or bare /voice
        if voice.active:
            console.print("[dim]VoiceIO is already active.[/dim]")
            return {"continue": True}
        # Build VoiceIOConfig from the typed config.voice_io Pydantic model
        voice_cfg = VoiceIOConfig()
        if config is not None:
            from eaudev.common.config_model import VoiceIOConfig as _VoiceIOCfgModel
            raw = getattr(config, "voice_io", None)
            if raw is not None and isinstance(raw, _VoiceIOCfgModel):
                # Copy all fields from the config model into the VoiceIO dataclass
                for field_name in raw.model_fields:
                    val = getattr(raw, field_name, None)
                    if val is not None and hasattr(voice_cfg, field_name):
                        setattr(voice_cfg, field_name, val)
        
        # Auto-detect piper_model if not configured
        if not voice_cfg.piper_model and config is not None:
            detected = config.voice_io.get_piper_model()
            if detected:
                voice_cfg.piper_model = detected
                console.print(f"[dim]Auto-detected Piper model: {detected}[/dim]")
            else:
                console.print("[yellow]No Piper model found. TTS will be disabled.[/yellow]")
                console.print("[dim]Install a model or set voice_io.piper_model in config.[/dim]")
        
        voice.__init__(voice_cfg)
        try:
            voice.start()
            console.print("[green]VoiceIO active[/green] — speak your prompt, then pause.")
        except Exception as exc:
            console.print(f"[red]VoiceIO failed to start:[/red] {exc}")
        return {"continue": True}


@registry.register("/mcp", None, "Manage MCP servers. Subcommands: status, config (default).")
def _mcp_cmd(*args, **kwargs):
    message_list: list[str] = kwargs.get("_message_list", [])
    subcommand = message_list[0].lower() if message_list else "config"

    if subcommand == "status":
        mcp = get_mcp_manager()
        servers = mcp.server_status()
        if not servers:
            console.print("  [bright_black]No MCP servers configured in ~/.eaudev/mcp.json[/bright_black]")
            console.print("  [bright_black]Run /mcp config to edit the config file.[/bright_black]")
            return {"continue": True}

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Server", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Tools", justify="right")
        table.add_column("Tool Names", style="bright_black")

        for name, info in servers.items():
            alive = info.get("alive", False)
            status_str = "[green]running[/green]" if alive else "[red]stopped[/red]"
            tool_count = str(info.get("tool_count", 0))
            tool_names = ", ".join(info.get("tools", [])[:6])
            if len(info.get("tools", [])) > 6:
                tool_names += f" … +{len(info['tools']) - 6} more"
            table.add_row(name, status_str, tool_count, tool_names)

        console.print()
        console.print(table)
        console.print()
        return {"continue": True}

    # Default: open config in editor
    mcp_path = Path.home() / ".eaudev" / "mcp.json"
    if not mcp_path.exists():
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text('{\n    "mcpServers": {}\n}')
    open_file_in_editor(str(mcp_path), create_if_missing=False)
    return {"continue": True}


@registry.register("/hooks", None, "Manage EauDev hooks. Subcommands: list, add, remove, enable, disable, test.")
def _hooks_cmd(*args, **kwargs):
    from eaudev.common.config import save_config
    from eaudev.common.config_model import HookEntry
    from eaudev.modules.hooks import run_pre_tool_hooks

    message_list: list[str] = kwargs.get("_message_list", [])
    config: EauDevConfig = kwargs["_config"]
    hooks_cfg = config.hooks

    subcommand = message_list[0].lower() if message_list else "list"

    # ── /hooks list ────────────────────────────────────────────────────────────
    if subcommand in ("list", "ls"):
        enabled_label = "[green]enabled[/green]" if hooks_cfg.enabled else "[red]disabled[/red]"
        console.print(f"\n  Hooks master switch: {enabled_label}\n")

        for lifecycle in ("PreToolUse", "PostToolUse", "SessionStart"):
            entries: list[HookEntry] = getattr(hooks_cfg, lifecycle)
            table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 2),
                title=f"[bold]{lifecycle}[/bold]",
            )
            table.add_column("Matcher", style="cyan", no_wrap=True)
            table.add_column("Command", style="yellow")
            if lifecycle == "PreToolUse":
                table.add_column("Blocking", justify="center")

            if not entries:
                table.add_row("[bright_black](none)[/bright_black]", "")
            else:
                for entry in entries:
                    if lifecycle == "PreToolUse":
                        # PreToolUse hooks are always blocking by nature (exit 2 blocks tool call)
                        table.add_row(entry.matcher, entry.command, "[green]yes[/green]")
                    else:
                        table.add_row(entry.matcher, entry.command)

            console.print(table)
            console.print()

        return {"continue": True}

    # ── /hooks enable / disable ────────────────────────────────────────────────
    if subcommand == "enable":
        hooks_cfg.enabled = True
        save_config(config, CONFIG_PATH)
        console.print("  [green]Hooks enabled.[/green]")
        return {"continue": True}

    if subcommand == "disable":
        hooks_cfg.enabled = False
        save_config(config, CONFIG_PATH)
        console.print("  [yellow]Hooks disabled.[/yellow]")
        return {"continue": True}

    # ── /hooks add <PreToolUse|PostToolUse> <matcher> <command> ───────────────
    if subcommand == "add":
        if len(message_list) < 4:
            console.print("  [bright_black]Usage: /hooks add <PreToolUse|PostToolUse|SessionStart> <matcher> <command>[/bright_black]")
            return {"continue": True}
        lifecycle = message_list[1]
        if lifecycle not in ("PreToolUse", "PostToolUse", "SessionStart"):
            console.print(f"  [red]Unknown lifecycle '{lifecycle}'. Use PreToolUse, PostToolUse, or SessionStart.[/red]")
            return {"continue": True}
        matcher = message_list[2]
        command = " ".join(message_list[3:])
        new_entry = HookEntry(matcher=matcher, command=command)
        getattr(hooks_cfg, lifecycle).append(new_entry)
        save_config(config, CONFIG_PATH)
        console.print(f"  [green]Added {lifecycle} hook:[/green] matcher=[cyan]{matcher}[/cyan] command=[yellow]{command}[/yellow]")
        return {"continue": True}

    # ── /hooks remove <PreToolUse|PostToolUse> <matcher> ──────────────────────
    if subcommand in ("remove", "rm"):
        if len(message_list) < 3:
            console.print("  [bright_black]Usage: /hooks remove <PreToolUse|PostToolUse|SessionStart> <matcher>[/bright_black]")
            return {"continue": True}
        lifecycle = message_list[1]
        if lifecycle not in ("PreToolUse", "PostToolUse", "SessionStart"):
            console.print(f"  [red]Unknown lifecycle '{lifecycle}'. Use PreToolUse, PostToolUse, or SessionStart.[/red]")
            return {"continue": True}
        matcher = message_list[2]
        entries: list[HookEntry] = getattr(hooks_cfg, lifecycle)
        before_count = len(entries)
        remaining = [e for e in entries if e.matcher != matcher]
        removed_count = before_count - len(remaining)
        setattr(hooks_cfg, lifecycle, remaining)
        save_config(config, CONFIG_PATH)
        if removed_count:
            console.print(f"  [green]Removed {removed_count} {lifecycle} hook(s) with matcher=[cyan]{matcher}[/cyan].[/green]")
        else:
            console.print(f"  [yellow]No {lifecycle} hooks found with matcher=[cyan]{matcher}[/cyan].[/yellow]")
        return {"continue": True}

    # ── /hooks test <tool_name> <json_input> ──────────────────────────────────
    if subcommand == "test":
        if len(message_list) < 3:
            console.print("  [bright_black]Usage: /hooks test <tool_name> <json_input>[/bright_black]")
            console.print('  [bright_black]Example: /hooks test run_bash \'{"command": "ls"}\'[/bright_black]')
            return {"continue": True}
        tool_name = message_list[1]
        json_input_str = " ".join(message_list[2:])
        try:
            tool_input = json.loads(json_input_str)
        except json.JSONDecodeError as exc:
            console.print(f"  [red]Invalid JSON input: {exc}[/red]")
            return {"continue": True}

        console.print(f"\n  [bold]Dry-run PreToolUse hooks for tool:[/bold] [cyan]{tool_name}[/cyan]")
        if not hooks_cfg.enabled:
            console.print("  [yellow]Hooks are disabled — no hooks would run.[/yellow]\n")
            return {"continue": True}

        from eaudev.modules.hooks import _find_matching_hooks
        matching = _find_matching_hooks(tool_name, hooks_cfg.PreToolUse)
        if not matching:
            console.print(f"  [bright_black]No PreToolUse hooks match tool '{tool_name}'.[/bright_black]\n")
            return {"continue": True}

        console.print(f"  [bright_black]{len(matching)} hook(s) would run:[/bright_black]")
        hook_result = run_pre_tool_hooks(tool_name, tool_input, hooks_cfg, session_id="__test__")
        if hook_result.allowed:
            console.print("  [green]Result: ALLOW[/green] — tool would proceed.")
        else:
            console.print("  [red]Result: BLOCK[/red] — tool would be blocked.")
            if hook_result.message:
                console.print(f"  [bright_black]Message: {hook_result.message}[/bright_black]")
        console.print()
        return {"continue": True}

    # ── Unknown subcommand ─────────────────────────────────────────────────────
    console.print(f"  [yellow]Unknown subcommand '{subcommand}'.[/yellow]")
    console.print("  [bright_black]Subcommands: list, add, remove, enable, disable, test[/bright_black]")
    return {"continue": True}


@registry.register("/feedback", None, "Report a bug or leave feedback.")
def _feedback_cmd(*args, **kwargs):
    console.print("""
[bold]Leave Feedback[/bold]
[bright_black]EauDev is a local-first tool. Please report issues or ideas via GitHub or your preferred channel.[/bright_black]

[bold]Log file location:[/bold]
  [cyan]~/.eaudev/eaudev.log[/cyan]
  (run /log to open it)

[bold]To report a bug:[/bold]
  Include the log file, your model name, and the prompt that caused the issue.
""")
    return {"continue": True}


# /help is handled directly by registry.dispatch — no registration needed


# ── Input helper ───────────────────────────────────────────────────────────────

def _read_user_input_simple() -> str:
    """Fallback plain input with multiline support (triple-quote mode)."""
    try:
        line = input("You: ").strip()
    except EOFError:
        raise KeyboardInterrupt
    if line == '"""':
        print('  (multiline mode — type """ on a line by itself to finish)')
        lines: list[str] = []
        while True:
            try:
                l = input()
            except EOFError:
                break
            if l.rstrip() == '"""':
                break
            lines.append(l)
        return "\n".join(lines)
    return line



# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> None:
    """Main entry point for EauDev."""
    import argparse

    parser = argparse.ArgumentParser(prog="eaudev", description=f"EauDev v{VERSION} — local AI coding agent")
    parser.add_argument("--restore", action="store_true", help="Resume the most recent saved session")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stderr")
    parser.add_argument("--voice", action="store_true", help="Start with VoiceIO mode active")
    parser.add_argument("--workdir", default=str(Path.cwd()), help="Working directory (default: auto-detected from cwd)")
    parser.add_argument("message", nargs="*", help="Initial prompt to send to the agent on startup")
    args = parser.parse_args()

    # ── Git root detection ──────────────────────────────────────────────────────
    explicit_workdir = args.workdir != str(Path.cwd())
    if not explicit_workdir:
        # Walk up from cwd to find the best workspace root.
        # Markers that indicate a workspace root, in priority order:
        #   1. .git directory (strongest signal)
        #   2. ARCHITECTURE.md (Cluster convention)
        #   3. README.md + multiple subdirs (generic project root)
        # Stop at home directory — never use home as workdir.
        home = Path.home()
        check = Path(args.workdir).resolve()
        best = check  # fallback: stay at cwd
        while check != check.parent and check != home:
            if (check / ".git").exists():
                best = check
                break  # .git is definitive — stop immediately
            if (check / "ARCHITECTURE.md").exists():
                best = check  # strong Cluster marker — keep walking up for .git
            elif (check / "README.md").exists() and best == Path(args.workdir).resolve():
                best = check  # weak signal — only use if nothing better found yet
            check = check.parent
        args.workdir = str(best)

    workdir = str(Path(args.workdir).resolve())

    # ── Initial message from CLI ────────────────────────────────────────────────
    initial_message: str | None = " ".join(args.message).strip() if args.message else None

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config(CONFIG_PATH)
    setup_logging(config.logging.path, verbose=args.verbose)
    persistence_dir = Path(config.sessions.persistence_dir).expanduser()
    persistence_dir.mkdir(parents=True, exist_ok=True)

    # ── MCP client — start servers and discover tools ───────────────────────────
    mcp_manager = get_mcp_manager()
    mcp_manager.start_all()

    # ── Memory store — EauDev's persistent brain-state ───────────────────────────
    mem_store = get_memory_store()

    # ── Banner ─────────────────────────────────────────────────────────────────
    console.print(BANNER_TEMPLATE.format(banner=BANNER, workdir=workdir))
    if Path(workdir).resolve() == Path.home().resolve():
        console.print(HOME_DIR_WARNING + "\n")

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory_instructions = get_memory_instructions(log_paths=True)

    # ── Session ────────────────────────────────────────────────────────────────
    thinking_directive = "" if config.agent.inference.enable_thinking else "/no_think"
    system_content = SYSTEM_PROMPT.format(workdir=workdir, home=str(Path.home()), thinking_directive=thinking_directive)
    if memory_instructions:
        system_content = system_content + "\n\n" + memory_instructions
    # Append user-configured additional system prompt (config.agent.additional_system_prompt)
    if config.agent.additional_system_prompt:
        system_content = system_content + "\n\n" + config.agent.additional_system_prompt.strip()
    # MCP tools are now passed via the `tools` API parameter in _run_agent,
    # not injected as text into the system prompt.

    workspace_path = Path(workdir)
    if args.restore:
        restored = get_most_recent_session(persistence_dir, workspace_path=workspace_path)
        if restored:
            session = restored
            # Ensure system prompt is current
            if session.message_history and session.message_history[0].get("role") == "system":
                session.message_history[0]["content"] = system_content
            else:
                session.message_history.insert(0, {"role": "system", "content": system_content})
            user_visible = sum(
                1 for m in session.message_history
                if m.get("role") in ("user", "assistant")
                and not str(m.get("content", "")).startswith("[")
            )
            console.print(
                f"[bright_black]Restoring session: {session.title} ({user_visible} exchanges)[/bright_black]\n"
            )
        else:
            console.print("[bright_black]No saved sessions found — starting new session.[/bright_black]\n")
            session = Session(workspace_path=workdir)
            session.message_history = [{"role": "system", "content": system_content}]
    else:
        session = Session(workspace_path=workdir)
        session.message_history = [{"role": "system", "content": system_content}]

    # Bind memory store to session and inject prior context
    mem_store.start(session.session_id, title=session.title)
    _mem_ctx = mem_store.load_context(max_turns=10)
    if _mem_ctx:
        if session.message_history and session.message_history[0].get("role") == "system":
            session.message_history[0]["content"] = session.message_history[0]["content"] + "\n\n" + _mem_ctx

    # ── Permissions ────────────────────────────────────────────────────────────
    perms = ToolPermissionManager(config, str(CONFIG_PATH))

    use_rich_ui = sys.stdin.isatty() and sys.stdout.isatty()

    # ── Prompt session (Rich UI — only when interactive) ───────────────────────
    prompt_session: PromptSession | None = None
    if use_rich_ui:
        history_path = Path.home() / ".eaudev" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = FilteredFileHistory(str(history_path))
        prompt_session = PromptSession(history=history)

    console.print("[bright_black]Type /help for commands. Ctrl+C to interrupt.[/bright_black]\n")

    # ── Show session state indicator ───────────────────────────────────────────
    # Display current session info at startup (inspired by Qwen Code workspace indicator)
    console.print(
        f"[dim]Session: {session.title[:50]}{'...' if len(session.title) > 50 else ''} | "
        f"Workspace: {workdir.split('/')[-1]}[/dim]\n"
    )

    # ── Auto-start inference server if not running ─────────────────────────────
    from eaudev.modules.model_registry import (
        load_model_registry, autostart_server, get_active_server_proc
    )
    _health_url = config.agent.inference.endpoint.replace("/v1/chat/completions", "/health")
    _server_alive = False
    try:
        with urllib.request.urlopen(_health_url, timeout=2) as _r:
            _server_alive = _r.status == 200
    except Exception:
        pass

    if not _server_alive:
        _models = load_model_registry()
        _default = next(
            (m for m in _models if "default" in m.tags),
            _models[0] if _models else None
        )
        if _default:
            _started = autostart_server(_default)
            if _started:
                config.agent.inference.model = _default.api_model_name
                config.agent.inference.endpoint = _default.endpoint
            else:
                console.print("[yellow]Could not start inference server. Messages will fail until a server is running.[/yellow]")
        else:
            console.print("[yellow]No models.yml found. Start a server manually on port 8080.[/yellow]")
    else:
        # Server was pre-existing — sync model name by matching registry against loaded models
        try:
            _v1models_url = config.agent.inference.endpoint.replace("/v1/chat/completions", "/v1/models")
            with urllib.request.urlopen(_v1models_url, timeout=3) as _r:
                _v1data = json.loads(_r.read())
                _loaded_ids = {m["id"] for m in _v1data.get("data", [])}
            _registry = load_model_registry()
            _matched = next(
                (m.api_model_name for m in _registry if m.api_model_name in _loaded_ids),
                None
            )
            if _matched:
                config.agent.inference.model = _matched
        except Exception:
            pass

    # ── Register server shutdown on exit ───────────────────────────────────────
    def _shutdown_server() -> None:
        proc = get_active_server_proc()
        if proc is None:
            return  # server was pre-existing — leave it running
        try:
            proc.terminate()
            proc.wait(timeout=5)
            console.print("[bright_black]Inference server stopped.[/bright_black]")
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    atexit.register(_shutdown_server)
    atexit.register(mcp_manager.stop_all)
    # flush() uses the session title set via mem_store.set_title() during the loop
    atexit.register(mem_store.flush)

    # Query llama.cpp for actual context window size
    context_limit: int = _get_context_limit(config.agent.inference.endpoint)

    last_tokens: int = 0  # token count from most recent agent response — never goes back to 0 once set
    _pending_message: str | None = initial_message  # from --message CLI arg

    # ── VoiceIO — activate if --voice flag passed ───────────────────────────────
    voice = get_voice_io()
    if args.voice:
        try:
            voice.start()
            console.print("[green]VoiceIO active[/green] — speak your prompt, then pause.")
        except Exception as exc:
            console.print(f"[red]VoiceIO failed to start:[/red] {exc}")
    atexit.register(lambda: voice.stop() if voice.active else None)

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            # Use pending message (from --message flag) on first iteration
            if _pending_message is not None:
                user_input = _pending_message
                _pending_message = None
                console.print(f"[bright_black]> {user_input}[/bright_black]")
            elif voice.active:
                # VoiceIO mode — listen for spoken input
                # Show a listening indicator so the user knows EauDev is listening
                console.print(Panel(
                    "[bold]Listening...[/bold]  [dim]Speak now. Press [bold]Ctrl+C[/bold] to exit voice mode.[/dim]",
                    border_style="bright_black",
                    padding=(0, 1),
                ))
                try:
                    spoken = voice.listen(timeout=30.0)
                except KeyboardInterrupt:
                    voice.stop()
                    console.print("[dim]VoiceIO deactivated.[/dim]")
                    continue
                if spoken is None:
                    # Timeout — show listening panel again on next iteration
                    console.print("[dim]No speech detected — listening again.[/dim]")
                    continue
                user_input = spoken
                console.print(f"[bright_black]> {user_input}[/bright_black]")
            elif use_rich_ui and prompt_session is not None:
                try:
                    # Show token bar after first message has been sent
                    # Token bar appears from second prompt onward (after first response)
                    show_tokens = session.num_messages >= 1
                    user_input = asyncio.run(
                        prompt_session.prompt_async(
                            session_context=(last_tokens, context_limit) if show_tokens else None
                        )
                    )
                except (SystemExit, EOFError):
                    raise KeyboardInterrupt
            else:
                user_input = _read_user_input_simple()

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle exit keywords
            if user_input in DEFAULT_EXIT_COMMANDS:
                raise KeyboardInterrupt

            # ── Slash commands ─────────────────────────────────────────────────
            if user_input.startswith("/") or user_input.startswith("#"):
                dispatched = user_input in registry.commands or any(
                    user_input.startswith(cmd) for cmd in registry.commands
                )
                result = registry.dispatch(
                    user_input,
                    _session=session,
                    _persistence_dir=persistence_dir,
                    _config=config,
                )

                # None means either: command was handled (e.g. /help prints and returns None)
                # or command was not found. Distinguish by checking registry membership.
                if result is None:
                    if not dispatched:
                        # Suggest similar commands (inspired by Qwen Code's command suggestions)
                        all_cmds = registry.commands
                        similar = difflib.get_close_matches(user_input, all_cmds, n=2, cutoff=0.6)
                        if similar:
                            console.print(
                                f"[yellow]Unknown command: {user_input}[/yellow]\n"
                                f"  [dim]Did you mean: {' or '.join(similar)}?[/dim]"
                            )
                        else:
                            console.print(
                                f"[bright_black]Unknown command: {user_input}. Type /help for available commands.[/bright_black]"
                            )
                    continue

                if isinstance(result, dict):
                    # /sessions — start a new session
                    if result.get("new_session"):
                        session.set_title_from_first_message()
                        session.save(persistence_dir)
                        session = Session(workspace_path=workdir)
                        session.message_history = [{"role": "system", "content": system_content}]
                        console.print("  [bright_black]New session started.[/bright_black]")
                        continue

                    # /sessions — switch to an existing session
                    if result.get("switch_session"):
                        session.set_title_from_first_message()
                        session.save(persistence_dir)
                        session = result["switch_session"]
                        # Refresh system prompt
                        if session.message_history and session.message_history[0].get("role") == "system":
                            session.message_history[0]["content"] = system_content
                        console.print(
                            f"  [bright_black]Switched to: {session.title} ({session.num_messages} messages)[/bright_black]"
                        )
                        continue

                    # /clear — wipe message history
                    if "message_history" in result and result["message_history"] is None:
                        session.message_history = [{"role": "system", "content": system_content}]
                        last_tokens = 0
                        console.print("  [bright_black]Context cleared.[/bright_black]")
                        continue

                    # /prune — re-query /slots for accurate token count post-prune
                    if result.get("refresh_tokens"):
                        refreshed = _get_n_past(config.agent.inference.endpoint)
                        if refreshed > 0:
                            last_tokens = refreshed
                        continue

                    # /compact — update token estimate to post-compact value
                    if "reset_tokens" in result:
                        last_tokens = result["reset_tokens"]
                        continue

                    # /model — update context limit for token bar
                    if "new_context_limit" in result:
                        context_limit = result["new_context_limit"]
                        continue

                    # Command returned a queued message for the agent
                    queued = result.get("message")
                    if queued:
                        user_input = queued
                        # Fall through to agent below
                    else:
                        continue

                elif isinstance(result, str):
                    user_input = result
                    # Fall through to agent below
                else:
                    continue

            # ── Agent turn ─────────────────────────────────────────────────────
            session.message_history.append({"role": "user", "content": user_input})
            mem_store.record_turn("user", user_input)
            if not session.initial_prompt:
                session.initial_prompt = user_input[:200]

            new_tokens = _run_agent(session, perms, config, workdir, context_limit=context_limit)
            if new_tokens > 0:
                last_tokens = new_tokens  # only update if we got a real reading
            # Speak the last assistant response if VoiceIO is active
            if voice.active and session.message_history:
                last_msg = session.message_history[-1]
                if last_msg.get("role") == "assistant":
                    # Strip tool call blocks before speaking
                    import re as _re
                    speakable = _re.sub(r"<tool>[\s\S]*?</tool>", "", last_msg["content"])
                    speakable = _re.sub(r"<think>[\s\S]*?</think>", "", speakable).strip()
                    if speakable:
                        voice.speak(speakable)

            # Auto-save
            if not session.title or session.title in ('New Session', 'Untitled Session'):
                session.title = _generate_session_title(session.message_history, config)
                mem_store.set_title(session.title)
            session.save(persistence_dir)

        except EauDevError as e:
            style = "yellow" if e.role == "warning" else "red"
            console.print(Panel(e.message, title=f"[bold]{e.title}[/bold]", border_style=style))
        except KeyboardInterrupt:
            # If VoiceIO is active, Ctrl+C deactivates voice mode first
            # Second Ctrl+C exits EauDev
            if voice.active:
                voice.stop()
                console.print("\n[dim]VoiceIO deactivated. Press Ctrl+C again to exit.[/dim]")
                continue
            console.print("\n")
            session.set_title_from_first_message()
            session.save(persistence_dir)
            console.print("  [bright_black]Session saved. Goodbye.[/bright_black]")
            # LoRA pipeline will run via atexit handler (mem_store.flush)
            sys.exit(0)
        except Exception as e:
            logger.exception("Unhandled error in main loop")
            console.print(f"\n[red][error: {e}][/red]")
