"""
Generate Hook MCP Server

Provides tools for generating, managing, and testing EauDev lifecycle hooks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generate_hook")

# ── Constants ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("~/.eaudev/config.yml").expanduser()
LLM_ENDPOINT = "http://localhost:8080/v1/chat/completions"

HOOK_SYSTEM_PROMPT = """\
You generate EauDev hook scripts. Hooks are Python scripts that:
1. Read JSON from stdin: {tool_name, tool_input, session_id}
2. Perform their logic
3. Exit with: 0=allow, 1=warn (print warning to stderr), 2=block (print reason to stderr)

Generate ONLY the Python script content, no explanation.\
"""

VERDICT_MAP = {0: "allow", 1: "warn", 2: "block"}


# ── Config helpers ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load ~/.eaudev/config.yml, returning an empty dict if it doesn't exist."""
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def _save_config(config: dict) -> None:
    """Write config dict back to ~/.eaudev/config.yml."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _ensure_hooks_structure(config: dict) -> dict:
    """Ensure hooks section with PreToolUse/PostToolUse lists exists."""
    config.setdefault("hooks", {})
    config["hooks"].setdefault("enabled", True)
    config["hooks"].setdefault("PreToolUse", [])
    config["hooks"].setdefault("PostToolUse", [])
    return config


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def generate_hook(
    description: str,
    lifecycle: str,
    matcher: str,
    output_path: str,
) -> dict:
    """
    Generate a Python hook script using a local LLM based on a natural language description.

    Args:
        description: Natural language description of what the hook should do,
                     e.g. 'block any bash command containing rm -rf'.
        lifecycle:   'PreToolUse' or 'PostToolUse'.
        matcher:     Tool name to match or '*' for all tools.
        output_path: Where to write the hook script, e.g. '~/.eaudev/hooks/bash_guard.py'.

    Returns:
        dict with keys: script (str), output_path (str), config_entry (dict).
    """
    if lifecycle not in ("PreToolUse", "PostToolUse"):
        raise ValueError("lifecycle must be 'PreToolUse' or 'PostToolUse'")

    # Build the user prompt with context about lifecycle and matcher
    post_note = (
        "\nNote: this is a PostToolUse hook — stdin JSON also includes 'tool_result'."
        if lifecycle == "PostToolUse"
        else ""
    )
    matcher_note = (
        f"\nThe hook will be triggered for tool: {matcher!r}."
        if matcher != "*"
        else "\nThe hook will be triggered for ALL tools (matcher='*')."
    )

    user_prompt = (
        f"Generate a Python hook script that does the following:\n{description}"
        f"{matcher_note}{post_note}"
    )

    # Call local LLM
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": HOOK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    try:
        import urllib.request
        req_data = json.dumps(payload).encode()
        req = urllib.request.Request(
            LLM_ENDPOINT,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_json = json.loads(resp.read().decode())
        script = response_json["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        # Return a placeholder script if the LLM is unavailable
        script = (
            "#!/usr/bin/env python3\n"
            "# Hook generated offline — LLM was unavailable.\n"
            "# TODO: implement hook logic for: " + description + "\n\n"
            "import json\nimport sys\n\n"
            "data = json.load(sys.stdin)\n"
            "# Your logic here\n"
            "sys.exit(0)  # 0=allow, 1=warn, 2=block\n"
        )
        script += f"\n# LLM error: {exc}\n"

    # Write script to output_path
    resolved_path = Path(output_path).expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(script)
    resolved_path.chmod(0o755)

    config_entry = {
        "matcher": matcher,
        "command": f"python3 {resolved_path}",
    }

    return {
        "script": script,
        "output_path": str(resolved_path),
        "config_entry": config_entry,
    }


@mcp.tool()
def list_hooks() -> dict:
    """
    Read ~/.eaudev/config.yml and return all registered hooks.

    Returns:
        dict with keys: PreToolUse (list), PostToolUse (list), enabled (bool).
    """
    config = _load_config()
    hooks_cfg = config.get("hooks", {})
    return {
        "PreToolUse": hooks_cfg.get("PreToolUse", []),
        "PostToolUse": hooks_cfg.get("PostToolUse", []),
        "enabled": hooks_cfg.get("enabled", True),
    }


@mcp.tool()
def register_hook(lifecycle: str, matcher: str, command: str) -> dict:
    """
    Add a hook entry to ~/.eaudev/config.yml.

    Args:
        lifecycle: 'PreToolUse' or 'PostToolUse'.
        matcher:   Tool name or '*' for all tools.
        command:   Shell command to run as the hook.

    Returns:
        dict with keys: success (bool), message (str).
    """
    if lifecycle not in ("PreToolUse", "PostToolUse"):
        return {
            "success": False,
            "message": f"Invalid lifecycle {lifecycle!r}. Must be 'PreToolUse' or 'PostToolUse'.",
        }

    config = _load_config()
    config = _ensure_hooks_structure(config)

    new_entry = {"matcher": matcher, "command": command}
    config["hooks"][lifecycle].append(new_entry)
    _save_config(config)

    return {
        "success": True,
        "message": (
            f"Registered {lifecycle} hook: matcher={matcher!r}, command={command!r}"
        ),
    }


@mcp.tool()
def remove_hook(lifecycle: str, matcher: str) -> dict:
    """
    Remove all hook entries matching the given lifecycle and matcher from ~/.eaudev/config.yml.

    Args:
        lifecycle: 'PreToolUse' or 'PostToolUse'.
        matcher:   Tool name or '*' to match all-tools hooks.

    Returns:
        dict with keys: removed (int), message (str).
    """
    if lifecycle not in ("PreToolUse", "PostToolUse"):
        return {
            "removed": 0,
            "message": f"Invalid lifecycle {lifecycle!r}. Must be 'PreToolUse' or 'PostToolUse'.",
        }

    config = _load_config()
    config = _ensure_hooks_structure(config)

    hooks_list: list[dict] = config["hooks"][lifecycle]
    before_count = len(hooks_list)
    config["hooks"][lifecycle] = [
        h for h in hooks_list if h.get("matcher") != matcher
    ]
    removed_count = before_count - len(config["hooks"][lifecycle])
    _save_config(config)

    if removed_count == 0:
        message = f"No {lifecycle} hooks found with matcher={matcher!r}."
    else:
        message = (
            f"Removed {removed_count} {lifecycle} hook(s) with matcher={matcher!r}."
        )

    return {"removed": removed_count, "message": message}


@mcp.tool()
def test_hook(command: str, tool_name: str, tool_input: dict) -> dict:
    """
    Run a hook script against a sample input and report the verdict.

    Args:
        command:    Shell command to run (e.g. 'python3 ~/.eaudev/hooks/bash_guard.py').
        tool_name:  Tool name to include in the stdin JSON payload.
        tool_input: Tool input dict to include in the stdin JSON payload.

    Returns:
        dict with keys: exit_code (int), stdout (str), stderr (str), verdict (str).
        verdict is one of 'allow', 'warn', 'block'.
    """
    payload = json.dumps({
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": "test-session",
    })

    try:
        proc = subprocess.run(
            command,
            shell=True,
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ},
        )
        exit_code = proc.returncode
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Hook timed out after 10 seconds.",
            "verdict": "allow",
        }
    except Exception as exc:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "verdict": "allow",
        }

    verdict = VERDICT_MAP.get(exit_code, f"unknown (exit {exit_code})")
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "verdict": verdict,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
