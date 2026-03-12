#!/usr/bin/env python3
"""
EauDev SessionStart hook — Session initialisation and context injection.

Runs once when an EauDev session starts. Checks whether hooks are enabled in
~/.eaudev/config.yml and optionally injects a note about any .agent.md found
in the working directory. Also injects Narrow MCP Server registry summary so
EauDev knows which servers are available at session start.

Protocol:
  stdin  → JSON: {event, session_id, context}
             context may contain: {workdir: str, ...}
  stdout → JSON: {"additionalContext": "..."}
  exit 0 always (never blocks the session)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH   = Path.home() / ".eaudev" / "config.yml"
AGENT_MD_PEEK = 200   # characters to read from .agent.md

# Registry — graceful fallback if module unavailable
_REGISTRY_AVAILABLE = False
try:
    _eaudev_root = Path(__file__).resolve().parents[2]
    if str(_eaudev_root) not in sys.path:
        sys.path.insert(0, str(_eaudev_root))
    from eaudev.modules.server_registry import get_registry_summary
    _REGISTRY_AVAILABLE = True
except Exception:
    pass

# LoRA lifecycle — graceful fallback if memory package unavailable
_LORA_AVAILABLE = False
try:
    from eaudev.memory.lora_lifecycle import get_current_adapter_path
    _LORA_AVAILABLE = True
except Exception:
    pass


def _hooks_enabled_in_config() -> bool:
    """Read hooks.enabled from ~/.eaudev/config.yml. Defaults True on any error."""
    if not CONFIG_PATH.exists():
        return True
    try:
        import yaml
        raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        return raw.get("hooks", {}).get("enabled", True)
    except Exception:
        return True  # fail-open


def main() -> None:
    # ── 1. Read stdin ─────────────────────────────────────────────────────────
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        # Always exit 0; never block session startup
        print(json.dumps({"additionalContext": "Session started. Hooks active."}))
        sys.exit(0)

    context = payload.get("context") or {}
    workdir = context.get("workdir", "")

    # ── 2. Check hooks.enabled ────────────────────────────────────────────────
    hooks_on = _hooks_enabled_in_config()
    hooks_label = "Hooks active." if hooks_on else "Hooks disabled."

    # ── 3. Look for .agent.md in the working directory ────────────────────────
    agent_md_note = ""
    if workdir:
        agent_md_path = Path(workdir) / ".agent.md"
        if agent_md_path.is_file():
            try:
                preview = agent_md_path.read_text(errors="replace")[:AGENT_MD_PEEK]
                # Collapse newlines for a compact inline note
                preview_inline = " ".join(preview.split())
                agent_md_note = f" Agent instructions found (.agent.md): \"{preview_inline}\""
            except Exception:
                agent_md_note = " Agent instructions found (.agent.md) but could not be read."

    # ── 4. Narrow MCP Server registry summary ────────────────────────────────
    registry_note = ""
    if _REGISTRY_AVAILABLE:
        try:
            registry_note = "\n" + get_registry_summary()
        except Exception:
            registry_note = "\n[registry] Registry unavailable at session start."

    # ── 5. LoRA adapter state ─────────────────────────────────────────────────
    adapter_note = ""
    if _LORA_AVAILABLE:
        try:
            adapter_path = get_current_adapter_path()
            if adapter_path:
                adapter_file = Path(adapter_path).expanduser()
                if adapter_file.exists():
                    adapter_note = f"\n[lora] Active session adapter: {adapter_path}"
                else:
                    adapter_note = (
                        f"\n[lora] WARNING: Adapter path set ({adapter_path}) "
                        f"but file does not exist. Run session_to_lora.py with --model "
                        f"to train and fuse the adapter."
                    )
        except Exception:
            pass

    # ── 6. Emit additionalContext ─────────────────────────────────────────────
    additional = f"Session started. {hooks_label}{agent_md_note}{registry_note}{adapter_note}"
    print(json.dumps({"additionalContext": additional}))
    sys.exit(0)


if __name__ == "__main__":
    main()
