"""
EauDev — Narrow MCP Server Registry
=====================================
Loads, validates, and manages the narrow MCP server registry.

Registry location: ~/.eaudev/narrow_server_registry.yaml
Model cards:       ~/.eaudev/model_cards/<server_id>.yaml

The registry is EauDev's phonebook for all Cluster MCP servers.
Every server — Archive, Analyst, Memory, Extractor, etc. — has a model card
that is registered here. EauDev reads this at session start to:

  1. Know what servers exist and what they do
  2. Check resource headroom before spawning
  3. Match user intent to the right server
  4. Export registry state to LoRA consolidation artefact at session end
     (so EauDev's weights carry intrinsic knowledge of available servers)

The registry is also the target of the server registration tool — when a new
narrow server is built, it is registered here and the model card is written.
EauDev's next session LoRA will incorporate the new server's capabilities.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import yaml

EAUDEV_DIR = Path("~/.eaudev").expanduser()
REGISTRY_PATH = EAUDEV_DIR / "narrow_server_registry.yaml"
MODEL_CARDS_DIR = EAUDEV_DIR / "model_cards"

VALID_STATES = {
    "DORMANT", "INITIALISING", "NOMINAL",
    "AWAITING_INPUT", "CONSULTATION", "DEGRADED", "CRITICAL_FAILURE"
}
VALID_TIERS = {"light", "standard", "heavy", "exclusive"}
VALID_LIFECYCLE_MODES = {"one_shot", "persistent", "agentic_loop", "repl"}


# ── Registry file structure ────────────────────────────────────────────────────

def _default_registry() -> dict:
    return {
        "version": "1.0",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "servers": {}
    }


def _load_registry() -> dict:
    """Load the registry file. Creates it if it doesn't exist."""
    EAUDEV_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        reg = _default_registry()
        REGISTRY_PATH.write_text(yaml.dump(reg, default_flow_style=False, allow_unicode=True))
        return reg
    return yaml.safe_load(REGISTRY_PATH.read_text()) or _default_registry()


def _save_registry(registry: dict) -> None:
    """Persist the registry file."""
    registry["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    EAUDEV_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        yaml.dump(registry, default_flow_style=False, allow_unicode=True)
    )


# ── Model card loading ─────────────────────────────────────────────────────────

def load_model_card(server_id: str) -> dict | None:
    """Load a model card by server_id. Returns None if not found."""
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    card_path = MODEL_CARDS_DIR / f"{server_id}.yaml"
    if not card_path.exists():
        return None
    return yaml.safe_load(card_path.read_text()) or {}


def save_model_card(card: dict) -> Path:
    """Save a model card. Returns the path written."""
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    server_id = card.get("server_id", "")
    if not server_id:
        raise ValueError("Model card must have a server_id")
    card_path = MODEL_CARDS_DIR / f"{server_id}.yaml"
    card_path.write_text(
        yaml.dump(card, default_flow_style=False, allow_unicode=True)
    )
    return card_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_model_card(card: dict) -> list[str]:
    """
    Validate a model card against the Narrow MCP Server Standard Schema.
    Returns a list of validation errors. Empty list = valid.
    """
    errors = []

    # Required identity fields
    for field in ["server_id", "server_name", "server_version", "description"]:
        if not card.get(field):
            errors.append(f"Missing required field: {field}")

    # base_model required only when lora_merged is True
    if card.get("lora_merged") and not card.get("base_model"):
        errors.append("base_model is required when lora_merged is True")

    # Lifecycle mode
    lm = card.get("lifecycle_mode", "")
    if lm and lm not in VALID_LIFECYCLE_MODES:
        errors.append(f"Invalid lifecycle_mode '{lm}'. Must be one of: {VALID_LIFECYCLE_MODES}")

    # Resource profile
    rp = card.get("resource_profile", {})
    tier = rp.get("tier", "")
    if tier and tier not in VALID_TIERS:
        errors.append(f"Invalid resource tier '{tier}'. Must be one of: {VALID_TIERS}")

    # Operational state
    ops = card.get("operational_state", {})
    state = ops.get("current_state", "")
    if state and state not in VALID_STATES:
        errors.append(f"Invalid operational_state '{state}'. Must be one of: {VALID_STATES}")

    # Paths — server_entry must exist if declared
    paths = card.get("paths", {})
    entry = paths.get("server_entry", "")
    if entry and not Path(entry).expanduser().exists():
        errors.append(f"paths.server_entry does not exist: {entry}")

    return errors


# ── Registration ──────────────────────────────────────────────────────────────

def register_server(card: dict, overwrite: bool = False) -> dict:
    """
    Register a narrow MCP server from its model card.

    Args:
        card:      Complete model card dict (validated before writing)
        overwrite: If True, overwrite existing registration

    Returns:
        {"success": bool, "server_id": str, "errors": [...], "card_path": str}
    """
    errors = validate_model_card(card)
    if errors:
        return {"success": False, "server_id": card.get("server_id", ""), "errors": errors}

    server_id = card["server_id"]
    registry = _load_registry()

    if server_id in registry.get("servers", {}) and not overwrite:
        return {
            "success": False,
            "server_id": server_id,
            "errors": [f"Server '{server_id}' already registered. Use overwrite=True to update."]
        }

    # Write model card
    card_path = save_model_card(card)

    # Registry entry — lightweight summary pointing to full model card
    registry.setdefault("servers", {})[server_id] = {
        "server_name":    card.get("server_name", ""),
        "server_version": card.get("server_version", "1.0.0"),
        "description":    card.get("description", "")[:200],  # truncated for registry
        "capability_tags": card.get("capability_tags", []),
        "lifecycle_mode": card.get("lifecycle_mode", "one_shot"),
        "tier":           card.get("resource_profile", {}).get("tier", "standard"),
        "model_size_gb":  card.get("resource_profile", {}).get("model_size_gb", 0.0),
        "card_path":      str(card_path),
        "registered_at":  datetime.datetime.utcnow().isoformat() + "Z",
        "current_state":  "DORMANT",
    }

    _save_registry(registry)

    return {
        "success":   True,
        "server_id": server_id,
        "errors":    [],
        "card_path": str(card_path),
    }


def deregister_server(server_id: str) -> dict:
    """Remove a server from the registry (does not delete the model card)."""
    registry = _load_registry()
    if server_id not in registry.get("servers", {}):
        return {"success": False, "error": f"Server '{server_id}' not found in registry"}
    del registry["servers"][server_id]
    _save_registry(registry)
    return {"success": True, "server_id": server_id}


# ── State management ──────────────────────────────────────────────────────────

def update_server_state(server_id: str, state: str, active_task: str = "") -> None:
    """Update a server's operational state in the registry."""
    if state not in VALID_STATES:
        return
    registry = _load_registry()
    if server_id in registry.get("servers", {}):
        registry["servers"][server_id]["current_state"] = state
        registry["servers"][server_id]["active_task"] = active_task
        registry["servers"][server_id]["last_heartbeat"] = datetime.datetime.utcnow().isoformat() + "Z"
        _save_registry(registry)


# ── Session start ─────────────────────────────────────────────────────────────

def get_session_context(available_memory_gb: float = 64.0) -> dict:
    """
    Load registry at session start.
    Returns structured context about available servers for EauDev injection.

    Args:
        available_memory_gb: Current available memory. Used to flag servers
                             that cannot be spawned given current headroom.
    """
    registry = _load_registry()
    servers = registry.get("servers", {})

    available = []
    unavailable = []

    for server_id, entry in servers.items():
        model_size = entry.get("model_size_gb", 0.0)
        tier = entry.get("tier", "standard")
        can_spawn = model_size <= available_memory_gb or tier == "light"

        record = {
            "server_id":      server_id,
            "server_name":    entry.get("server_name", ""),
            "description":    entry.get("description", ""),
            "capability_tags": entry.get("capability_tags", []),
            "lifecycle_mode": entry.get("lifecycle_mode", "one_shot"),
            "tier":           tier,
            "model_size_gb":  model_size,
            "current_state":  entry.get("current_state", "DORMANT"),
        }

        if can_spawn:
            available.append(record)
        else:
            unavailable.append(record)

    return {
        "total_servers":        len(servers),
        "available_servers":    available,
        "unavailable_servers":  unavailable,
        "registry_path":        str(REGISTRY_PATH),
    }


# ── LoRA consolidation export ─────────────────────────────────────────────────

def export_registry_to_jsonl(output_path: Path) -> int:
    """
    Export registry state as alpaca JSONL training examples.
    Appended to the session consolidation artefact that feeds session_to_lora.py.

    EauDev's LoRA will incorporate knowledge of available servers —
    what they do, when to spawn them, what they produce.

    Returns: number of examples written.
    """
    registry = _load_registry()
    servers = registry.get("servers", {})
    if not servers:
        return 0

    examples = []

    # Example 1 — What servers exist?
    server_list = "\n".join(
        f"- {e.get('server_name', sid)} ({sid}): {e.get('description', '')[:120]}"
        for sid, e in servers.items()
    )
    examples.append({
        "instruction": "What MCP servers are registered in Cluster?",
        "input": "",
        "output": f"The following MCP servers are registered:\n{server_list}"
    })

    # Example 2 — Per-server capability examples
    for server_id, entry in servers.items():
        name = entry.get("server_name", server_id)
        desc = entry.get("description", "")
        tags = ", ".join(entry.get("capability_tags", []))
        lifecycle = entry.get("lifecycle_mode", "one_shot")
        tier = entry.get("tier", "standard")

        # Load full model card for richer detail
        card = load_model_card(server_id)
        spawn_trigger = card.get("spawn_trigger", "") if card else ""
        tool_sequence = card.get("tool_sequence", []) if card else []
        tools_str = ""
        if tool_sequence:
            tools_str = "\nTool sequence: " + " → ".join(str(t) for t in tool_sequence)

        examples.append({
            "instruction": f"When should EauDev spawn the {name} server?",
            "input": "",
            "output": spawn_trigger or f"Spawn {name} when {desc}"
        })

        examples.append({
            "instruction": f"What does the {name} server do?",
            "input": "",
            "output": (
                f"{name} ({server_id}): {desc}\n"
                f"Capabilities: {tags}\n"
                f"Lifecycle: {lifecycle} | Tier: {tier}"
                f"{tools_str}"
            )
        })

    # Example 3 — Intent matching summary
    intent_lines = []
    for server_id, entry in servers.items():
        name = entry.get("server_name", server_id)
        tags = entry.get("capability_tags", [])
        card = load_model_card(server_id)
        trigger = card.get("spawn_trigger", "") if card else ""
        if trigger:
            intent_lines.append(f"- {name}: {trigger[:100]}")
        elif tags:
            intent_lines.append(f"- {name}: handles {', '.join(tags[:3])}")

    if intent_lines:
        examples.append({
            "instruction": "How do I know which Cluster MCP server to use for a given task?",
            "input": "",
            "output": "Match the task intent to the server's spawn trigger:\n" + "\n".join(intent_lines)
        })

    # Write JSONL — append to existing file if present
    with open(output_path, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)


# ── Summary for display ───────────────────────────────────────────────────────

def get_registry_summary() -> str:
    """Return a human-readable registry summary for EauDev session start display."""
    ctx = get_session_context()
    total = ctx["total_servers"]
    available = len(ctx["available_servers"])

    if total == 0:
        return "[registry] No servers registered."

    lines = [f"[registry] {total} server(s) registered, {available} available:"]
    for s in ctx["available_servers"]:
        state = s["current_state"]
        state_icon = "●" if state == "NOMINAL" else "○"
        lines.append(
            f"  {state_icon} {s['server_name']} ({s['server_id']}) "
            f"[{s['tier']}] — {s['description'][:60]}..."
        )
    for s in ctx["unavailable_servers"]:
        lines.append(f"  ✗ {s['server_name']} ({s['server_id']}) — insufficient memory ({s['model_size_gb']}GB)")

    return "\n".join(lines)
