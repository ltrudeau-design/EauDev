"""
EauDev — Register Narrow MCP Server Tool
=========================================
Called when a new narrow MCP server is built and needs to be registered
in the Cluster registry. Validates the model card, writes it to
~/.eaudev/model_cards/, and updates ~/.eaudev/narrow_server_registry.yaml.

The next session's LoRA consolidation will incorporate the new server's
capabilities into EauDev's weights — so EauDev boots with intrinsic
knowledge of the new server.

Usage (from EauDev tool call):
    register_narrow_server(card_path="/path/to/model_card.yaml")
    register_narrow_server(card={"server_id": "...", ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eaudev.modules.server_registry import (
    register_server,
    deregister_server,
    load_model_card,
    get_registry_summary,
    validate_model_card,
    REGISTRY_PATH,
    MODEL_CARDS_DIR,
)


def register_narrow_server(
    card_path: str | None = None,
    card: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Register a narrow MCP server in the Cluster registry.

    Provide either a path to a model card YAML file, or the card dict directly.

    Args:
        card_path: Path to a model card YAML file
        card:      Model card as a dict (alternative to card_path)
        overwrite: If True, overwrite existing registration

    Returns:
        {
            "success": bool,
            "server_id": str,
            "errors": [...],
            "card_path": str,
            "registry_path": str,
            "summary": str,
            "lora_note": str
        }
    """
    if card_path:
        p = Path(card_path).expanduser()
        if not p.exists():
            return {
                "success": False,
                "error": f"Model card file not found: {card_path}",
            }
        try:
            card = yaml.safe_load(p.read_text())
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to parse model card YAML: {exc}",
            }

    if not card:
        return {
            "success": False,
            "error": "Provide either card_path or card dict",
        }

    result = register_server(card, overwrite=overwrite)

    if result["success"]:
        result["registry_path"] = str(REGISTRY_PATH)
        result["summary"] = get_registry_summary()
        result["lora_note"] = (
            f"Server '{card.get('server_id')}' registered. "
            "EauDev's next session LoRA will incorporate this server's capabilities. "
            "Run session_to_lora.py at end of session to update weights."
        )

    return result


def list_registered_servers() -> dict:
    """Return all registered servers with their current state."""
    from eaudev.modules.server_registry import _load_registry
    registry = _load_registry()
    servers = registry.get("servers", {})
    return {
        "success": True,
        "total": len(servers),
        "servers": [
            {
                "server_id":   sid,
                "server_name": entry.get("server_name", ""),
                "tier":        entry.get("tier", ""),
                "state":       entry.get("current_state", "DORMANT"),
                "description": entry.get("description", "")[:100],
            }
            for sid, entry in servers.items()
        ],
        "registry_path": str(REGISTRY_PATH),
    }


def validate_server_card(card_path: str) -> dict:
    """Validate a model card without registering it."""
    p = Path(card_path).expanduser()
    if not p.exists():
        return {"success": False, "error": f"File not found: {card_path}"}
    try:
        card = yaml.safe_load(p.read_text())
    except Exception as exc:
        return {"success": False, "error": f"YAML parse error: {exc}"}

    errors = validate_model_card(card)
    return {
        "success": len(errors) == 0,
        "server_id": card.get("server_id", ""),
        "errors": errors,
        "valid": len(errors) == 0,
    }


def remove_server(server_id: str) -> dict:
    """Remove a server from the registry (model card is preserved)."""
    return deregister_server(server_id)
