"""
lora_lifecycle.py — LoRA adapter state tracking via the facts layer.

State is persisted under category="lora_lifecycle" in PersistentFacts.

Fields tracked:
    current_adapter  — path to current adapter .safetensors
    session_count    — sessions accumulated in the active adapter
    merge_generation — how many merge operations have been performed
    base_model_hash  — sha256 of the base model (set externally after training)
    adapter_started  — ISO timestamp when current adapter was started

LoRA stacking ceiling: 2 adapters maximum (enforced via session_count threshold).
    Sessions  1-20:  Base + LoRA_A  (building)
    Sessions 21-40:  Base + LoRA_A (frozen) + LoRA_B (building)
    Session 41:      MERGE required → new base checkpoint, reset counter

Merge trigger: session_count % 40 == 0 (and session_count > 0)
Warning threshold: session_count % 20 == 0 (adapter slot boundary)

All operations are deterministic. No inference.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_EAUDEV_DIR = Path.home() / '.eaudev'
_LORA_LIFECYCLE_CATEGORY = 'lora_lifecycle'
_SESSIONS_PER_ADAPTER = 20     # sessions before the active slot changes
_ADAPTER_CEILING = 2           # maximum stacked adapters


def _get_facts(facts_db_path: Optional[str] = None):
    from eaudev.memory.layers.facts import PersistentFacts
    path = facts_db_path or str(_EAUDEV_DIR / 'facts.db')
    return PersistentFacts(db_path=path)


def get_lora_status(facts_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return current LoRA lifecycle state.

    Reads persisted values from facts layer under category="lora_lifecycle".
    Returns defaults for any fields not yet written.

    Returns:
        dict with: current_adapter, session_count, merge_generation,
                   base_model_hash, adapter_started, active_slot,
                   merge_required, adapter_ceiling_warning
    """
    facts = _get_facts(facts_db_path)

    def _get(key: str, default: Any) -> Any:
        val = facts.get_fact(_LORA_LIFECYCLE_CATEGORY, key)
        return val if val is not None else default

    session_count    = _get('session_count', 0)
    merge_generation = _get('merge_generation', 0)

    # Active slot: A for sessions 1-20, B for 21-40, back to A after merge
    slot_index = (session_count // _SESSIONS_PER_ADAPTER) % _ADAPTER_CEILING
    active_slot = chr(ord('A') + slot_index)

    # Merge required when we cross a full cycle of both adapter slots
    full_cycle = _SESSIONS_PER_ADAPTER * _ADAPTER_CEILING  # 40
    merge_required = session_count > 0 and session_count % full_cycle == 0

    # Warning: approaching slot boundary (adapter A saturated, B starting)
    adapter_ceiling_warning = (
        session_count > 0
        and session_count % _SESSIONS_PER_ADAPTER == 0
        and not merge_required
    )

    return {
        "current_adapter":        _get('current_adapter', None),
        "session_count":          session_count,
        "merge_generation":       merge_generation,
        "base_model_hash":        _get('base_model_hash', None),
        "adapter_started":        _get('adapter_started', None),
        "active_slot":            active_slot,
        "merge_required":         merge_required,
        "adapter_ceiling_warning": adapter_ceiling_warning,
        "adapter_ceiling":        _ADAPTER_CEILING,
        "sessions_per_adapter":   _SESSIONS_PER_ADAPTER,
    }


def increment_session_count(facts_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Increment session_count in the lifecycle state and return updated status.

    Called by session_to_lora.py after each successful session pipeline run.
    """
    facts = _get_facts(facts_db_path)

    current   = facts.get_fact(_LORA_LIFECYCLE_CATEGORY, 'session_count') or 0
    new_count = current + 1

    facts.set_fact(
        _LORA_LIFECYCLE_CATEGORY, 'session_count', new_count,
        fact_type='fact', confidence=1.0,
    )

    return get_lora_status(facts_db_path)


def set_current_adapter(
    adapter_path: str,
    base_model_hash: Optional[str] = None,
    facts_db_path: Optional[str] = None,
) -> None:
    """
    Persist the current adapter path. Sets adapter_started to now if path changed.

    Args:
        adapter_path:    Path to the active .safetensors adapter file.
        base_model_hash: Optional sha256 of the base model used for training.
        facts_db_path:   Override facts DB path.
    """
    facts = _get_facts(facts_db_path)

    old_path = facts.get_fact(_LORA_LIFECYCLE_CATEGORY, 'current_adapter')

    facts.set_fact(
        _LORA_LIFECYCLE_CATEGORY, 'current_adapter', adapter_path,
        fact_type='fact', confidence=1.0,
    )

    # Reset adapter_started timestamp when the adapter path changes
    if adapter_path != old_path:
        facts.set_fact(
            _LORA_LIFECYCLE_CATEGORY, 'adapter_started',
            datetime.now().isoformat(),
            fact_type='fact', confidence=1.0,
        )

    if base_model_hash:
        facts.set_fact(
            _LORA_LIFECYCLE_CATEGORY, 'base_model_hash', base_model_hash,
            fact_type='fact', confidence=1.0,
        )


def get_current_adapter_path(facts_db_path: Optional[str] = None) -> Optional[str]:
    """
    Return the path to the currently active fused LoRA adapter, or None if not set.

    This is the path written by session_to_lora.py after a successful mlx_lm.fuse.
    The returned path points to the fused .safetensors file — never a raw delta adapter.

    Returns None if no adapter has been trained yet.
    """
    facts = _get_facts(facts_db_path)
    return facts.get_fact(_LORA_LIFECYCLE_CATEGORY, 'current_adapter')


def record_merge(
    new_adapter_path: str,
    base_model_hash: Optional[str] = None,
    facts_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a merge operation: increment merge_generation, reset session_count,
    update current_adapter.

    Call this AFTER a successful mlx_lm.fuse merge that produces a new base checkpoint.

    Returns updated status dict.
    """
    facts = _get_facts(facts_db_path)

    current_gen = facts.get_fact(_LORA_LIFECYCLE_CATEGORY, 'merge_generation') or 0
    new_gen     = current_gen + 1

    facts.set_fact(
        _LORA_LIFECYCLE_CATEGORY, 'merge_generation', new_gen,
        fact_type='fact', confidence=1.0,
    )
    # Reset session counter — the new adapter starts fresh
    facts.set_fact(
        _LORA_LIFECYCLE_CATEGORY, 'session_count', 0,
        fact_type='fact', confidence=1.0,
    )

    set_current_adapter(new_adapter_path, base_model_hash=base_model_hash,
                        facts_db_path=facts_db_path)

    return get_lora_status(facts_db_path)
