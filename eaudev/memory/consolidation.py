"""
consolidation.py — export_consolidation_artefact()

Exports episodic record + high-confidence facts as alpaca JSONL for mlx_lm.lora.

Output format (one JSON object per line):
  {"instruction": "Summarise what was accomplished in this session.", "input": "<keywords>", "output": "<summary>"}
  {"instruction": "What worked well in this session?", "input": "<keywords>", "output": "<what_worked>"}
  {"instruction": "What should be avoided based on this session?", "input": "<keywords>", "output": "<what_to_avoid>"}
  {"instruction": "Given this decision, what is the rationale?", "input": "<category.key>", "output": "<fact value>"}

All operations are deterministic. No inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_EAUDEV_DIR = Path.home() / '.eaudev'

# Map fact types to appropriate instruction templates for better training signal
_FACT_INSTRUCTIONS: Dict[str, str] = {
    "fact":             "What is known about this?",
    "preference":       "What is the user's preference regarding this?",
    "working_solution": "What is the confirmed working approach for this?",
    "gotcha":           "What is the counterintuitive behaviour to remember about this?",
    "decision":         "What is the rationale for this architectural decision?",
    "failure":          "What approach failed and why?",
}
_DEFAULT_INSTRUCTION = "Given this decision, what is the rationale?"


def export_consolidation_artefact(
    session_id: str,
    output_path: str,
    include_facts: bool = True,
    min_fact_confidence: float = 0.8,
    include_graph_context: bool = False,
    episodic_db_path: Optional[str] = None,
    facts_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export episodic record + high-confidence facts as alpaca JSONL.

    Args:
        session_id:           Session to export (most recent episode for this session_id).
        output_path:          Path to write the JSONL file.
        include_facts:        Include high-confidence facts in output (default True).
        min_fact_confidence:  Minimum confidence threshold for included facts (default 0.8).
        include_graph_context: Reserved for Level 2 — unused at Level 1 (default False).
        episodic_db_path:     Override episodic DB path (defaults to ~/.eaudev/episodic.db).
        facts_db_path:        Override facts DB path (defaults to ~/.eaudev/facts.db).

    Returns:
        dict with: success, session_id, output_path, record_count, episode_found, facts_count
    """
    from eaudev.memory.layers.episodic import EpisodicMemory
    from eaudev.memory.layers.facts import PersistentFacts

    ep_path = episodic_db_path or str(_EAUDEV_DIR / 'episodic.db')
    fa_path = facts_db_path or str(_EAUDEV_DIR / 'facts.db')

    episodic = EpisodicMemory(db_path=ep_path)
    records: list[dict] = []

    # ── Episodic record ────────────────────────────────────────────────────────
    episode = episodic.get_by_session(session_id)
    episode_found = episode is not None

    if episode_found:
        keywords_str = ', '.join(episode.get('keywords') or [])
        summary      = (episode.get('summary') or '').strip()
        what_worked  = (episode.get('what_worked') or '').strip()
        what_to_avoid = (episode.get('what_to_avoid') or '').strip()

        if summary:
            records.append({
                "instruction": "Summarise what was accomplished in this session.",
                "input":  keywords_str,
                "output": summary,
            })

        if what_worked:
            records.append({
                "instruction": "What worked well in this session?",
                "input":  keywords_str,
                "output": what_worked,
            })

        if what_to_avoid:
            records.append({
                "instruction": "What should be avoided based on this session?",
                "input":  keywords_str,
                "output": what_to_avoid,
            })

    # ── High-confidence facts ──────────────────────────────────────────────────
    facts_count = 0
    if include_facts:
        facts = PersistentFacts(db_path=fa_path)
        high_conf = facts.list_facts(min_confidence=min_fact_confidence)

        # lora_lifecycle facts are internal state — exclude from training material
        high_conf = [f for f in high_conf if f.get('category') != 'lora_lifecycle']

        for fact in high_conf:
            key      = fact.get('key', '')
            value    = fact.get('value', '')
            category = fact.get('category', '')
            fact_type = fact.get('type', 'fact')
            instruction = _FACT_INSTRUCTIONS.get(fact_type, _DEFAULT_INSTRUCTION)
            records.append({
                "instruction": instruction,
                "input":  f"{category}.{key}",
                "output": str(value),
            })
        facts_count = len(high_conf)

    # ── Write JSONL ────────────────────────────────────────────────────────────
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return {
        "success":      True,
        "session_id":   session_id,
        "output_path":  str(out_path),
        "record_count": len(records),
        "episode_found": episode_found,
        "facts_count":  facts_count,
    }
