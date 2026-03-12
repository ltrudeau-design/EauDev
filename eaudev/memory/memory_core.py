"""
memory_core.py — Unified memory facade for EauDev.

Wraps all 5 SQLite-backed memory layers as a single coherent interface.
This is the internal equivalent of Memory MCP's MemoryMCP class — used
directly by EauDev rather than via MCP protocol.

All operations are deterministic. No inference at read or write time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eaudev.memory.layers.observation import ObservationBuffer
from eaudev.memory.layers.facts import PersistentFacts
from eaudev.memory.layers.fts5 import FullTextSearch
from eaudev.memory.layers.graph import KnowledgeGraph
from eaudev.memory.layers.episodic import EpisodicMemory


class MemoryCore:
    """
    Unified memory infrastructure — 5 integrated SQLite layers.

    Layers:
      ObservationBuffer  — rolling conversation turns (working memory)
      EpisodicMemory     — session-scoped narrative records
      PersistentFacts    — typed, confidence-scored, provenance-tracked facts
      FullTextSearch     — BM25 keyword search over indexed content
      KnowledgeGraph     — local SQLite entity/relationship graph
    """

    def __init__(self, config_path: str = "~/.eaudev/config.yaml"):
        self.config = self._load_config(config_path)

        obs_cfg = self.config.get("observation", {})
        storage = self.config.get("storage", {})

        self.observation = ObservationBuffer(
            max_turns=obs_cfg.get("max_turns", 50),
            db_path=storage.get("observations_path", "~/.eaudev/observations.db"),
            scope=obs_cfg.get("scope", "global"),
        )
        self.facts = PersistentFacts(
            db_path=storage.get("facts_path", "~/.eaudev/facts.db")
        )
        self.fts5 = FullTextSearch(
            db_path=storage.get("fts5_path", "~/.eaudev/fts5.db")
        )
        self.graph = KnowledgeGraph(
            db_path=storage.get("graph_path", "~/.eaudev/graph.db"),
        )
        self.episodic = EpisodicMemory(
            db_path=storage.get("episodic_path", "~/.eaudev/episodic.db"),
        )

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def session_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read-only context injection: recent episodes + facts + buffer stats."""
        recent_episodes = int(params.get("recent_episodes", 3))
        min_confidence  = float(params.get("min_confidence", 0.7))
        try:
            episodes = self.episodic.get_recent(limit=recent_episodes)
            facts    = self.facts.list_facts(min_confidence=min_confidence)
            obs      = self.observation.get_stats()
            return {
                "success":         True,
                "recent_episodes": episodes,
                "facts":           facts,
                "observation":     obs,
                "message": (
                    f"Context loaded: {len(episodes)} recent episodes, "
                    f"{len(facts)} facts (confidence ≥ {min_confidence}), "
                    f"{obs['turn_count']} turns in buffer."
                ),
            }
        except Exception as e:
            return {"error": f"session_start failed: {str(e)}"}

    async def session_end(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress observation buffer → episodic record, optionally clear buffer.
        Caller must provide session_id and summary.
        """
        session_id = params.get("session_id")
        summary    = params.get("summary")
        if not session_id or not summary:
            return {"error": "Missing required params: session_id, summary"}
        clear_buffer = params.get("clear_buffer", True)
        try:
            turns = self.observation.get_messages_for_llm(max_turns=200)
            turn_dicts = [{"role": t["role"], "text": t["content"]} for t in turns]
            episode = self.episodic.compress_and_store(
                session_id=session_id,
                turns=turn_dicts,
                summary=summary,
                what_worked=params.get("what_worked"),
                what_to_avoid=params.get("what_to_avoid"),
                keywords=params.get("keywords"),
                store_source_turns=params.get("store_source_turns", True),
            )
            if clear_buffer:
                self.observation.clear()
            return {
                "success":        True,
                "episode_id":     episode["episode_id"],
                "session_id":     session_id,
                "turn_count":     episode["turn_count"],
                "keywords":       episode["keywords"],
                "buffer_cleared": clear_buffer,
                "message": (
                    f"Session '{session_id}' closed. "
                    f"{episode['turn_count']} turns compressed → episode #{episode['episode_id']}. "
                    f"Buffer {'cleared' if clear_buffer else 'retained'}."
                ),
            }
        except Exception as e:
            return {"error": f"session_end failed: {str(e)}"}

    # ── Episodic layer ────────────────────────────────────────────────────────

    async def store_episode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = params.get("session_id")
        summary    = params.get("summary")
        if not session_id or not summary:
            return {"error": "Missing required params: session_id, summary"}
        try:
            episode_id = self.episodic.store_episode(
                session_id=session_id,
                summary=summary,
                what_worked=params.get("what_worked"),
                what_to_avoid=params.get("what_to_avoid"),
                keywords=params.get("keywords"),
                timestamp=params.get("timestamp"),
                source_turns=params.get("source_turns"),
            )
            return {"success": True, "episode_id": episode_id, "session_id": session_id}
        except Exception as e:
            return {"error": f"store_episode failed: {str(e)}"}

    async def get_recent_episodes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(params.get("limit", 5))
        try:
            results = self.episodic.get_recent(limit=limit)
            return {"success": True, "episodes": results, "count": len(results)}
        except Exception as e:
            return {"error": f"get_recent_episodes failed: {str(e)}"}

    async def search_episodes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        limit = int(params.get("limit", 5))
        if not query:
            return {"error": "Missing required param: query"}
        try:
            results = self.episodic.search(query, limit=limit)
            return {"success": True, "episodes": results, "count": len(results)}
        except Exception as e:
            return {"error": f"search_episodes failed: {str(e)}"}

    # ── Observation layer ─────────────────────────────────────────────────────

    async def store_observation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        role = params.get("role")
        text = params.get("text", "")
        if not role or not text:
            return {"error": "Missing required params: role and text"}
        try:
            self.observation.add_turn(role, text)
            return {"success": True, "message": "Observation stored"}
        except Exception as e:
            return {"error": f"Failed to store observation: {str(e)}"}

    async def get_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_turns = params.get("max_turns", 10)
        try:
            messages = self.observation.get_messages_for_llm(max_turns=max_turns)
            return {"success": True, "messages": messages}
        except Exception as e:
            return {"error": f"Failed to retrieve context: {str(e)}"}

    # ── Facts layer ───────────────────────────────────────────────────────────

    async def store_fact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        category = params.get("category")
        key = params.get("key")
        value = params.get("value")
        if not category or not key or value is None:
            return {"error": "Missing required params: category, key, value"}
        fact_type      = params.get("type", "fact")
        confidence     = float(params.get("confidence", 1.0))
        source_session = params.get("source_session")
        try:
            self.facts.set_fact(
                category, key, value,
                fact_type=fact_type,
                confidence=confidence,
                source_session=source_session,
            )
            return {"success": True, "message": f"Fact stored: {category}.{key} [{fact_type}, conf={confidence}]"}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to store fact: {str(e)}"}

    async def get_fact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        category = params.get("category")
        key = params.get("key")
        if not category or not key:
            return {"error": "Missing required params: category, key"}
        try:
            result = self.facts.get_fact_full(category, key)
            if result is None:
                return {"success": True, "found": False, "fact": None}
            return {"success": True, "found": True, "fact": result}
        except Exception as e:
            return {"error": f"Failed to get fact: {str(e)}"}

    async def list_facts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        category       = params.get("category")
        fact_type      = params.get("type")
        min_confidence = float(params.get("min_confidence", 0.0))
        try:
            results = self.facts.list_facts(
                category=category,
                fact_type=fact_type,
                min_confidence=min_confidence,
            )
            return {"success": True, "facts": results, "count": len(results)}
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to list facts: {str(e)}"}

    # ── FTS5 search layer ─────────────────────────────────────────────────────

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        limit = params.get("limit", 10)
        source = params.get("source")
        if not query:
            return {"error": "Missing required param: query"}
        try:
            results = self.fts5.search(query, limit=limit, source=source)
            return {"success": True, "results": results}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    # ── Knowledge graph layer ─────────────────────────────────────────────────

    async def add_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name     = params.get("name")
        type_    = params.get("type")
        metadata = params.get("metadata", {})
        if not name or not type_:
            return {"error": "Missing name/type"}
        try:
            success = self.graph.add_entity(name, type_, metadata)
            return {"success": success}
        except Exception as e:
            return {"error": f"add_entity failed: {str(e)}"}

    async def add_relationship(self, params: Dict[str, Any]) -> Dict[str, Any]:
        source_name   = params.get("source_name")
        source_type   = params.get("source_type")
        target_name   = params.get("target_name")
        target_type   = params.get("target_type")
        relation_type = params.get("relation_type", "related_to")
        detail        = params.get("detail")
        if not all([source_name, source_type, target_name, target_type]):
            return {"error": "Missing required params"}
        try:
            success = self.graph.add_relationship(
                source_name, source_type,
                target_name, target_type,
                relation_type=relation_type,
                detail=detail,
            )
            return {"success": success}
        except Exception as e:
            return {"error": f"add_relationship failed: {str(e)}"}

    async def get_related(self, params: Dict[str, Any]) -> Dict[str, Any]:
        entity_name = params.get("entity_name")
        max_depth   = int(params.get("max_depth", 2))
        limit       = int(params.get("limit", 20))
        if not entity_name:
            return {"error": "Missing required param: entity_name"}
        try:
            results = self.graph.get_related_entities(entity_name, max_depth=max_depth, limit=limit)
            return {"success": True, "entities": results, "count": len(results)}
        except Exception as e:
            return {"error": f"get_related failed: {str(e)}"}

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "success":     True,
                "observation": self.observation.get_stats(),
                "episodic":    self.episodic.get_stats(),
                "facts":       self.facts.get_stats(),
                "fts5":        self.fts5.get_stats(),
                "graph":       self.graph.get_stats(),
            }
        except Exception as e:
            return {"error": f"get_stats failed: {str(e)}"}

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> Dict[str, Any]:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            return {
                "observation": {"max_turns": 50, "scope": "global"},
                "storage": {
                    "observations_path": "~/.eaudev/observations.db",
                    "episodic_path":     "~/.eaudev/episodic.db",
                    "facts_path":        "~/.eaudev/facts.db",
                    "fts5_path":         "~/.eaudev/fts5.db",
                    "graph_path":        "~/.eaudev/graph.db",
                },
            }
        with open(path_obj) as f:
            return yaml.safe_load(f) or {}
