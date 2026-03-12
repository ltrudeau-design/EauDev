"""
eaudev.memory — EauDev's intrinsic memory system.

Five SQLite-backed layers, all deterministic (no inference at read/write time).
All data lives in ~/.eaudev/.

Layers (importable from eaudev.memory.layers):
    ObservationBuffer  — rolling conversation turns
    EpisodicMemory     — session-scoped narrative records
    PersistentFacts    — typed, confidence-scored facts
    FullTextSearch     — BM25 keyword search
    KnowledgeGraph     — entity/relationship graph with BFS traversal

LoRA consolidation pipeline:
    export_consolidation_artefact() — episodic + facts → alpaca JSONL
    get_lora_status()               — read/write LoRA lifecycle state via facts layer
"""
from eaudev.memory.layers import (
    ObservationBuffer,
    EpisodicMemory,
    PersistentFacts,
    FullTextSearch,
    KnowledgeGraph,
)
from eaudev.memory.memory_core import MemoryCore
from eaudev.memory.consolidation import export_consolidation_artefact
from eaudev.memory.lora_lifecycle import get_lora_status, increment_session_count, get_current_adapter_path

__all__ = [
    "ObservationBuffer",
    "EpisodicMemory",
    "PersistentFacts",
    "FullTextSearch",
    "KnowledgeGraph",
    "MemoryCore",
    "export_consolidation_artefact",
    "get_lora_status",
    "increment_session_count",
    "get_current_adapter_path",
]
