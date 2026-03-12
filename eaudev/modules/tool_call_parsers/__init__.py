"""
EauDev Tool Call Parser Registry
=================================
Normalises raw model output into a clean tool call dict + preamble string,
regardless of which local model produced the output.

EauDev's canonical tool call format:
    <tool>{"name": "tool_name", "arg1": "val1", ...}</tool>

Local models may deviate — this registry catches those deviations before
they reach the dispatch layer.

Usage (internal — called by command.py):

    from eaudev.modules.tool_call_parsers import get_parser, ParseResult
    parser = get_parser(model_name)          # e.g. "qwen3.5-35b-a3b"
    tool_call, preamble = parser.parse(text) # tool_call is dict | None

Registration (add a new model):

    from eaudev.modules.tool_call_parsers import register_parser, BaseToolCallParser

    @register_parser("my-model-name")
    class MyModelParser(BaseToolCallParser):
        def parse(self, text: str) -> ParseResult:
            ...
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Type

# ParseResult: (tool_call_dict | None, preamble_str)
ParseResult = tuple[dict | None, str]

# ── Registry ──────────────────────────────────────────────────────────────────

# Maps lowercase model name fragments → parser class
# Registration uses substring matching so "qwen3.5-35b-a3b-q4_k_m" → "qwen3.5"
_PARSER_REGISTRY: dict[str, Type["BaseToolCallParser"]] = {}


def register_parser(*keys: str):
    """
    Decorator to register a parser class under one or more model name keys.
    Keys are matched as substrings of the model name (case-insensitive).

    Example:
        @register_parser("qwen3.5", "qwen3-5")
        class Qwen35Parser(BaseToolCallParser): ...
    """
    def decorator(cls: Type["BaseToolCallParser"]) -> Type["BaseToolCallParser"]:
        for key in keys:
            _PARSER_REGISTRY[key.lower()] = cls
        return cls
    return decorator


def get_parser(model_name: str) -> "BaseToolCallParser":
    """
    Return the best parser for the given model name.
    Falls back to StandardParser if no specific match is found.

    Args:
        model_name: Model name string from config (e.g. "Qwen3.5-35B-A3B-Q4_K_M")
    """
    name_lower = model_name.lower()
    for key, cls in _PARSER_REGISTRY.items():
        if key in name_lower:
            return cls()
    return StandardParser()


def list_parsers() -> list[str]:
    """Return all registered parser keys."""
    return sorted(_PARSER_REGISTRY.keys())


# ── Base class ────────────────────────────────────────────────────────────────

class BaseToolCallParser(ABC):
    """
    Base class for all EauDev tool call parsers.

    Each parser must implement parse(text) → ParseResult.
    The return value is always:
      - (dict, preamble_str): a valid tool call was found
      - (None, original_text): no tool call found
    """

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """
        Parse raw model output for a tool call.

        Returns:
            (tool_call_dict, preamble) if a tool call was found.
            (None, text) if no tool call was found.
        """
        ...

    # ── Shared utilities available to all subclasses ──────────────────────────

    _KNOWN_TOOLS = {
        "read_file", "write_file", "run_bash", "list_directory",
        "create_file", "move_file", "delete_file",
    }

    def _is_known_tool(self, name: str) -> bool:
        return name in self._KNOWN_TOOLS

    def _try_json(self, s: str) -> dict | None:
        """Attempt to parse s as JSON. Returns dict or None."""
        try:
            obj = json.loads(s.strip())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_json_repair(self, s: str) -> dict | None:
        """
        Attempt to recover a truncated or slightly malformed JSON object.
        Strategies:
          1. Strip trailing incomplete key-value pairs
          2. Close unclosed braces
          3. Re-parse
        """
        s = s.strip()
        if not s.startswith("{"):
            return None

        # Count open braces — close any that are unclosed
        open_count = s.count("{") - s.count("}")
        if open_count > 0:
            # Trim trailing incomplete token (common on context truncation)
            # Find the last complete key-value pair by working backwards from last comma
            last_comma = s.rfind(",")
            if last_comma != -1:
                candidate = s[:last_comma] + ("}" * open_count)
            else:
                candidate = s + ("}" * open_count)
            result = self._try_json(candidate)
            if result:
                return result

        return None


# ── Standard parser (default fallback) ───────────────────────────────────────

class StandardParser(BaseToolCallParser):
    """
    Default parser for the canonical EauDev <tool>{...json...}</tool> format.
    Handles all known fallback patterns from the original _extract_tool_call:
      - Clean JSON in <tool> tags
      - Bare tool name in tags + JSON in fenced block after
      - Bare tool name in tags + raw JSON after
      - Raw JSON with no tags (model skipped the wrapper)
      - Truncated / malformed JSON (repair attempt)
    """

    def parse(self, text: str) -> ParseResult:
        # ── Path A: <tool> tags present ───────────────────────────────────────
        if "<tool>" in text:
            return self._parse_with_tags(text)

        # ── Path B: no tags — try to find bare JSON ───────────────────────────
        return self._parse_bare_json(text)

    def _parse_with_tags(self, text: str) -> ParseResult:
        start = text.find("<tool>")
        end = text.find("</tool")
        if end == -1:
            # Unclosed tag — content may be truncated
            json_str = text[start + 6:].strip()
            preamble = text[:start].strip()
            obj = self._try_json(json_str) or self._try_json_repair(json_str)
            if obj and "name" in obj:
                return obj, preamble
            return None, text

        json_str = text[start + 6:end].strip()
        preamble = text[:start].strip()

        # Happy path
        obj = self._try_json(json_str)
        if obj and "name" in obj:
            return obj, preamble

        # Repair attempt (truncated JSON)
        obj = self._try_json_repair(json_str)
        if obj and "name" in obj:
            return obj, preamble

        # Bare tool name in tags — JSON may follow outside the tags
        bare_name = json_str.strip()
        if "{" not in bare_name and bare_name:
            after = text[end:]
            # Try fenced code block
            fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", after, re.DOTALL)
            if fence:
                obj = self._try_json(fence.group(1))
                if obj:
                    obj.setdefault("name", bare_name)
                    return obj, preamble

            # Try raw JSON object
            raw = re.search(r"\{[^{}]*\}", after, re.DOTALL)
            if raw:
                obj = self._try_json(raw.group(0))
                if obj:
                    obj.setdefault("name", bare_name)
                    return obj, preamble

        return None, text

    def _parse_bare_json(self, text: str) -> ParseResult:
        """No <tool> tags — try to extract JSON directly from the text."""
        stripped = text.strip()

        # Entire response is JSON
        if stripped.startswith("{"):
            obj = self._try_json(stripped)
            if obj and "name" in obj and self._is_known_tool(obj["name"]):
                return obj, ""
            obj = self._try_json_repair(stripped)
            if obj and "name" in obj and self._is_known_tool(obj["name"]):
                return obj, ""

        # JSON starts somewhere in the middle (after preamble)
        brace = text.find("{")
        if brace != -1:
            candidate = text[brace:]
            obj = self._try_json(candidate)
            if obj and "name" in obj and self._is_known_tool(obj["name"]):
                return obj, text[:brace].strip()
            obj = self._try_json_repair(candidate)
            if obj and "name" in obj and self._is_known_tool(obj["name"]):
                return obj, text[:brace].strip()

        # Shallow search — simple flat JSON objects only
        for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
            obj = self._try_json(m.group(0))
            if obj and "name" in obj and self._is_known_tool(obj["name"]):
                return obj, text[:m.start()].strip()

        return None, text


# ── Import parsers to trigger registration ────────────────────────────────────
# Each parser module registers itself via @register_parser on import.

from eaudev.modules.tool_call_parsers.qwen35_parser import Qwen35Parser          # noqa: E402, F401
from eaudev.modules.tool_call_parsers.glm_parser import Glm4Parser               # noqa: E402, F401
from eaudev.modules.tool_call_parsers.intern_parser import InternLMParser        # noqa: E402, F401
