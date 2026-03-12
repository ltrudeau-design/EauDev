"""
InternLM Tool Call Parser
=========================
InternLM2 / InternLM2.5 / InternS1-Mini may emit tool calls in two formats:

Format A — EauDev canonical:
    <tool>{"name": "read_file", "path": "/foo.py"}</tool>

Format B — InternLM native plugin format:
    <|plugin|>{"name": "read_file", "parameters": {"path": "/foo.py"}}<eot_id>

Format C — InternLM action format:
    <|action_start|><|plugin|>
    {"name": "read_file", "parameters": {"path": "/foo.py"}}
    <|action_end|>

All variants normalise to EauDev's flat dict:
    {"name": "read_file", "path": "/foo.py"}
"""

from __future__ import annotations

import json
import re

from eaudev.modules.tool_call_parsers import BaseToolCallParser, ParseResult, StandardParser, register_parser


@register_parser("internlm", "intern-lm", "interns1", "intern_s1", "internvl")
class InternLMParser(BaseToolCallParser):
    """
    Parser for InternLM2 / InternLM2.5 / InternS1 model families.
    Handles the <|plugin|> and <|action_start|> tool call formats.
    """

    # Format B: <|plugin|>{...}<eot_id> or <|plugin|>{...}
    _PLUGIN_RE = re.compile(
        r"<\|plugin\|>(.*?)(?:<eot_id>|<\|action_end\|>|$)", re.DOTALL
    )

    # Format C: <|action_start|><|plugin|>{...}<|action_end|>
    _ACTION_RE = re.compile(
        r"<\|action_start\|>\s*<\|plugin\|>(.*?)<\|action_end\|>", re.DOTALL
    )

    def parse(self, text: str) -> ParseResult:
        # Format C first (more specific)
        action_match = self._ACTION_RE.search(text)
        if action_match:
            body = action_match.group(1).strip()
            preamble = text[: text.find("<|action_start|>")].strip()
            return self._parse_body(body, preamble, text)

        # Format B
        plugin_match = self._PLUGIN_RE.search(text)
        if plugin_match:
            body = plugin_match.group(1).strip()
            preamble = text[: text.find("<|plugin|>")].strip()
            return self._parse_body(body, preamble, text)

        # Fall through to standard parser
        return StandardParser().parse(text)

    def _parse_body(self, body: str, preamble: str, original: str) -> ParseResult:
        """Parse and normalise a JSON body from an InternLM tool call."""
        obj = self._try_json(body) or self._try_json_repair(body)
        if not obj or "name" not in obj:
            return None, original

        # InternLM wraps args under "parameters" key
        if "parameters" in obj and isinstance(obj["parameters"], dict):
            flat = {"name": obj["name"]}
            flat.update(obj["parameters"])
            obj = flat

        if not self._is_known_tool(obj["name"]):
            return None, original

        return obj, preamble
