"""
GLM4 Tool Call Parser
=====================
GLM-4 (4.5 and 4.7) may emit tool calls in two formats:

Format A — EauDev canonical:
    <tool>{"name": "read_file", "path": "/foo.py"}</tool>

Format B — GLM native XML:
    <tool_call>
    <arg_key>name</arg_key>
    <arg_value>read_file</arg_value>
    <arg_key>path</arg_key>
    <arg_value>/foo.py</arg_value>
    </tool_call>

Format C — GLM native with function wrapper:
    <tool_call>{"name": "read_file", "arguments": {"path": "/foo.py"}}</tool_call>

GLM 4.7 adds newlines between arg_key/arg_value pairs and may wrap
the entire JSON in a <tool_call> block instead of EauDev's <tool> block.
"""

from __future__ import annotations

import json
import re

from eaudev.modules.tool_call_parsers import BaseToolCallParser, ParseResult, StandardParser, register_parser


@register_parser("glm4", "glm-4", "glm45", "glm47", "glm4.5", "glm4.7")
class Glm4Parser(BaseToolCallParser):
    """
    Parser for GLM-4 model family (4.5 and 4.7).
    Handles GLM's <tool_call> wrapper with arg_key/arg_value pairs,
    and the JSON-inside-tool_call variant.
    """

    # Match complete or unclosed <tool_call> blocks
    _TOOL_CALL_RE = re.compile(
        r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL
    )

    # GLM 4.5/4.7: <arg_key>k</arg_key> ... <arg_value>v</arg_value>
    # Handles optional whitespace/newlines between tags
    _ARG_PAIR_RE = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        # Check for GLM's <tool_call> block
        tc_match = self._TOOL_CALL_RE.search(text)
        if tc_match:
            body = tc_match.group(1).strip()
            preamble = text[: text.find("<tool_call>")].strip()

            # Variant C: JSON directly inside <tool_call>
            obj = self._try_json(body)
            if obj:
                return self._normalise(obj, preamble, text)

            # Repair truncated JSON
            obj = self._try_json_repair(body)
            if obj:
                return self._normalise(obj, preamble, text)

            # Variant B: arg_key / arg_value pairs
            result = self._parse_arg_pairs(body, preamble, text)
            if result[0] is not None:
                return result

        # Fall through to standard parser
        return StandardParser().parse(text)

    def _normalise(self, obj: dict, preamble: str, original: str) -> ParseResult:
        """
        Normalise GLM's tool call object to EauDev's flat dict format.

        GLM sometimes wraps args under an "arguments" key:
          {"name": "read_file", "arguments": {"path": "/foo.py"}}
        EauDev expects:
          {"name": "read_file", "path": "/foo.py"}
        """
        if "arguments" in obj and isinstance(obj["arguments"], dict):
            flat = {"name": obj.get("name", "")}
            flat.update(obj["arguments"])
            obj = flat

        if "name" not in obj or not self._is_known_tool(obj["name"]):
            return None, original

        return obj, preamble

    def _parse_arg_pairs(self, body: str, preamble: str, original: str) -> ParseResult:
        """Parse GLM's <arg_key>/<arg_value> interleaved format."""
        pairs = self._ARG_PAIR_RE.findall(body)
        if not pairs:
            return None, original

        obj: dict = {}
        for key, val in pairs:
            key = key.strip()
            val = val.strip()
            # Try to parse value as JSON (handles numbers, booleans, objects)
            try:
                obj[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                obj[key] = val

        if "name" not in obj or not self._is_known_tool(obj["name"]):
            return None, original

        return obj, preamble
