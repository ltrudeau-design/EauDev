"""
Qwen3.5 Tool Call Parser
========================
Qwen3.5-9B MLX (via mlx_lm.server) emits tool calls in the nous/hermes format:

Format A — Nous JSON (primary — what Qwen3.5-9B generates via mlx_lm.server):
    <tool_call>
    {"name": "read_file", "arguments": {"path": "/foo.py"}}
    </tool_call>

Format B — Qwen3-Coder XML (legacy, when model uses its native chat template):
    <tool_call>
    <function=read_file>
    <parameter=path>/foo.py</parameter>
    </function>
    </tool_call>

Format C — Legacy EauDev canonical (RovoDev-style):
    <tool>{"name": "read_file", "path": "/foo.py"}</tool>

Format D — Thinking preamble followed by any of the above:
    <think>
    I need to read the file...
    </think>
    <tool_call>...</tool_call>

This parser handles all four, stripping thinking blocks first and normalising
the tool call into a flat dict that _dispatch_tool expects.
Note: "arguments" flattening is also done in _dispatch_tool as a safety net.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from eaudev.modules.tool_call_parsers import BaseToolCallParser, ParseResult, StandardParser, register_parser


def _coerce_value(s: str) -> Any:
    """Convert a parameter string to a native Python type."""
    s = s.strip()
    if s.lower() == "null":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        pass
    return s


@register_parser("qwen3.5", "qwen3-5", "qwen3_coder", "qwen3-coder")
class Qwen35Parser(BaseToolCallParser):
    """
    Parser for Qwen3.5 and Qwen3-Coder model families.
    Handles both the canonical EauDev <tool> format and the native
    Qwen3-Coder XML <tool_call><function=...> format, plus thinking blocks.
    """

    # Strip <think>...</think> blocks (including partial/unclosed ones at context edge)
    _THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

    # Qwen3-Coder XML format patterns
    _TOOL_CALL_RE = re.compile(
        r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL
    )
    _FUNCTION_RE = re.compile(
        r"<function=(.*?)(?:</function>|$)", re.DOTALL
    )
    _PARAMETER_RE = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
        re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        # Strip thinking blocks first — must never parse <tool_call> inside <think>
        clean = self._THINK_RE.sub("", text).strip()

        # Try <tool_call> tag (covers both nous JSON and Qwen3-Coder XML)
        tc_match = self._TOOL_CALL_RE.search(clean)
        if tc_match:
            body = tc_match.group(1).strip()
            preamble = clean[:clean.find("<tool_call>")].strip()

            # Attempt nous JSON format first:
            # {"name": "...", "arguments": {...}}
            result = self._parse_nous_json(body, preamble, clean)
            if result[0] is not None:
                return result

            # Attempt Qwen3-Coder XML format:
            # <function=...><parameter=k>v</parameter></function>
            result = self._parse_xml_format(body, clean)
            if result[0] is not None:
                return result

        # Fall through to standard <tool> / bare JSON parser
        return StandardParser().parse(clean)

    def _parse_nous_json(self, body: str, preamble: str, original: str) -> ParseResult:
        """Parse nous hermes JSON format: {"name": "fn", "arguments": {...}}"""
        obj = self._try_json(body) or self._try_json_repair(body)
        if obj is None or "name" not in obj:
            return None, original

        fn_name = obj.get("name", "")
        if not fn_name or not self._is_known_tool(fn_name):
            # MCP tools contain "__" — also valid
            if "__" not in str(fn_name):
                return None, original

        # Flatten arguments dict into the root dict
        arguments = obj.get("arguments", {})
        if isinstance(arguments, dict):
            flat = {"name": fn_name, **arguments}
        else:
            flat = {"name": fn_name}

        return flat, preamble

    def _parse_xml_format(self, tool_call_body: str, original: str) -> ParseResult:
        """Parse Qwen3-Coder <function=name><parameter=k>v</parameter></function>."""
        fn_match = self._FUNCTION_RE.search(tool_call_body)
        if not fn_match:
            return None, original

        fn_header = fn_match.group(1)
        # Function name is the first line/word of the header
        fn_name = fn_header.split("\n")[0].strip().rstrip(">")

        fn_body = fn_match.group(0)[len(fn_match.group(0)) - len(fn_match.group(0)):]
        # Re-extract body after the function name line
        fn_body = fn_match.group(1)[len(fn_name):]

        params: dict[str, Any] = {"name": fn_name}
        for pm in self._PARAMETER_RE.finditer(fn_body):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            params[key] = _coerce_value(val)

        if not self._is_known_tool(fn_name):
            # Unknown tool name — don't dispatch, return as-is
            return None, original

        preamble = original[: original.find("<tool_call>")].strip()
        return params, preamble
