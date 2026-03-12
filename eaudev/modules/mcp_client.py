"""EauDev MCP client — JSON-RPC 2.0 over stdio.

Spawns each configured MCP server as a child process, discovers its tools
via tools/list, and routes tool calls via tools/call.

Config: ~/.eaudev/mcp.json
Format:
    {
        "mcpServers": {
            "archive": {
                "command": "python3",
                "args": ["/abs/path/to/server.py"],
                "env": {}          # optional extra env vars
            }
        }
    }

Usage (from command.py):
    from eaudev.modules.mcp_client import MCPClientManager
    mcp_manager = MCPClientManager()
    mcp_manager.start_all()           # spawn all servers, discover tools
    tool_names = mcp_manager.tool_names()
    result = mcp_manager.call_tool("archive__search_archive_tool", {"query": "...", "limit": 5})
    mcp_manager.stop_all()            # on exit

Tool naming convention: "{server_name}__{tool_name}"
e.g. archive server with tool "search_archive_tool" → "archive__search_archive_tool"
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from eaudev.constants import MCP_CONFIG_PATH

console = Console()

# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

_rpc_id = 0
_rpc_lock = threading.Lock()


def _next_id() -> int:
    global _rpc_id
    with _rpc_lock:
        _rpc_id += 1
        return _rpc_id


def _make_request(method: str, params: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": method,
        "params": params or {},
    }


def _make_notification(method: str, params: dict | None = None) -> dict:
    """A JSON-RPC notification has no 'id' field — server sends no response."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }


# ── MCPServer — one per spawned child ─────────────────────────────────────────

class MCPServer:
    """Manages a single stdio MCP server process."""

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.tools: list[dict] = []   # raw tool schemas from tools/list
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Spawn the server process and perform MCP initialisation handshake.

        Returns True on success, False on failure.
        """
        cmd = [self.command] + self.args
        env = {**os.environ, **self.env}

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            logger.warning(f"[mcp:{self.name}] command not found: {e}")
            return False
        except Exception as e:
            logger.warning(f"[mcp:{self.name}] failed to spawn: {e}")
            return False

        # Poll loop — wait up to 0.5s for process to stabilise
        for _ in range(10):
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)

        # Check it didn't immediately exit
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            logger.warning(f"[mcp:{self.name}] process exited immediately. stderr: {stderr[:300]}")
            return False

        # MCP initialise handshake
        try:
            init_result = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eaudev", "version": "0.1.0"},
            })
            if "error" in init_result:
                logger.warning(f"[mcp:{self.name}] initialize error: {init_result['error']}")
                return False

            # Acknowledge with initialized notification (no id — no response expected)
            self._notify("notifications/initialized")

            # Discover tools
            tools_result = self._request("tools/list")
            if "error" in tools_result:
                logger.warning(f"[mcp:{self.name}] tools/list error: {tools_result['error']}")
                return False

            self.tools = tools_result.get("result", {}).get("tools", [])
            self._started = True
            logger.info(f"[mcp:{self.name}] started — {len(self.tools)} tool(s): "
                        f"{[t['name'] for t in self.tools]}")
            return True

        except Exception as e:
            logger.warning(f"[mcp:{self.name}] handshake failed: {e}")
            return False

    def stop(self) -> None:
        """Terminate the server process cleanly."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._started = False

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── RPC transport ─────────────────────────────────────────────────────────

    def _send(self, obj: dict) -> None:
        """Write a JSON-RPC message to the child's stdin."""
        if not self._proc or not self._proc.stdin:
            raise RuntimeError(f"[mcp:{self.name}] process not running")
        line = json.dumps(obj) + "\n"
        with self._lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def _recv(self, timeout: float = 15.0) -> dict:
        """Read one JSON-RPC message from the child's stdout."""
        if not self._proc or not self._proc.stdout:
            raise RuntimeError(f"[mcp:{self.name}] process not running")

        deadline = time.time() + timeout
        while time.time() < deadline:
            # Non-blocking readline via select on the fd
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.1)
            if ready:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"[mcp:{self.name}] stdout closed")
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as e:
                    logger.debug(f"[mcp:{self.name}] non-JSON line: {line[:120]} — {e}")
                    continue

        raise TimeoutError(f"[mcp:{self.name}] timeout waiting for response")

    def _request(self, method: str, params: dict | None = None, timeout: float = 15.0) -> dict:
        """Send a request and wait for the matching response."""
        req = _make_request(method, params)
        self._send(req)
        req_id = req["id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._recv(timeout=max(0.1, deadline - time.time()))
            # Skip notifications (no "id") while waiting for our response
            if resp.get("id") == req_id:
                return resp
            # It's a notification or a mismatched id — log and discard
            if "id" not in resp:
                logger.debug(f"[mcp:{self.name}] notification: {resp.get('method', '?')}")
            else:
                logger.debug(f"[mcp:{self.name}] unexpected id {resp.get('id')}, expected {req_id}")

        raise TimeoutError(f"[mcp:{self.name}] no response to {method} within {timeout}s")

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send a notification (no response expected)."""
        self._send(_make_notification(method, params))

    # ── Tool call ─────────────────────────────────────────────────────────────

    # Tools that run long-running processes (inference servers, batch jobs)
    # get an extended timeout. All others use the default 30s.
    _LONG_RUNNING_TOOLS = {
        "archive_document_tool",   # starts inference server, enriches chunks
        "analyze_target",          # full Analyst MCP pipeline run
        "survey_target",           # vision processing + chunking
    }
    _DEFAULT_TIMEOUT = 30.0
    _LONG_TIMEOUT    = 900.0   # 15 minutes for enrichment runs

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return a plain-text result string for context injection."""
        if not self._started or not self.alive:
            return f"[mcp:{self.name}] server not running"

        timeout = self._LONG_TIMEOUT if tool_name in self._LONG_RUNNING_TOOLS else self._DEFAULT_TIMEOUT

        try:
            resp = self._request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            }, timeout=timeout)
        except TimeoutError as e:
            return f"[mcp:{self.name}:{tool_name} error: {e}]"
        except Exception as e:
            return f"[mcp:{self.name}:{tool_name} error: {e}]"

        if "error" in resp:
            err = resp["error"]
            return f"[mcp:{self.name}:{tool_name} error: {err.get('message', err)}]"

        # tools/call result: {"content": [{"type": "text", "text": "..."}], "isError": bool}
        result_payload = resp.get("result", {})
        is_error = result_payload.get("isError", False)
        content_items = result_payload.get("content", [])

        parts: list[str] = []
        for item in content_items:
            item_type = item.get("type", "text")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "resource":
                # Embedded resource — stringify the URI and text
                resource = item.get("resource", {})
                parts.append(f"[resource: {resource.get('uri', '?')}]\n{resource.get('text', '')}")
            else:
                # Unknown content type — JSON-encode it
                parts.append(json.dumps(item))

        result_text = "\n".join(parts).strip() or "(empty result)"
        prefix = f"[mcp:{self.name}:{tool_name}{'  ERROR' if is_error else ''}]\n"
        return prefix + result_text


# ── MCPClientManager ──────────────────────────────────────────────────────────

class MCPClientManager:
    """Manages the pool of all configured MCP server connections.

    Loads ~/.eaudev/mcp.json, spawns servers, discovers tools.
    Tool names are namespaced: "{server_name}__{tool_name}".
    """

    # Separator used between server name and tool name in the combined namespace
    SEP = "__"

    def __init__(self, config_path: Path | str = MCP_CONFIG_PATH) -> None:
        self._config_path = Path(config_path).expanduser()
        self._servers: dict[str, MCPServer] = {}
        # Maps qualified tool name → (server, raw_tool_name, schema)
        self._tool_map: dict[str, tuple[MCPServer, str, dict]] = {}
        self._map_lock = threading.Lock()

    # ── Config loading ────────────────────────────────────────────────────────

    def _load_config(self) -> dict[str, dict]:
        """Return {server_name: {command, args, env}} from mcp.json."""
        if not self._config_path.exists():
            return {}
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            return data.get("mcpServers", {})
        except Exception as e:
            logger.warning(f"[mcp] failed to load {self._config_path}: {e}")
            return {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_all(self) -> None:
        """Spawn all servers listed in mcp.json and discover their tools."""
        server_configs = self._load_config()
        if not server_configs:
            return

        for name, cfg in server_configs.items():
            command = cfg.get("command", "python3")
            args = cfg.get("args", [])
            env = cfg.get("env", {})

            server = MCPServer(name=name, command=command, args=args, env=env)
            ok = server.start()

            if ok:
                self._servers[name] = server
                with self._map_lock:
                    for tool in server.tools:
                        tool_name_raw = tool.get("name", "")
                        qualified = f"{name}{self.SEP}{tool_name_raw}"
                        self._tool_map[qualified] = (server, tool_name_raw, tool)
            else:
                console.print(
                    f"[bright_black][mcp] Failed to start server '{name}' — "
                    f"it will be unavailable this session.[/bright_black]"
                )

        if self._tool_map:
            console.print(
                f"[bright_black][mcp] {len(self._tool_map)} tool(s) from "
                f"{len(self._servers)} server(s): "
                f"{', '.join(self._servers.keys())}[/bright_black]"
            )

    def stop_all(self) -> None:
        """Terminate all running server processes."""
        for name, server in self._servers.items():
            server.stop()
            logger.debug(f"[mcp] stopped server '{name}'")
        self._servers.clear()
        with self._map_lock:
            self._tool_map.clear()

    # ── Tool discovery ────────────────────────────────────────────────────────

    def tool_names(self) -> list[str]:
        """Return all qualified tool names available from running servers."""
        with self._map_lock:
            return list(self._tool_map.keys())

    def tool_schemas(self) -> list[dict]:
        """Return all tool schemas with qualified names, suitable for system prompt injection."""
        schemas = []
        with self._map_lock:
            for qualified, (_, raw_name, schema) in self._tool_map.items():
                schemas.append({
                    **schema,
                    "name": qualified,
                    # Keep original name available for reference
                    "_server_tool": raw_name,
                })
        return schemas

    def build_openai_tools(self, exclude_servers: frozenset[str] | None = None) -> list[dict]:
        """Return MCP tools as OpenAI function schema objects for the `tools` API parameter."""
        exclude = exclude_servers or frozenset()
        tools = []
        with self._map_lock:
            for qualified, (server, raw_name, schema) in self._tool_map.items():
                if server.name in exclude:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                })
        return tools

    def has_tool(self, qualified_name: str) -> bool:
        return qualified_name in self._tool_map

    def server_status(self) -> dict[str, dict]:
        """Return status info for every configured server.

        Returns a dict keyed by server name:
            {
                "alive": bool,
                "tool_count": int,
                "tools": [list of qualified tool names],
            }
        """
        result: dict[str, dict] = {}
        with self._map_lock:
            for server in self._servers.values():
                qualified_tools = [
                    q for q, (srv, _, _) in self._tool_map.items()
                    if srv.name == server.name
                ]
                result[server.name] = {
                    "alive": server.alive,
                    "tool_count": len(qualified_tools),
                    "tools": qualified_tools,
                }
        return result

    # ── Tool execution ────────────────────────────────────────────────────────

    def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate server.

        Args:
            qualified_name: "{server}__{tool}" e.g. "archive__search_archive_tool"
            arguments: Dict of arguments matching the tool's inputSchema.

        Returns:
            Plain text result string for injection into the LLM context.
        """
        with self._map_lock:
            if qualified_name not in self._tool_map:
                return f"[mcp: unknown tool '{qualified_name}']"

            server, raw_name, _ = self._tool_map[qualified_name]

        # Restart dead server transparently
        if not server.alive:
            logger.warning(f"[mcp:{server.name}] server died — attempting restart")
            ok = server.start()
            if not ok:
                return f"[mcp:{server.name}:{raw_name} error: server not running and restart failed]"
            # Re-register tools in case they changed
            with self._map_lock:
                self._tool_map = {
                    k: v for k, v in self._tool_map.items()
                    if v[0].name != server.name
                }
                for tool in server.tools:
                    q = f"{server.name}{self.SEP}{tool['name']}"
                    self._tool_map[q] = (server, tool["name"], tool)
            logger.info(f"[mcp:{server.name}] restarted — {len(server.tools)} tool(s) re-registered")

        return server.call_tool(raw_name, arguments)

    # ── System prompt injection ───────────────────────────────────────────────

    def build_tool_descriptions(self, exclude_servers: frozenset[str] | None = None) -> str:
        """Build a concise tool listing for injection into the system prompt.

        Args:
            exclude_servers: Server names to hide from the model (e.g. internal
                             infrastructure servers the model should not call directly).

        Returns empty string if no MCP tools are available (after exclusions).
        """
        exclude = exclude_servers or frozenset()
        with self._map_lock:
            visible = {
                q: v for q, v in self._tool_map.items()
                if v[0].name not in exclude  # v[0] is the MCPServer object; .name is the string key
            }
        if not visible:
            return ""

        lines = ["", "# MCP Tools", ""]
        lines.append(
            "Additional tools are available via MCP servers. "
            "Call them exactly like the 7 built-in tools — "
            "one <tool_call>...</tool_call> block per response.\n"
        )

        for qualified, (server, raw_name, schema) in visible.items():
            desc = schema.get("description", "").strip()
            input_schema = schema.get("inputSchema", {})
            props = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            # Build a compact argument list
            arg_parts = []
            for prop_name, prop_schema in props.items():
                prop_type = prop_schema.get("type", "any")
                is_req = prop_name in required
                suffix = "" if is_req else "?"
                arg_parts.append(f'"{prop_name}"{suffix}: {prop_type}')

            args_str = ", ".join(arg_parts)
            lines.append(f'  {qualified}({args_str})')
            if desc:
                lines.append(f'    → {desc}')
            lines.append("")

        lines.append(
            'Example call:\n'
            '  <tool_call>\n'
            '  {"name": "archive__search_archive_tool", "arguments": {"query": "DSP algorithms", "limit": 5}}\n'
            '  </tool_call>'
        )
        return "\n".join(lines)


# ── Singleton accessor ────────────────────────────────────────────────────────

_manager: MCPClientManager | None = None
_manager_lock = threading.Lock()


def get_mcp_manager() -> MCPClientManager:
    """Return the global MCPClientManager instance (created on first call)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPClientManager()
        return _manager
