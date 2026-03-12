"""Tool permission manager for EauDev — no pydantic_ai, no logfire."""

from __future__ import annotations

import re
import sys
from typing import Any, Literal

try:
    import bashlex
    _BASHLEX_AVAILABLE = True
except ImportError:
    _BASHLEX_AVAILABLE = False

from eaudev.common.config_model import BashCommandConfig, EauDevConfig, Permission, ToolPermissionsConfig
from eaudev.common.config import load_config, save_config

PermissionScope = Literal["always", "session", "once"]


class ToolPermissionManager:
    """Interactive allow/deny/ask permission manager for EauDev tools."""

    def __init__(self, config: EauDevConfig, config_path: str, interactive: bool = True) -> None:
        self._tool_permissions: ToolPermissionsConfig = config.tool_permissions
        self._config_path = config_path
        self._interactive = interactive and sys.stdin.isatty()
        self._session_bash_decisions: dict[str, Permission] = {}  # Session-scoped bash decisions

    # ── Public API ────────────────────────────────────────────────────────────

    def check_tool(self, tool_call: dict) -> tuple[bool, str | None]:
        """Check whether a tool call is allowed.

        Args:
            tool_call: dict with at least {"name": str} and optionally args.

        Returns:
            (allowed: bool, suppress_message: str | None)
        """
        tool_name = tool_call.get("name", "")

        if self._tool_permissions.allow_all:
            return True, None

        if tool_name in ("bash", "run_bash"):
            return self._handle_bash(tool_call)
        else:
            return self._handle_generic(tool_name)

    # ── Internal handlers ─────────────────────────────────────────────────────

    def _handle_bash(self, tool_call: dict) -> tuple[bool, str | None]:
        args = tool_call.get("args", tool_call)
        command = args.get("command", tool_call.get("command", ""))

        is_compound = _is_compound_or_redirection(command)
        matched = self._find_bash_permission(command, is_compound)

        if matched is None:
            # Check session-scoped decision first
            session_perm = self._session_bash_decisions.get(command)
            if session_perm is not None:
                return (True, None) if session_perm == "allow" else (False, self._deny_msg("session"))
            matched = BashCommandConfig(command=command, permission=self._tool_permissions.bash.default)

        if matched.permission == "allow":
            return True, None
        if matched.permission == "deny":
            return False, "run_bash: denied by config. Do not attempt this command again this session."

        # ask
        pattern = None if is_compound else (command.split()[0] + ".*")
        pattern_or_cmd, permission, scope = self._ask_permission(
            f"bash — `{command}`", pattern=pattern
        )
        if pattern_or_cmd == pattern and pattern:
            matched.command = pattern
        self._apply_scope(scope, permission, bash_cmd=matched.command, is_bash=True)
        matched.permission = permission if scope in ("session", "always") else matched.permission
        if permission == "deny":
            return False, self._deny_msg(scope)
        return True, None

    def _handle_generic(self, tool_name: str) -> tuple[bool, str | None]:
        # MCP tools (server__tool naming) require explicit allowlist
        if "__" in tool_name:
            server_name = tool_name.split("__")[0]
            allowed = self._tool_permissions.allowed_mcp_servers
            if not allowed or server_name in allowed:
                return True, None
            return False, f"{tool_name}: MCP server '{server_name}' is not in allowed_mcp_servers."

        if tool_name not in self._tool_permissions.tools:
            self._tool_permissions.tools[tool_name] = self._tool_permissions.default

        perm = self._tool_permissions.tools[tool_name]
        if perm == "allow":
            return True, None
        if perm == "deny":
            return False, f"{tool_name}: denied by config. Do not attempt this again this session."

        # ask
        _, permission, scope = self._ask_permission(tool_name)
        self._apply_scope(scope, permission, tool_name=tool_name)
        if scope in ("session", "always"):
            self._tool_permissions.tools[tool_name] = permission
        if permission == "deny":
            return False, self._deny_msg(scope)
        return True, None

    def _find_bash_permission(self, command: str, is_compound: bool) -> BashCommandConfig | None:
        # Exact match first
        for entry in self._tool_permissions.bash.commands:
            if command == entry.command:
                return entry
        if not is_compound:
            # Regex match
            for entry in self._tool_permissions.bash.commands:
                if _valid_regex(entry.command) and re.fullmatch(entry.command, command):
                    return entry
        return None

    def _apply_scope(
        self,
        scope: PermissionScope,
        permission: Permission,
        bash_cmd: str | None = None,
        tool_name: str | None = None,
        is_bash: bool = False,
    ) -> None:
        if scope == "always":
            if is_bash and bash_cmd:
                self._write_bash_permission(bash_cmd, permission)
            elif tool_name:
                self._write_tool_permission(tool_name, permission)
        elif scope == "session" and is_bash and bash_cmd:
            # Record session-scoped decision for bash commands
            self._session_bash_decisions[bash_cmd] = permission

    def _write_tool_permission(self, tool_name: str, permission: Permission) -> None:
        try:
            config = load_config(self._config_path)
            config.tool_permissions.tools[tool_name] = permission
            save_config(config, self._config_path)
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to persist permission for '{tool_name}': {e}")

    def _write_bash_permission(self, command: str, permission: Permission) -> None:
        try:
            config = load_config(self._config_path)
            for entry in config.tool_permissions.bash.commands:
                if entry.command == command:
                    entry.permission = permission
                    break
            else:
                config.tool_permissions.bash.commands.append(
                    BashCommandConfig(command=command, permission=permission)
                )
            save_config(config, self._config_path)
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to persist permission for '{command}': {e}")

    def _ask_permission(
        self, label: str, pattern: str | None = None
    ) -> tuple[str, Permission, PermissionScope]:
        if not self._interactive:
            return label, "deny", "session"

        from eaudev.ui.components.user_menu_panel import Choice

        display = label if len(label) <= 200 else label[:200] + "..."
        enable_always = len(label) <= 200

        choices: list[Choice] = [
            Choice(name="Allow (once)",    value=(label, "allow", "once")),
            Choice(name="Allow (session)", value=(label, "allow", "session")),
        ]
        if enable_always:
            choices.append(Choice(name="Allow (always)", value=(label, "allow", "always")))
        if pattern:
            pattern_cmd = pattern.rstrip(".*")
            choices.append(
                Choice(
                    name=f"Allow (always) for all '{pattern_cmd}' commands",
                    value=(pattern, "allow", "always"),
                )
            )
        choices.extend([
            Choice(name="Deny (once)",    value=(label, "deny", "once")),
            Choice(name="Deny (session)", value=(label, "deny", "session")),
        ])
        if enable_always:
            choices.append(Choice(name="Deny (always)", value=(label, "deny", "always")))

        # user_menu_panel is async. We need to run it from sync code.
        # Check for existing event loop and handle both cases:
        import asyncio
        import threading
        from eaudev.ui.components.user_menu_panel import user_menu_panel

        result_holder: list[Any] = [None]
        exc_holder: list[Exception | None] = [None]

        def _run_with_new_loop():
            """Run user_menu_panel in a new event loop (thread-safe)."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_holder[0] = loop.run_until_complete(
                    user_menu_panel(
                        choices=choices,
                        message=f"Requesting permission to use: {display}",
                        title="Permission required",
                        escape_return_value=(label, "deny", "once"),
                    )
                )
            except Exception as e:
                exc_holder[0] = e
            finally:
                loop.close()

        def _run_with_existing_loop(loop: asyncio.AbstractEventLoop):
            """Schedule user_menu_panel on existing loop (must be called from thread)."""
            from rich.console import Console
            console = Console()
            console.print("[yellow]⏳ Waiting for permission response…[/yellow]")
            future = asyncio.run_coroutine_threadsafe(
                user_menu_panel(
                    choices=choices,
                    message=f"Requesting permission to use: {display}",
                    title="Permission required",
                    escape_return_value=(label, "deny", "once"),
                ),
                loop,
            )
            try:
                result_holder[0] = future.result(timeout=60)
            except TimeoutError:
                console.print("[red]Permission prompt timed out — defaulting to deny.[/red]")
                result_holder[0] = None
            except Exception as e:
                exc_holder[0] = e

        # Try to detect if we're inside a running event loop
        try:
            loop = asyncio.get_running_loop()
            # We're inside a loop — run in a thread using the existing loop
            t = threading.Thread(target=_run_with_existing_loop, args=(loop,), daemon=True)
        except RuntimeError:
            # No running loop — safe to create new one in thread
            t = threading.Thread(target=_run_with_new_loop, daemon=True)

        t.start()
        t.join()
        if exc_holder[0]:
            raise exc_holder[0]
        result = result_holder[0]
        if result is None:
            result = (label, "deny", "once")
        return result

    @staticmethod
    def _deny_msg(scope: PermissionScope) -> str:
        if scope in ("session", "always"):
            return "User denied permission. DO NOT attempt to use this tool again."
        return "User denied permission for this call."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_compound_or_redirection(command: str) -> bool:
    if not _BASHLEX_AVAILABLE:
        return False
    try:
        for ast in bashlex.parse(command):
            if _has_compound_or_redirection(ast):
                return True
        return False
    except Exception:
        return True  # Treat unparseable commands as compound


def _has_compound_or_redirection(node) -> bool:
    if hasattr(node, "kind"):
        if node.kind in ("operator", "redirect", "commandsubstitution", "processsubstitution", "pipe"):
            return True
        for attr in ("parts", "list", "command"):
            value = getattr(node, attr, None)
            if value:
                children = value if isinstance(value, list) else [value]
                if any(_has_compound_or_redirection(c) for c in children):
                    return True
    return False


def _valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False
