"""EauDev CommandRegistry — ported from rovodev, cloud deps removed."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from eaudev.constants import DEFAULT_PANEL_WIDTH

HELP_PREFIX = """\
[bold]Tools[/bold]
[bright_black]EauDev uses local tools: read_file, write_file, run_bash, list_directory.[/bright_black]

[bold]Commands[/bold]
[bright_black]Commands allow you to quickly execute pre-defined EauDev features.[/bright_black]
"""

console = Console()


class CommandRegistry:
    _instance = None
    _commands: dict[str, dict] = defaultdict(dict)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, command: str, sub_command: str | None, help: str, extended_help: str | None = None):
        """Decorator to register a command handler."""
        def decorator(func):
            assert command != "/help"
            assert sub_command != "help"
            assert sub_command not in self._commands[command], \
                f"Duplicate registration: {command} {sub_command}"
            self._commands[command][sub_command] = (func, help, extended_help)
            return func
        return decorator

    @property
    def commands(self) -> list[str]:
        return list(self._commands.keys()) + ["/help"]

    def dispatch(self, message: str, *args, **kwargs) -> Any:
        if not message:
            return None
        # Slash commands are NOT shell — split on whitespace only
        parts = message.strip().split()
        command = parts[0]
        command_args = parts[1:]

        if command not in self.commands:
            return None

        if command == "/help":
            self.help()
            return None

        if command_args and command_args[0] == "help":
            self.help(command)
            return None

        # Determine sub_command
        has_subcommands = len(self._commands[command]) > 1 or (
            len(self._commands[command]) == 1 and None not in self._commands[command]
        )
        if has_subcommands and command_args and command_args[0] in self._commands[command]:
            sub_command = command_args[0]
            remaining = command_args[1:]
        else:
            sub_command = None
            remaining = command_args

        if sub_command not in self._commands[command]:
            if None in self._commands[command]:
                sub_command = None
            else:
                console.print(f"[red]Unknown subcommand: {command} {sub_command}[/red]")
                return None

        func, _, _ = self._commands[command][sub_command]
        kwargs["_message_list"] = remaining
        return func(*args, **kwargs)

    def render_help_table(
        self,
        command: str | None = None,
        sub_command: str | None = None,
        show_header: bool = True,
        command_filter: str | None = None,
    ) -> Table:
        table = Table(show_header=show_header, box=None, pad_edge=False, padding=(0, 5))
        table.add_column("Command", style="bright_black", no_wrap=True)
        table.add_column("Description", style="bright_black")

        cmds = self._commands if command is None else {command: self._commands.get(command, {})}

        for cmd, subcmds in cmds.items():
            if command_filter and not cmd.startswith(command_filter):
                continue
            for sub, (_, help_text, _) in subcmds.items():
                display = f"{cmd} {sub}" if sub else cmd
                if command_filter and not display.startswith(command_filter):
                    continue
                table.add_row(display, help_text)

        return table

    def help(self, command: str | None = None) -> None:
        if command and command in self._commands:
            # Show extended help for specific command
            for sub, (_, help_text, extended) in self._commands[command].items():
                display = f"{command} {sub}" if sub else command
                if extended:
                    console.print(f"\n{extended}\n")
                else:
                    console.print(f"\n  {display} — {help_text}\n")
            return
        table = self.render_help_table()
        console.print(
            Group(
                HELP_PREFIX,
                Panel(table, width=DEFAULT_PANEL_WIDTH, border_style="bright_black"),
                "",
            )
        )


registry = CommandRegistry()
