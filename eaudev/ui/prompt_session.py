"""FilteredFileHistory + PromptSession wrapper for EauDev."""

from __future__ import annotations

from prompt_toolkit import PromptSession as _PTSession
from prompt_toolkit.history import FileHistory

from eaudev.constants import DEFAULT_EXIT_COMMANDS


class FilteredFileHistory(FileHistory):
    """FileHistory that filters out exit commands and blanks, and tracks last string."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_string: str | None = None

    @property
    def last_string(self) -> str:
        return self._last_string or ""

    def append_string(self, string: str) -> None:
        self._last_string = string
        if not string.strip() or string in DEFAULT_EXIT_COMMANDS:
            return
        return super().append_string(string)

    def load_history_strings(self):
        from eaudev.commands.run.command_registry import registry
        return list(super().load_history_strings()) + registry.commands


class PromptSession(_PTSession):
    """PromptSession that uses the Rich Live input panel."""

    def prompt_async(self, *args, session_context=None, **kwargs):
        from eaudev.ui.components.user_input_panel import user_input_panel
        return user_input_panel(self, session_context)
