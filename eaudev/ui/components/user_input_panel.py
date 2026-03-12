"""Rich Live input panel with > prompt, backslash continuation, tab-indent, and suggestions."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import to_filter
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output import create_output
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from eaudev.constants import DEFAULT_PANEL_WIDTH
from eaudev.ui.components.token_display import display_token_usage

FOOTER_TEXT = (
    '[bright_black]Type "/" for available commands.'
    + (" " * 64)
    + "Uses AI. Verify results.[/bright_black]"
)

console = Console()


async def user_input_panel(
    prompt_session: PromptSession | None = None,
    session_context=None,  # token count or None; kept compatible with rovodev API
) -> str:
    """Async Rich Live input panel. Returns the user's input string."""
    buffer = (
        Buffer(multiline=True)
        if prompt_session is None
        else prompt_session._create_default_buffer()
    )
    buffer.multiline = to_filter(True)
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(exception=SystemExit(0))

    def _accept_suggestion() -> bool:
        if (
            buffer.document.is_cursor_at_the_end
            and buffer.suggestion
            and buffer.suggestion.text
        ):
            buffer.insert_text(buffer.suggestion.text)
            return True
        return False

    @kb.add("right")
    def _(event: KeyPressEvent) -> None:
        if not _accept_suggestion():
            buffer.cursor_right()

    @kb.add("tab")
    def _(event: KeyPressEvent) -> None:
        if not _accept_suggestion():
            buffer.insert_text("    ")

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        doc = event.app.current_buffer.document
        # \ at end of line → newline continuation
        if doc.cursor_position > 0 and doc.text[doc.cursor_position - 1] == "\\":
            before = doc.text[: doc.cursor_position - 1]
            after = doc.text[doc.cursor_position :]
            buffer.text = before + "\n" + after
            buffer.cursor_position = len(before) + 1
        else:
            event.app.exit()

    layout = Layout(Window(BufferControl(buffer=buffer)))
    output = create_output()
    output.enable_bracketed_paste()
    output.flush()
    output.flush = lambda: None  # suppress double-flush

    from prompt_toolkit.application import Application
    app = Application(
        layout=layout,
        output=output,
        key_bindings=merge_key_bindings([load_basic_bindings(), kb]),
        full_screen=False,
        erase_when_done=False,
        mouse_support=False,
    )

    def _create_panel() -> Group:
        cursor_line = buffer.document.cursor_position_row
        cursor_col = buffer.document.cursor_position_col
        text = Text()
        lines = buffer.text.split("\n")
        suggestion_text = None
        if cursor_line == len(lines) - 1 and cursor_col >= len(lines[-1]):
            if buffer.suggestion:
                suggestion_text = buffer.suggestion.text
        for i, line in enumerate(lines):
            text.append("> " if i == 0 else "  ")
            if i == cursor_line:
                rich_line = Text(line)
                if suggestion_text:
                    rich_line.append(Text(suggestion_text, style="dim italic"))
                    rich_line.stylize("reverse", cursor_col, cursor_col + 1)
                elif cursor_col >= len(line):
                    rich_line.append("█")
                else:
                    rich_line.stylize("reverse", cursor_col, cursor_col + 1)
                text.append(rich_line)
            else:
                text.append(line)
            if i < len(lines) - 1:
                text.append("\n")
        # Dynamic footer: show command list when typing a slash command
        from eaudev.commands.run.command_registry import registry
        footer = FOOTER_TEXT
        stripped = buffer.text.strip()
        if stripped == "/":
            footer = registry.render_help_table(show_header=False)
        elif stripped.startswith("/"):
            try:
                filtered = registry.render_help_table(show_header=False, command_filter=stripped)
                # Only show if the table has rows
                footer = filtered if filtered.row_count > 0 else FOOTER_TEXT
            except Exception as e:
                # Log but don't crash — footer is cosmetic
                from loguru import logger as _logger
                _logger.debug(f"Slash command footer render failed: {e}")
        return Group("", Panel(text, width=DEFAULT_PANEL_WIDTH), footer)

    with Live(_create_panel(), auto_refresh=False, transient=True) as live:
        def before_render(_: Application) -> None:
            live.update(_create_panel())
            live.refresh()
        app.before_render += before_render
        await app.run_async()

    lines = buffer.text.split("\n") if buffer.text else []
    if lines and buffer.text.strip():
        # Echo input back — transient=True on Live erases the panel on submit,
        # so we always need to reprint what the user typed.
        console.print()
        for j, line in enumerate(lines):
            prefix = "> " if j == 0 else "  "
            console.print(f"{prefix}{line}", highlight=False)

    # Print session context bar AFTER the Live exits — printing before causes
    # the transient erase to wipe it when the panel closes.
    # Show bar whenever session_context is provided — even tokens=0 is valid
    # (means context was just cleared/compacted).
    if session_context is not None:
        if isinstance(session_context, tuple) and len(session_context) == 2:
            tokens, ctx_limit = session_context
            console.print()
            display_token_usage(tokens, context_limit=ctx_limit)
        elif isinstance(session_context, int):
            console.print()
            display_token_usage(session_context)

    if prompt_session is not None and buffer.text:
        prompt_session.history.append_string(buffer.text)
    return buffer.text
