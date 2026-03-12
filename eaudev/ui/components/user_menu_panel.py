"""Arrow-key selection menu — rendered purely via prompt_toolkit (no Rich Live conflict)."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, TypedDict

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output import create_output

from eaudev.constants import DEFAULT_PANEL_WIDTH

DEFAULT_MENU_FOOTER = "↑↓: Navigate | Enter ⏎: {action} | Esc: Cancel"


class Choice(TypedDict):
    name: str
    value: Any


def _render_menu(
    choices: list[Choice],
    index: int,
    message: str | None,
    title: str | None,
    footer: str,
    width: int,
) -> str:
    """Render the menu as an ANSI string using box-drawing chars."""
    W = min(width, DEFAULT_PANEL_WIDTH)
    inner = W - 2  # inside borders

    lines: list[str] = []

    # Top border with title
    if title:
        t = f" {title} "
        pad = W - 2 - len(t)
        lines.append(f"\033[33m╭─{t}{'─' * pad}╮\033[0m")
    else:
        lines.append(f"\033[33m╭{'─' * (W - 2)}╮\033[0m")

    # Message
    if message:
        lines.append(f"\033[33m│\033[0m \033[1m{message[:inner - 2]}\033[0m{' ' * max(0, inner - 2 - len(message))} \033[33m│\033[0m")
        lines.append(f"\033[33m│\033[0m{' ' * inner}\033[33m│\033[0m")

    # Choices
    for i, choice in enumerate(choices):
        name = choice["name"]
        if i == index:
            marker = "\033[34;1m> \033[0m\033[34;1m"
            reset = "\033[0m"
            padded = (name[:inner - 4] if len(name) > inner - 4 else name).ljust(inner - 4)
            row = f"\033[33m│\033[0m {marker}{padded}{reset}  \033[33m│\033[0m"
        else:
            padded = (name[:inner - 4] if len(name) > inner - 4 else name).ljust(inner - 4)
            row = f"\033[33m│\033[0m   {padded}  \033[33m│\033[0m"
        lines.append(row)

    # Bottom border
    lines.append(f"\033[33m╰{'─' * (W - 2)}╯\033[0m")

    # Footer
    lines.append(f"\033[90m{footer[:W]}\033[0m")

    return "\n".join(lines)


async def user_menu_panel(
    choices: list[Choice],
    message: str | None = None,
    selection: int | None = None,
    title: str | None = None,
    border_color: str = "dark_orange",   # kept for API compat, unused
    title_color: str = "orange1",         # kept for API compat, unused
    header: str | None = None,
    footer: str = DEFAULT_MENU_FOOTER,
    escape_return_value: Any = None,
    action_name: str = "Select",
) -> Any:
    """Display an arrow-key menu rendered purely via prompt_toolkit."""
    footer_str = footer.format(action=action_name)
    index = selection or 0
    info: dict[str, Any] = {"index": index, "cancelled": False}
    width = DEFAULT_PANEL_WIDTH

    kb = KeyBindings()

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        info["cancelled"] = True
        event.app.exit()

    @kb.add("escape")
    def _(event: KeyPressEvent) -> None:
        info["cancelled"] = True
        event.app.exit()

    @kb.add("down")
    def _(event: KeyPressEvent) -> None:
        info["index"] = (info["index"] + 1) % len(choices)
        event.app.invalidate()

    @kb.add("up")
    def _(event: KeyPressEvent) -> None:
        info["index"] = (info["index"] - 1) % len(choices)
        event.app.invalidate()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        event.app.exit()

    def get_text():
        return ANSI(_render_menu(choices, info["index"], message, title, footer_str, width))

    layout = Layout(
        Window(
            FormattedTextControl(get_text, focusable=True),
            dont_extend_height=True,
        )
    )

    app = Application(
        output=create_output(),
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        paste_mode=True,
    )

    await app.run_async()

    if info["cancelled"]:
        return escape_return_value
    return choices[info["index"]]["value"]


def user_menu_panel_sync(
    choices: list[Choice],
    message: str | None = None,
    selection: int | None = None,
    title: str | None = None,
    border_color: str = "dark_orange",
    title_color: str = "orange1",
    header: str | None = None,
    footer: str = DEFAULT_MENU_FOOTER,
    escape_return_value: Any = None,
    action_name: str = "Select",
) -> Any:
    """Synchronous wrapper — safe to call from within an existing asyncio.run()."""
    import threading

    result_holder: list[Any] = [None]
    exc_holder: list[BaseException | None] = [None]

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(
                user_menu_panel(
                    choices=choices,
                    message=message,
                    selection=selection,
                    title=title,
                    border_color=border_color,
                    title_color=title_color,
                    header=header,
                    footer=footer,
                    escape_return_value=escape_return_value,
                    action_name=action_name,
                )
            )
        except Exception as e:
            exc_holder[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    # Brief pause to allow the terminal to redraw after prompt_toolkit app exits.
    # Without this, the next console.print() can overwrite the last line of the menu.
    time.sleep(0.05)
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]
