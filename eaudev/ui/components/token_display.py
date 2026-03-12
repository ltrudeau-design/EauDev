"""Token usage display components for the EauDev CLI."""

from __future__ import annotations

from rich.console import Console

console = Console()

# Default context limit for Qwen3-Coder on llama.cpp (can be overridden by config)
DEFAULT_CONTEXT_LIMIT = 32768


def format_tokens(tokens: int) -> str:
    """Format token count in human-readable form (K/M)."""
    if tokens >= 1_000_000:
        formatted = f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        formatted = f"{tokens / 1_000:.1f}K"
    else:
        formatted = str(tokens)
    # Remove .0 from whole numbers
    if formatted.endswith(".0K"):
        formatted = formatted[:-3] + "K"
    elif formatted.endswith(".0M"):
        formatted = formatted[:-3] + "M"
    return formatted


def display_token_usage(
    tokens: int,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> None:
    """Display a session context usage bar.

    Shows a ▮ progress bar with token counts. Prints a /prune tip when
    context usage exceeds 50%. The daily cloud usage bar from rovodev is
    intentionally omitted — EauDev uses local inference.

    Args:
        tokens: Current session token count (prompt + completion).
        context_limit: Maximum context window size for the model.
    """
    proportion = min(tokens / context_limit, 1.0)
    width = 10
    filled = int(proportion * width)
    unfilled = width - filled

    bar = (
        "[dim]Session context: [/dim]"
        "[bold blue]" + "▮" * filled + "[/bold blue]"
        "[dim]" + "▮" * unfilled + "[/dim]"
    )
    counts = f"[reset][dim]{format_tokens(tokens)}/{format_tokens(context_limit)}[/dim][/reset]"

    if proportion > 0.5:
        console.print(
            bar,
            counts,
            "[dim]| [bold]Tip:[/bold] use [reset]/prune[/reset] to reduce context size[/dim]",
        )
    else:
        console.print(bar, counts)
