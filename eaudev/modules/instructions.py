"""Instruction handling for EauDev CLI.

Loads pre-defined instruction templates from YAML files and presents them
via an arrow-key menu. Ported from rovodev with all Atlassian-specific
built-ins removed. Local instructions only.

Config file: .eaudev/instructions.yml (workspace or home dir)
Format:
    instructions:
      - name: my-instruction
        description: What it does
        content_file: my_instruction.md
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from eaudev.ui.components.user_menu_panel import Choice, user_menu_panel_sync

console = Console()

INSTRUCTIONS_YAML = "instructions.yml"
EAUDEV_DIR = ".eaudev"

INSTRUCTIONS_MESSAGE = (
    "[bright_black]Speed up your workflow with pre-created prompts.\n"
    "Edit or create instructions in [bold].eaudev/instructions.yml[/bold].[/bright_black]\n\n"
    "Select an instruction to run:"
)


class Instruction(BaseModel):
    name: str
    description: str | None = None
    content_file: str
    content: str | None = None


class InstructionConfig(BaseModel):
    instructions: list[Instruction] = []


def _get_config_locations() -> list[Path]:
    """Return candidate paths for instructions.yml in priority order."""
    locations: list[Path] = []

    # Walk up from cwd to find a .git root
    current = Path.cwd()
    while current.parent != current:
        if (current / ".git").exists():
            locations.append(current / EAUDEV_DIR / INSTRUCTIONS_YAML)
            break
        current = current.parent

    # cwd .eaudev
    locations.append(Path.cwd() / EAUDEV_DIR / INSTRUCTIONS_YAML)

    # User home .eaudev
    locations.append(Path.home() / EAUDEV_DIR / INSTRUCTIONS_YAML)

    return locations


def _load_content(config_path: Path, instruction: Instruction) -> str | None:
    """Resolve and load a content_file relative to the config YAML."""
    candidates = [
        config_path.parent / instruction.content_file,
        config_path.parent.parent / instruction.content_file,
        Path(instruction.content_file),
    ]
    for p in candidates:
        if p.exists():
            return p.read_text()
    return None


def _load_file(path: Path) -> InstructionConfig | None:
    """Load and validate a single instructions.yml file."""
    try:
        raw = yaml.safe_load(path.read_text())
        if not raw:
            return None
        cfg = InstructionConfig.model_validate(raw)
        for inst in cfg.instructions:
            inst.content = _load_content(path, inst)
            if not inst.content:
                logger.warning(f"Instruction content file not found: {inst.content_file}")
        cfg.instructions = [i for i in cfg.instructions if i.content]
        return cfg
    except Exception as e:
        logger.warning(f"Failed to load instructions from {path}: {e}")
        return None


def load_instruction_config() -> InstructionConfig:
    """Merge instructions from all found config locations."""
    merged = InstructionConfig()
    seen: set[str] = set()
    for loc in _get_config_locations():
        if not loc.exists():
            continue
        cfg = _load_file(loc)
        if cfg:
            for inst in cfg.instructions:
                if inst.name not in seen:
                    merged.instructions.append(inst)
                    seen.add(inst.name)
    return merged


def handle_instructions_command(args: str | None = None) -> str | None:
    """Handle the /instructions [name] [extra] command.

    Args:
        args: Optional instruction name and/or extra context string.

    Returns:
        The instruction content string to queue as the next user message,
        or None if cancelled / nothing selected.
    """
    config = load_instruction_config()

    if not config.instructions:
        console.print(
            "  [dim]No instructions found. Create [bold].eaudev/instructions.yml[/bold] "
            "in your project or home directory.[/dim]"
        )
        return None

    # If a name was provided, try to find it directly
    name: str | None = None
    extra: str | None = None
    if args:
        parts = args.strip().split(None, 1)
        name = parts[0] if parts else None
        extra = parts[1] if len(parts) > 1 else None

    if name:
        match = next((i for i in config.instructions if i.name == name), None)
        if match and match.content:
            content = match.content
            if extra:
                content = f"{content}\n\n{extra}"
            return content
        console.print(f"  [red]Instruction '{name}' not found.[/red]")
        return None

    # Interactive arrow-key menu
    choices: list[Choice] = [
        Choice(
            name=f"{inst.name}" + (f"  [dim]{inst.description}[/dim]" if inst.description else ""),
            value=inst,
        )
        for inst in config.instructions
    ]

    selected: Instruction | None = user_menu_panel_sync(
        choices=choices,
        message=INSTRUCTIONS_MESSAGE,
        title="Instructions",
        action_name="Run",
        escape_return_value=None,
    )

    if selected is None or not selected.content:
        return None

    content = selected.content
    if extra:
        content = f"{content}\n\n{extra}"
    return content
