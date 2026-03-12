"""Memory loading from .agent.md files — faithful port of rovodev modules/memory.py."""

from __future__ import annotations

import re
from pathlib import Path

from eaudev.common.editor import open_file_in_editor

from rich.console import Console

from eaudev.constants import USER_MEMORY_FILE_NAMES, WORKSPACE_MEMORY_FILE_NAMES

console = Console()

USER_MEMORY_PROMPT_TEMPLATE = """\
Here are information or guidelines that you should consider when resolving requests:
{memory}\
"""

WORKSPACE_MEMORY_PROMPT_TEMPLATE = """\
Here are information or guidelines specific to this workspace that you should consider when resolving requests:
{memory}\
"""

MEMORY_INIT_PROMPT = """\
Your goal is to explore the current workspace and create or update the top-level .agent.md file with information that \
will be useful for developers working on this project.

You should aim to include:
- Key information about the project (purpose, language, frameworks, etc.)
- Important files and directories
- Best practices and conventions you observe

Don't call grep tools with overly broad patterns as it will be very slow and provide noisy output.

Format the file primarily as lists of instructions grouped by simple markdown headers.
In addition, if any of the following files are present in the root directory (not sub-directories), you should migrate \
their content to the .agent.md file:
- CLAUDE.md, CLAUDE.local.md
- codex.md, .codex/*.md
- .cursor/rules/*.mdc, .cursorrules.md, .cursorrules
- rules.md, .rules.md\
"""


def load_memories_from_file_system(path: str | Path | None = None) -> tuple[str | None, list[Path]]:
    """Walk from cwd up to home collecting workspace memory files; also check user home dir.

    Mirrors rovodev's load_memories_from_file_system exactly.

    Returns:
        (combined_prompt_string | None, list_of_found_paths)
    """
    path = Path.cwd() if path is None else Path(path).expanduser().absolute().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    if path.is_file():
        path = path.parent

    # ── Workspace memories (cwd → home) ───────────────────────────────────────
    memory_paths: list[str] = []
    workspace_memories: list[str] = []

    for parent_level, workspace_dir in enumerate([path] + list(path.parents)):
        for pattern in WORKSPACE_MEMORY_FILE_NAMES:
            matches = list(workspace_dir.glob(pattern))
            if matches:
                if path == workspace_dir:
                    prefix = "Instructions for the current directory (these take precedence if relevant):\n\n"
                else:
                    prefix = f"Instructions for ancestor directory {'../' * parent_level}:\n\n"
                memory_paths.extend(["../" * parent_level + m.name for m in matches])
                workspace_memories.extend(
                    [prefix + _sanitize(match.read_text()) for match in matches]
                )
        # Stop at home directory
        if workspace_dir == Path.home():
            break

    # ── User-level memory (~/.eaudev/.agent.md) ───────────────────────────────
    home_dir = Path.home()
    user_memories: list[str] = []
    user_paths: list[Path] = []
    for pattern in USER_MEMORY_FILE_NAMES:
        matches = list(home_dir.glob(pattern))
        if matches:
            user_paths.extend(matches)
            user_memories.extend([_sanitize(m.read_text()) for m in matches])

    # ── Assemble ──────────────────────────────────────────────────────────────
    all_paths: list[Path] = list(user_paths)
    sections: list[str] = []

    user_memories = [m for m in user_memories if m.strip()]
    if user_memories:
        sections.append(
            USER_MEMORY_PROMPT_TEMPLATE.format(
                memory="\n".join(
                    f"<user_instruction>\n{m}\n</user_instruction>" for m in user_memories
                )
            )
        )

    workspace_memories = [m for m in workspace_memories if m.strip()]
    if workspace_memories:
        # Paths are string-relative — convert to Path objects for display
        all_paths.extend(Path(p) for p in memory_paths)
        # Reverse so current directory's instructions come last (highest precedence)
        sections.append(
            WORKSPACE_MEMORY_PROMPT_TEMPLATE.format(
                memory="\n".join(
                    f"<workspace_instruction>\n{m}\n</workspace_instruction>"
                    for m in workspace_memories[::-1]
                )
            )
        )

    return ("\n".join(sections) if sections else None), all_paths


def get_memory_instructions(log_paths: bool = False) -> str:
    """Return memory as a system prompt string, with the trailing .agent.md hint.

    This is the canonical call used by the run loop — mirrors rovodev's get_memory_instructions().
    """
    instructions = ""
    memory, memory_paths = load_memories_from_file_system(Path.cwd())
    if memory:
        if log_paths:
            console.print(
                f"[bright_black]Loaded memory from {', '.join(str(p) for p in memory_paths)}[/bright_black]"
            )
        instructions = memory

    instructions += """

Location-specific best practices, tips, and patterns may be found throughout the current workspace in .agent.md \
files. Before making any changes in a subdirectory, please read the contents of its .agent.md if present.\
"""
    return instructions.strip()


def handle_memory_note(note: str) -> str:
    """Add or remove a quick note from .agent.md in the current directory.

    Prefix the note with '!' to remove it.
    Returns updated memory instructions string.
    
    TODO: cache memory file reads — currently reads twice when called with get_memory_instructions().
    """
    # TODO: cache memory file reads — currently reads twice when called with get_memory_instructions().
    note = note.lstrip("#").strip()
    memory_path = Path(".agent.md")
    memory_content = _sanitize(memory_path.read_text()) if memory_path.exists() else ""

    existing_notes_section = re.search(
        r"(?:^|\n)# Workspace notes\s*\n(?:- [^\n]+(?:\n|$))*", memory_content
    )

    if note.startswith("!"):
        note = note[1:].strip()
        if note not in memory_content:
            console.print("[bright_black]Note not found in memory file.[/bright_black]")
            return get_memory_instructions()
        memory_content = "\n".join(
            line for line in memory_content.splitlines() if note not in line
        )
        verb = "removed from"
    else:
        if not existing_notes_section:
            memory_content = memory_content + f"\n\n# Workspace notes\n\n- {note}\n"
        else:
            existing = existing_notes_section.group()
            memory_content = memory_content.replace(
                existing, existing.rstrip() + f"\n- {note}\n"
            )
        verb = "added to"

    memory_path.write_text(memory_content.strip() + "\n")
    console.print(f"[bright_black]Memory note {verb} .agent.md[/bright_black]")
    return get_memory_instructions()


def handle_memory_command(command: str | None) -> str | None:
    """Handle /memory [user|init] — open editor or return init prompt.
    
    Returns:
        str: MEMORY_INIT_PROMPT if command == 'init' — inject into agent conversation.
        None: for all editor-open branches — no prompt action needed.
    """
    if command is None:
        return _open_in_editor(Path.cwd() / ".agent.md")
    command = command.strip().lower()
    if command == "user":
        mem_path = Path.home() / ".eaudev" / ".agent.md"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        return _open_in_editor(mem_path)
    if command == "init":
        return MEMORY_INIT_PROMPT
    # Fallback: workspace
    return _open_in_editor(Path.cwd() / ".agent.md")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_in_editor(path: Path) -> None:
    """Open file in the best available editor, creating it if needed.
    
    Returns None explicitly — callers must not expect a string.
    """
    open_file_in_editor(str(path), create_if_missing=True)
    return None


def _sanitize(content: str) -> str:
    """Strip invisible tag characters and neutralise XML tag injection attempts.
    
    Removes Unicode tag/invisible characters that could inject hidden instructions.
    Also strips attempts to close/reopen the wrapping XML tags used in prompt templates.
    """
    # Remove Unicode tag/invisible characters
    content = re.sub(r"[\U000E0001-\U000E007F\U000E0100-\U000E01EF\u2061-\u2065]", "", content)
    # Neutralise attempts to close/reopen the wrapping XML tags used in prompt templates
    # Replace < and > only when they form patterns that could escape the wrapper tags
    content = re.sub(r"</?(workspace_instruction|user_instruction|system)\s*/?>", "", content, flags=re.IGNORECASE)
    return content
