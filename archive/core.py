#!/usr/bin/env python3
"""
EauDev v0.1.0 — Local AI coding agent (PROTOTYPE — DEPRECATED)
Powered by Qwen3-Coder via llama.cpp

⚠  This file is a single-file prototype and is NOT the active codebase.
   The real EauDev application lives in the eaudev/ package and is
   installed via `pip install -e .` (see pyproject.toml).
   Run it with: eaudev  (or: python -m eaudev)

   This file is kept for historical reference only. It is not imported
   anywhere and will not receive further updates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
import datetime
import urllib.request
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ENDPOINT    = "http://localhost:8080/v1/chat/completions"
MODEL       = "qwen3-coder"
HOME        = str(Path.home())  # was hardcoded to /Users/eaumac — now dynamic
WORKDIR     = str(Path.cwd())   # was hardcoded to a machine-specific path — now dynamic
SESSION_DIR = Path.home() / ".eaudev" / "sessions"
CONFIG_PATH = Path.home() / ".eaudev" / "config.json"
VERSION     = "0.1.0"

# Permission scopes: once / session / always
# Permission states: allow / deny / ask
DEFAULT_TOOL_PERMISSION = "ask"
DEFAULT_BASH_PERMISSION = "ask"

SYSTEM = f"""You are EauDev, a local AI coding agent running on the user's Mac (Apple M1 Max, 64GB).

Environment:
  Working directory : {WORKDIR}
  Home directory    : {HOME}
  OS                : macOS Darwin arm64
  Shell             : /bin/zsh

You are helping build Cluster — a modular local AI system comprising:
  - Memory MCP      : unified memory (observation buffer, facts, FTS5, knowledge graph)
  - Specialist      : voice-based frontline agent (VAD → ASR → LLM → TTS)
  - EauDev          : this coding agent (you)
  - Archive MCP     : document retrieval and knowledge graph
  - KG Constraint   : canonical constraint validation layer

Available tools — invoke by emitting a <tool>...</tool> block containing ONLY valid JSON.
The content between <tool> and </tool> MUST be valid JSON. Never emit a bare tool name.
One tool call per response. No other text inside the tags.

  read_file        : read a file
    {{"name":"read_file","path":"/abs/path"}}

  write_file       : write/create a file (creates parent dirs)
    {{"name":"write_file","path":"/abs/path","content":"..."}}

  run_bash         : run a shell command (cwd = working directory)
    {{"name":"run_bash","command":"..."}}

  list_directory   : list directory contents
    {{"name":"list_directory","path":"/abs/path"}}

Workflow rules:
  1. For any multi-step task, state your plan in 1-3 lines BEFORE the first tool call
  2. Always use absolute paths under {HOME}/
  3. After completing a task, give a brief summary of what changed
  4. On errors, explain and suggest the fix
  5. Be concise — no filler, no apologies"""

# ── Memory loading ─────────────────────────────────────────────────────────────

def load_memory() -> tuple[str, list[str]]:
    """Load .agent.md memory files from user-level and workspace (cwd → home).

    Returns (combined_text, list_of_paths_found).
    """
    home = Path.home()
    sections: list[str] = []
    found_paths: list[str] = []

    # 1. User-level memory: ~/.eaudev/.agent.md
    user_mem = home / ".eaudev" / ".agent.md"
    if user_mem.exists():
        try:
            content = user_mem.read_text(encoding="utf-8").strip()
            if content:
                sections.append(
                    f"Here are information or guidelines that you should consider"
                    f" when resolving requests:\n{content}"
                )
                found_paths.append(str(user_mem))
        except Exception:
            pass

    # 2. Workspace memory: walk from cwd up to ~ looking for .agent.md
    workspace_contents: list[str] = []
    workspace_paths: list[str] = []
    current = Path.cwd().resolve()
    home_resolved = home.resolve()

    visited: set[Path] = set()
    while True:
        if current in visited:
            break
        visited.add(current)

        candidate = current / ".agent.md"
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    workspace_contents.append(content)
                    workspace_paths.append(str(candidate))
            except Exception:
                pass

        if current == home_resolved or current.parent == current:
            break
        current = current.parent

    if workspace_contents:
        # Innermost directory first (already in that order), join with separator
        combined = "\n\n---\n\n".join(reversed(workspace_contents))
        sections.append(
            f"Here are information or guidelines specific to this workspace"
            f" that you should consider when resolving requests:\n{combined}"
        )
        found_paths.extend(reversed(workspace_paths))

    return "\n\n".join(sections), found_paths


# ── Permission manager ─────────────────────────────────────────────────────────

class PermissionManager:
    """Three-scope permission model: once / session / always."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._session: dict[str, str] = {}   # tool_key -> allow/deny
        self._persistent: dict[str, str] = {}
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                self._persistent = data.get("tool_permissions", {})
            except Exception:
                pass

    def _save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tool_permissions": self._persistent}
        self.config_path.write_text(json.dumps(data, indent=2))

    def _tool_key(self, tool_name: str, detail: str = "") -> str:
        if detail:
            # Use first word of bash command as pattern key
            base = detail.split()[0]
            return f"bash:{base}.*"
        return tool_name

    def check(self, tool_name: str, detail: str = "") -> str:
        """Return 'allow', 'deny', or 'ask'."""
        key = self._tool_key(tool_name, detail)
        # Session overrides persistent
        if key in self._session:
            return self._session[key]
        if key in self._persistent:
            return self._persistent[key]
        # Also check exact match for bash
        if detail and detail in self._session:
            return self._session[detail]
        if detail and detail in self._persistent:
            return self._persistent[detail]
        return "ask"

    def ask(self, tool_name: str, detail: str = "") -> tuple[str, str]:
        """Prompt user. Returns (permission, scope)."""
        if tool_name == "run_bash":
            print(f"\n  ⚡ bash: {detail}")
        elif tool_name == "write_file":
            print(f"\n  ✏️  write: {detail}")
        else:
            print(f"\n  🔧 {tool_name}: {detail}")

        print("  [1] Allow once  [2] Allow session  [3] Allow always  [4] Deny once  [5] Deny session  [6] Deny always")
        try:
            choice = input("  Choice [1]: ").strip() or "1"
        except (KeyboardInterrupt, EOFError):
            return "deny", "once"

        mapping = {
            "1": ("allow", "once"),
            "2": ("allow", "session"),
            "3": ("allow", "always"),
            "4": ("deny", "once"),
            "5": ("deny", "session"),
            "6": ("deny", "always"),
        }
        permission, scope = mapping.get(choice, ("allow", "once"))

        key = self._tool_key(tool_name, detail)
        if scope == "session":
            self._session[key] = permission
        elif scope == "always":
            self._persistent[key] = permission
            self._save_config()

        return permission, scope

    def resolve(self, tool_name: str, detail: str = "") -> str:
        """Return final 'allow' or 'deny', prompting if needed."""
        # Read-only tools never need permission
        if tool_name in ("read_file", "list_directory"):
            return "allow"
        decision = self.check(tool_name, detail)
        if decision == "ask":
            permission, _ = self.ask(tool_name, detail)
            return permission
        return decision


# ── Session persistence ────────────────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str | None = None, workspace: str = WORKDIR):
        self.session_id   = session_id or str(uuid.uuid4())
        self.workspace    = workspace
        self.created      = datetime.datetime.now().isoformat()
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]
        self.path         = SESSION_DIR / self.session_id
        self._title       = "Untitled"

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        ctx = {
            "session_id": self.session_id,
            "workspace":  self.workspace,
            "created":    self.created,
            "title":      self._title,
            "messages":   self.messages,
        }
        (self.path / "session.json").write_text(json.dumps(ctx, indent=2))

    @classmethod
    def load(cls, session_id: str) -> "Session":
        path = SESSION_DIR / session_id / "session.json"
        data = json.loads(path.read_text())
        s = cls(session_id=data["session_id"], workspace=data.get("workspace", WORKDIR))
        s.created  = data["created"]
        s._title   = data.get("title", "Untitled")
        s.messages = data["messages"]
        return s

    @classmethod
    def list_sessions(cls) -> list[dict]:
        sessions = []
        if not SESSION_DIR.exists():
            return sessions
        for d in sorted(SESSION_DIR.iterdir(), reverse=True):
            f = d / "session.json"
            if f.exists():
                try:
                    data = json.loads(f.read_text())
                    sessions.append({
                        "id":      data["session_id"],
                        "title":   data.get("title", "Untitled"),
                        "created": data.get("created", ""),
                        "msgs":    len(data.get("messages", [])),
                    })
                except Exception:
                    pass
        return sessions

    def set_title_from_first_message(self):
        for m in self.messages:
            if m["role"] == "user":
                self._title = m["content"][:60].replace("\n", " ")
                break


# ── Tool implementations ───────────────────────────────────────────────────────

def read_file(path: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        content = p.read_text(encoding="utf-8")
        lines = content.count("\n") + 1
        return f"[read_file: {path} — {lines} lines]\n{content}"
    except Exception as e:
        return f"[read_file error: {e}]"

def write_file(path: str, content: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[write_file: {len(content)} bytes → {path}]"
    except Exception as e:
        return f"[write_file error: {e}]"

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=60,
            cwd=os.path.expanduser(WORKDIR),
            env={**os.environ, "TERM": "xterm-256color"},
        )
        output = (result.stdout + result.stderr).strip()
        return f"[run_bash exit={result.returncode}: {command}]\n{output or '(no output)'}"
    except subprocess.TimeoutExpired:
        return f"[run_bash error: timed out after 60s — {command}]"
    except Exception as e:
        return f"[run_bash error: {e}]"

def list_directory(path: str) -> str:
    try:
        p = Path(os.path.expanduser(path))
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"  📁 {e.name}/")
            else:
                size = e.stat().st_size
                lines.append(f"  📄 {e.name}  ({size:,} bytes)")
        return f"[list_directory: {path} — {len(entries)} entries]\n" + "\n".join(lines)
    except Exception as e:
        return f"[list_directory error: {e}]"

def dispatch_tool(tool_call: dict, perms: PermissionManager) -> str:
    name = tool_call.get("name", "")

    if name == "read_file":
        return read_file(tool_call.get("path", ""))

    elif name == "list_directory":
        return list_directory(tool_call.get("path", ""))

    elif name == "write_file":
        path = tool_call.get("path", "")
        decision = perms.resolve("write_file", path)
        if decision == "deny":
            return "[write_file: denied by user. Do not attempt this operation again this session.]"
        return write_file(path, tool_call.get("content", ""))

    elif name == "run_bash":
        command = tool_call.get("command", "")
        decision = perms.resolve("run_bash", command)
        if decision == "deny":
            return "[run_bash: denied by user. Do not attempt this command again this session.]"
        return run_bash(command)

    else:
        return f"[unknown tool: {name}]"


# ── Tool call parsing ──────────────────────────────────────────────────────────

def extract_tool_call(text: str) -> tuple[dict | None, str]:
    start = text.find("<tool>")
    end   = text.find("</tool")
    if start == -1 or end == -1:
        return None, text
    json_str = text[start + 6:end].strip()
    try:
        return json.loads(json_str), text[:start].strip()
    except json.JSONDecodeError:
        # Warn rather than silently drop — help the user diagnose model misbehaviour
        raw = json_str[:120] + ("..." if len(json_str) > 120 else "")
        print(f"\n[EauDev: model emitted malformed tool call — skipping: {raw}]\n")
        return None, text


# ── Streaming LLM call ─────────────────────────────────────────────────────────

def chat_stream(messages: list[dict]) -> str:
    payload = json.dumps({
        "model":       MODEL,
        "messages":    messages,
        "stream":      True,
        "temperature": 0.7,
        "max_tokens":  4096,
    }).encode()

    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json"}
    )

    chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        chunks.append(delta)
                except (json.JSONDecodeError, KeyError):
                    continue
    except urllib.error.URLError as e:
        print(f"\n[connection error: {e}]")
    print()
    return "".join(chunks)


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(session: Session, perms: PermissionManager):
    """Run the agentic loop until no more tool calls."""
    max_iterations = 20
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        response = chat_stream(session.messages)
        session.messages.append({"role": "assistant", "content": response})

        tool_call, _ = extract_tool_call(response)
        if tool_call is None:
            break

        tool_name = tool_call.get("name", "unknown")
        print(f"\n  ╔═ tool: {tool_name}", flush=True)
        result = dispatch_tool(tool_call, perms)
        print(f"  ╚═ {result[:200]}{'...' if len(result) > 200 else ''}\n", flush=True)

        session.messages.append({"role": "user", "content": result})

    if iterations >= max_iterations:
        print("\n[EauDev: reached max tool iterations — stopping]\n")


# ── Slash commands ─────────────────────────────────────────────────────────────

def handle_slash_command(cmd: str, session: Session, perms: PermissionManager) -> bool:
    """Handle /commands. Returns True if handled."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/help":
        print("""
  EauDev commands:
    /help              — show this help
    /sessions          — list saved sessions
    /new               — start a new session
    /save              — save current session
    /clear             — clear message history (keep system prompt)
    /workdir           — show current working directory
    /permissions       — show permission settings
    /memory            — edit workspace .agent.md in $EDITOR
    /memory user       — edit ~/.eaudev/.agent.md in $EDITOR
    /memory init       — prompt agent to create/update .agent.md
    /exit              — quit
""")
        return True

    elif command == "/sessions":
        sessions = Session.list_sessions()
        if not sessions:
            print("  No saved sessions.")
        else:
            print(f"\n  {'ID':36}  {'Title':40}  Messages")
            print("  " + "─" * 85)
            for s in sessions[:10]:
                print(f"  {s['id']:36}  {s['title'][:40]:40}  {s['msgs']}")
        print()
        return True

    elif command == "/new":
        print("  Starting new session...")
        session.__init__()
        return True

    elif command == "/save":
        session.set_title_from_first_message()
        session.save()
        print(f"  Session saved: {session.session_id}")
        return True

    elif command == "/clear":
        session.messages = [{"role": "system", "content": SYSTEM}]
        print("  Context cleared.")
        return True

    elif command == "/workdir":
        print(f"  Working directory: {WORKDIR}")
        return True

    elif command == "/permissions":
        print(f"  Persistent: {json.dumps(perms._persistent, indent=4)}")
        print(f"  Session:    {json.dumps(perms._session, indent=4)}")
        return True

    elif command == "/memory":
        subcommand = parts[1].lower() if len(parts) > 1 else ""
        editor = os.environ.get("EDITOR", "nano")

        if subcommand == "user":
            mem_path = Path.home() / ".eaudev" / ".agent.md"
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            mem_path.touch(exist_ok=True)
            subprocess.call([editor, str(mem_path)])
            print(f"  Saved: {mem_path}")

        elif subcommand == "init":
            return (
                "Explore the current workspace and create or update .agent.md with: "
                "project purpose, key files and directories, language/frameworks, "
                "conventions you observe. Format as markdown lists under simple headers."
            )

        else:
            # Default: workspace .agent.md in cwd
            mem_path = Path.cwd() / ".agent.md"
            mem_path.touch(exist_ok=True)
            subprocess.call([editor, str(mem_path)])
            print(f"  Saved: {mem_path}")

        return True

    elif command in ("/exit", "/quit"):
        session.set_title_from_first_message()
        session.save()
        print("  Session saved. Goodbye.")
        sys.exit(0)

    return False


# ── Banner ─────────────────────────────────────────────────────────────────────

def print_banner():
    print(f"""
  ███████  █████  ██    ██  ██████  ███████ ██    ██
  ██      ██   ██ ██    ██ ██    ██ ██      ██    ██
  █████   ███████ ██    ██ ██    ██ █████   ██    ██
  ██      ██   ██ ██    ██ ██    ██ ██       ██  ██
  ███████ ██   ██  ██████   ██████  ███████   ████   v{VERSION}

  Local AI coding agent — {MODEL} @ {ENDPOINT}
  Working directory: {WORKDIR}
  Type /help for commands, Ctrl+C to exit
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def read_user_input() -> str:
    """Read one user turn. Supports multiline mode triggered by triple-quotes."""
    line = input("You: ").strip()

    # Enter multiline mode if the line is exactly """
    if line == '"""':
        print('  (multiline mode — type """ on a line by itself to finish)')
        lines: list[str] = []
        while True:
            try:
                l = input()
            except EOFError:
                break
            if l.rstrip() == '"""':
                break
            lines.append(l)
        return "\n".join(lines)

    return line


def main():
    # ── Argument parsing ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="eaudev",
        description="EauDev — local AI coding agent",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Resume the most recent saved session",
    )
    args = parser.parse_args()

    print_banner()

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Memory loading ─────────────────────────────────────────────────────────
    memory_text, memory_paths = load_memory()
    if memory_paths:
        print(f"  Loaded memory from: {', '.join(memory_paths)}\n")

    # ── Session setup ──────────────────────────────────────────────────────────
    if args.restore:
        sessions = Session.list_sessions()
        if sessions:
            most_recent = sorted(sessions, key=lambda s: s["created"], reverse=True)[0]
            session = Session.load(most_recent["id"])
            print(f"  Restoring session: {session._title} ({len(session.messages)} messages)\n")
        else:
            print("  No saved sessions found — starting new session.\n")
            session = Session()
    else:
        session = Session()

    # Append memory to the system prompt if any was found
    if memory_text:
        session.messages[0]["content"] = SYSTEM + "\n\n" + memory_text

    perms = PermissionManager(CONFIG_PATH)

    while True:
        try:
            user_input = read_user_input().strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                result = handle_slash_command(user_input, session, perms)
                # /memory init returns the prompt string instead of True
                if isinstance(result, str):
                    user_input = result
                elif result:
                    continue

            session.messages.append({"role": "user", "content": user_input})
            print("EauDev: ", end="", flush=True)
            run_agent(session, perms)

            # Auto-save after each turn
            session.set_title_from_first_message()
            session.save()

        except KeyboardInterrupt:
            print("\n")
            session.set_title_from_first_message()
            session.save()
            print("  Session saved. Goodbye.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[error: {e}]")


if __name__ == "__main__":
    main()
