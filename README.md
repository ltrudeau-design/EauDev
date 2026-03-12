# EauDev

Local AI coding agent — powered by Qwen3-Coder on llama.cpp.  
A local-first port of Atlassian's Rovo Dev CLI with all cloud dependencies removed.

**Version:** 0.1.0 — Feature Complete  
**Status:** v1.0 — awaiting real-session validation

---

## Quick Start

```bash
cd /path/to/EauDev
pip install -e .
eaudev                    # start fresh
eaudev --restore          # resume last session
eaudev --workdir ~/path   # set working directory
eaudev --verbose          # verbose logging to stderr
```

**Prerequisites:** llama.cpp server running at `http://localhost:8080` with a Qwen3-Coder model loaded.  
For VoiceIO: run via `START_EAUDEV.command` to use the pytorch env with faster-whisper + sounddevice.

---

## Architecture

```
eaudev/
├── cli.py                        # main() → run()
├── constants.py                  # all paths, endpoint, defaults
├── common/
│   ├── config.py / config_model.py   # YAML config + pydantic models
│   ├── editor.py                 # Cursor/VSCode/JetBrains/nano detection
│   └── exceptions.py             # EauDevError, RequestTooLargeError, ServerError
├── modules/
│   ├── memory.py                 # .agent.md filesystem loader
│   ├── memory_store.py           # 5-layer persistent memory (SQLite)
│   ├── mcp_client.py             # MCPClientManager — JSON-RPC 2.0 over stdio
│   ├── sessions.py               # Session persistence (JSON)
│   ├── tool_permissions.py       # ToolPermissionManager — allowlist/denylist/confirm
│   ├── model_registry.py         # models.yml loader + live model switching
│   ├── voice_io.py               # VoiceIO — ASR (faster-whisper) + TTS (piper)
│   ├── instructions.py           # Instruction YAML template loader
│   └── logging.py                # loguru setup
├── ui/
│   ├── components/
│   │   ├── session_menu_panel.py # arrow-key session browser
│   │   ├── user_menu_panel.py    # generic arrow-key menu
│   │   ├── user_input_panel.py   # Rich Live prompt with \ continuation
│   │   └── token_display.py      # ▮▮▮▮▮▮ context usage bar
│   └── prompt_session.py         # FilteredFileHistory + PromptSession
└── commands/run/
    ├── command.py                 # main loop, agent loop, all slash commands
    └── command_registry.py        # CommandRegistry + /help table
```

---

## Native Tools

The agent has 7 native tools available via `<tool>{...}</tool>` XML syntax:

| Tool | Permission | Description |
|---|---|---|
| `read_file` | allow | Read file contents (truncates at 16K chars with notice) |
| `list_directory` | allow | List directory with sizes |
| `write_file` | ask | Write/overwrite file — shows unified diff on change |
| `create_file` | ask | Create new file — errors if already exists |
| `run_bash` | ask | Shell command with streaming output, 60s timeout |
| `move_file` | ask | Rename or move a file |
| `delete_file` | ask | Delete a single file (not directories) |

MCP tools are also available as `{server}__{tool}` — e.g. `archive__search`.  
See `~/.eaudev/mcp.json` to register MCP servers.

---

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Full command list with descriptions |
| `/sessions` | Arrow-key session browser — switch, fork, delete |
| `/clear` | Wipe message history, reinject system prompt |
| `/prune` | Strip tool results from context, shows token delta |
| `/compact` | LLM-assisted summarise + truncate to minimal skeleton |
| `/shadow <q>` | Ask the model using current context — response not saved |
| `/memory` | Edit workspace `.agent.md` in `$EDITOR` |
| `/memory user` | Edit `~/.eaudev/.agent.md` |
| `/memory init` | Ask agent to generate/update `.agent.md` for this workspace |
| `/memory stats` | Live diagnostics across all 5 memory layers |
| `/instructions` | Run saved instruction templates from YAML |
| `/config` | Open `~/.eaudev/config.yml` in `$EDITOR` |
| `/server` | llama.cpp server status — model, n_ctx, slots, KV usage |
| `/model` | Live model switcher — stops/starts llama-server |
| `# <note>` | Append quick note to workspace `.agent.md` |
| `#! <note>` | Remove matching note from workspace `.agent.md` |
| `/hooks` | Manage registered hooks — list, add, remove, enable/disable, test |
| `/exit` `/quit` `/q` | Save session and exit |

---

## Hooks

EauDev supports a hooks system — shell scripts or Python programs that run automatically at specific points in the tool execution lifecycle. Hooks are completely decoupled from the agent — they run as separate processes communicating via stdin/stdout/exit codes.

### Lifecycle Events

| Event | When | Can Block? |
|-------|------|------------|
| `SessionStart` | Once at session open | No |
| `PreToolUse` | Before every tool call | Yes (exit 2) |
| `PostToolUse` | After every tool call | No |

### Protocol

Hooks receive a JSON blob on **stdin**:

```json
{
  "tool_name": "run_bash",
  "tool_input": {"command": "grep -r foo ."},
  "session_id": "abc123",
  "timestamp": 1234567890.0
}
```

For `PostToolUse`, an additional `tool_result` field is included.

Hooks communicate back via **exit code** and optionally **stdout JSON**:

| Exit code | Meaning |
|-----------|---------|
| `0` | Allow / OK |
| `1` | Warning — show stderr to user, proceed |
| `2` | Block — abort tool call, show stderr to agent |

To inject context into the agent's conversation, return JSON on stdout:

```json
{"additionalContext": "⚠️ Context at 28% — consider /compact"}
```

### Configuration

Register hooks in `~/.eaudev/config.yml`:

```yaml
hooks:
  enabled: true
  SessionStart:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/session_start.py"
  PreToolUse:
    - matcher: "run_bash"
      command: "python3 ~/.eaudev/hooks/bash_guard.py"
  PostToolUse:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/context_monitor.py"
```

`matcher` supports `*` (all tools) or exact tool names (`run_bash`, `write_file`, `analyst__analyze_target`, etc.).

### Built-in Hooks

Two hooks ship with EauDev in `~/.eaudev/hooks/` (also at `EauDev/eaudev/hooks/`):

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `SessionStart` | Reads `.agent.md`, confirms hooks active |
| `context_monitor.py` | `PostToolUse` | Warns agent when context < 35%/25% |

### Writing a Hook

Minimal Python hook:

```python
#!/usr/bin/env python3
import json, sys

data = json.load(sys.stdin)
tool_name = data.get('tool_name', '')
tool_input = data.get('tool_input', {})

# Block rm -rf
if tool_name == 'run_bash' and 'rm -rf' in tool_input.get('command', ''):
    print('Blocked: rm -rf is not allowed', file=sys.stderr)
    sys.exit(2)

# Inject context into agent
print(json.dumps({'additionalContext': f'Tool {tool_name} completed.'}))
sys.exit(0)
```

### /hooks Command

| Command | Description |
|---------|-------------|
| `/hooks` | List all registered hooks |
| `/hooks add <event> <matcher> <command>` | Register a new hook |
| `/hooks remove <event> <matcher>` | Remove a hook |
| `/hooks enable` / `/hooks disable` | Toggle hooks on/off |
| `/hooks test <event> <tool_name>` | Dry-run hooks for a tool |

### Cluster Pipeline Hooks

Hooks enable deterministic pipeline chaining without agent orchestration:

```yaml
hooks:
  PostToolUse:
    - matcher: "analyst__analyze_target"
      command: "python3 ~/.eaudev/hooks/trigger_archive_ingest.py"
```

When EauDev completes an `analyst__analyze_target` call, the hook fires and automatically ingests the sealed package into Archive MCP — no conversation step required.

---

## Memory System

EauDev has two distinct memory subsystems:

### 1. File-based memory (`.agent.md`)

Loaded at session start, injected into the system prompt:

1. `~/.eaudev/.agent.md` — user-level global preferences
2. `.agent.md` / `.agent.local.md` / `AGENTS.md` — walked from `cwd` up to `~`

Use `/memory init` to ask the agent to generate one. Use `# note` to append inline.

### 2. Persistent memory store (`~/.eaudev/`)

5-layer SQLite-backed store — zero external dependencies:

| Layer | File | What it stores |
|---|---|---|
| **Observation buffer** | `observations.db` | Rolling conversation turns (working memory) |
| **Episodic memory** | `episodic.db` | Session records — what happened, what worked, what to avoid |
| **Facts** | `facts.db` | Typed facts: `fact \| preference \| working_solution \| gotcha \| decision \| failure` — with confidence + provenance |
| **FTS5 index** | `fts5.db` | BM25 keyword search across indexed content |
| **Knowledge graph** | `graph.db` | SQLite entity/relationship graph with recursive CTE traversal |

**Session lifecycle:**
- `session_start` — loads recent episodes + high-confidence facts into context
- turns are recorded into the observation buffer throughout the session
- `session_end` (atexit) — compresses buffer → episodic record, promotes facts

**Design principle:** Memory MCP is a front-line context delivery layer, not long-term storage. All retrieval is deterministic — no embeddings, no fuzzy search, no inference at query time. Long-term archival is handled by Archive MCP.

---

## Tool Permissions

Configurable in `~/.eaudev/config.yml`:

```yaml
tool_permissions:
  tools:
    read_file: allow
    list_directory: allow
    write_file: ask        # arrow-key confirm prompt
    run_bash: ask
    create_file: ask
    move_file: ask
    delete_file: ask
  bash:
    default: ask
    commands:
      - command: "ls.*"
        permission: allow
      - command: "git status"
        permission: allow
      - command: "rm -rf.*"
        permission: deny
```

Permission modes: `allow` · `ask` (interactive confirm) · `deny`

---

## Session Management

Sessions stored in `~/.eaudev/sessions/{uuid}/session_context.json`.  
Filtered by workspace path — each project keeps its own history.  
`/sessions` opens an arrow-key browser with fork, delete, and detail panel.

---

## Context Management

EauDev tracks real KV cache usage from llama.cpp `/slots`. When context fills:

1. `/prune` — strips tool results from history (fast, no LLM call)
2. `/compact` — LLM-assisted summarisation → ~200 word skeleton
3. Automatic model fallback — silently switches to next smaller model from `models.yml`

---

## MCP Integration

Register MCP servers in `~/.eaudev/mcp.json`:

```json
{
  "mcpServers": {
    "archive": {
      "command": "python3",
      "args": ["/path/to/Archive MCP/server.py"]
    },
    "memory": {
      "command": "python3",
      "args": ["/path/to/Memory MCP/server.py"]
    }
  }
}
```

Tools appear as `{server}__{tool}` in the agent's tool set automatically.

---

## VoiceIO

Push-to-talk voice input + TTS output. Requires pytorch env:

```bash
# Dependencies (pytorch env only)
pip install faster-whisper sounddevice torch

# Run with voice enabled
open START_EAUDEV.command
# or: eaudev --voice
```

ASR: faster-whisper (local, Apple Silicon)  
TTS: piper (local, must be on PATH) — set `voice_io.piper_model` in `config.yml` to your `.onnx` model path.

Configure in `~/.eaudev/config.yml`:

```yaml
voice_io:
  whisper_model: base          # tiny / base / small / medium
  whisper_language: en
  piper_model: /path/to/model.onnx   # required for TTS
  piper_cmd: piper
  speak_responses: true
  print_transcript: true
```

Activate with `/voice`, `/voice on`, `/voice off`, `/voice status`, or `/voice config`.

---

## Configuration

`~/.eaudev/config.yml` is auto-created on first run:

```yaml
agent:
  inference:
    endpoint: http://localhost:8080/v1/chat/completions
    model: qwen3-coder
    temperature: 0.3
    max_tokens: 16384
  streaming: true                   # set false for non-streaming (spinner) mode
  additional_system_prompt: ""      # appended to the system prompt every session

console:
  output_format: raw                # raw | markdown — affects non-streaming output rendering
  show_tool_results: true           # show tool call headers and results in terminal

tool_permissions:
  # ... (see Tool Permissions section)

voice_io:
  whisper_model: base
  piper_model: ""                   # path to .onnx piper model — required for TTS
  speak_responses: true
  print_transcript: true

hooks:
  enabled: true
  SessionStart:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/session_start.py"
  PreToolUse:
    - matcher: "run_bash"
      command: "python3 ~/.eaudev/hooks/bash_guard.py"
  PostToolUse:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/context_monitor.py"
```

---

## What Was Removed vs Rovodev

| Removed | Reason |
|---|---|
| `nemo` / `pydantic_ai` | Atlassian internal agent framework |
| `logfire` | Atlassian observability platform |
| Analytics / telemetry | No tracking in local-first tool |
| Jira / Confluence tools | Cloud-only |
| Atlassian auth | No cloud auth needed |
| `AdaptiveFallbackModel` | Anthropic model chain |
| `/models` (Anthropic list) | Replaced by `/model` via llama.cpp registry |
| `/feedback` | Atlassian support portal |
| `EntitlementCheckFailed` | Atlassian billing |
| `RateLimitExceededError` | Atlassian quota |
| MCP Atlassian native server | `mcp.atlassian.com` |
| Daily token bar | Requires `api.atlassian.com` |

---

## Known Issues

- `/prune` uses string-prefix matching to identify tool results — may miss MCP tool results with non-standard prefixes
- `output_format: simple` renders identically to `raw` — no distinct formatting implemented
- `read_file` and `list_directory` ignore `tool_permissions` config (always allowed — reads are non-destructive, but the config option has no effect)
- Session browser `fork` creates a clean copy but doesn't update `parent_session_id` in detail panel
- `/instructions` command is a stub — prints "not implemented"
- Knowledge graph memory layer (graph.db) has no query UI — data is written but not surfaced by any slash command
- Memory store integration is new (Feb 23, 2026) — awaiting real-session validation

---

## Dependencies

```
prompt_toolkit>=3.0   — Rich Live input panel, arrow-key menus
rich>=13.0            — all terminal rendering
pydantic>=2.0         — config models
pyyaml>=6.0           — config + memory files
bashlex>=0.18         — shell command parsing in tool permissions
loguru>=0.7           — structured logging
humanize>=4.0         — naturaltime() for session timestamps
```

```bash
pip3 install -e . --break-system-packages
```

---

## Source Reference

Ported from: `Index Ideas Sources/atlassian-rovo-source-code-z80-dump-master`  
Original: rovodev v0.6.8  
EauDev: v0.1.0 — built Feb 2026
