# generate_hook_mcp

An MCP server for generating, managing, and testing EauDev lifecycle hooks.

## Tools

| Tool | Description |
|------|-------------|
| `generate_hook` | Generate a Python hook script using a local LLM based on a natural language description |
| `list_hooks` | List all hooks registered in `~/.eaudev/config.yml` |
| `register_hook` | Add a hook entry to `~/.eaudev/config.yml` |
| `remove_hook` | Remove matching hook entries from `~/.eaudev/config.yml` |
| `test_hook` | Run a hook script against a sample input and report the verdict |

## Hook Protocol

Hooks are Python scripts that:
- Read JSON from stdin: `{"tool_name": str, "tool_input": dict, "session_id": str}`
- PostToolUse hooks also receive `"tool_result"` in the JSON
- Exit with: `0` = allow, `1` = warn (print to stderr), `2` = block (print reason to stderr)

## Hook Registration Format (`~/.eaudev/config.yml`)

```yaml
hooks:
  enabled: true
  PreToolUse:
    - matcher: "run_bash"
      command: "python3 ~/.eaudev/hooks/bash_guard.py"
  PostToolUse:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/audit_log.py"
```

## Running the Server

```bash
python3 -m EauDev.generate_hook_mcp.server
# or
fastmcp run EauDev/generate_hook_mcp/server.py
```
