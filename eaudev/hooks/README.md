# EauDev Built-in Hooks

This directory contains the built-in hooks that ship with EauDev.

## context_monitor.py

**Event:** `PostToolUse`  
**Matcher:** `*` (all tools)  
**Purpose:** Monitors context window usage and warns the agent when approaching limits.

### How it works

After each LLM response, EauDev writes context metrics to `/tmp/eaudev-ctx-{session_id}.json`:

```json
{
  "used_tokens": 45000,
  "total_tokens": 65536,
  "remaining_pct": 31.3,
  "timestamp": 1234567890.0
}
```

`context_monitor.py` reads this file and injects warnings into the agent's conversation:

- **< 35% remaining:** ⚠️ advisory warning — agent can choose to `/compact`
- **< 25% remaining:** 🚨 urgent warning — agent should `/compact` immediately

### Registration

```yaml
hooks:
  enabled: true
  PostToolUse:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/context_monitor.py"
```

---

## session_start.py

**Event:** `SessionStart`  
**Matcher:** `*`  
**Purpose:** Runs at session open — reads `.agent.md` and confirms hooks are active.

### Registration

```yaml
hooks:
  enabled: true
  SessionStart:
    - matcher: "*"
      command: "python3 ~/.eaudev/hooks/session_start.py"
```

---

## Writing your own hook

See `EauDev/README.md` → Hooks section for full protocol documentation.

The minimal structure:

```python
#!/usr/bin/env python3
import json, sys

data = json.load(sys.stdin)

# ... your logic ...

# Optionally inject context into agent conversation:
# print(json.dumps({"additionalContext": "your message"}))

sys.exit(0)  # allow
# sys.exit(1) # warn
# sys.exit(2) # block
```

## See also

- `Documents/GENERATE_HOOK_MCP_SPECIFICATION.md` — spec for the Generate Hook MCP server
- `EauDev/eaudev/modules/hooks.py` — hook runner implementation
- `EauDev/eaudev/common/config_model.py` — `HooksConfig` schema
