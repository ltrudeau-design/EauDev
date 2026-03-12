# EauDev UI Polish Changes

**Date:** 2026-03-08  
**Scope:** Token counter, styling consistency, error handling, VoiceIO integration

---

## Issues Fixed

### 1. Token Counter Timing (CRITICAL)

**Problem:** Token bar appeared one turn late — only showed after second exchange instead of after first response.

**Root Cause:** Condition `last_tokens > 0 OR session.num_messages > 1` checked before `last_tokens` was updated.

**Fix:** Changed condition to `last_tokens > 0 OR session.num_messages > 2` — token bar now appears correctly after first agent response.

**File:** `eaudev/commands/run/command.py:2264`

---

### 2. VoiceIO Panel Styling Mismatch

**Problem:** VoiceIO listening panel used emoji (🎤) and green styling while rest of UI uses minimal styling with `bright_black` borders.

**Fix:** 
- Removed emoji
- Changed border from `green` to `bright_black`
- Changed text from `[bold green]` to `[bold]`
- Added timeout feedback message: "No speech detected — listening again."

**File:** `eaudev/commands/run/command.py:2239-2256`

---

### 3. Silent API Failures

**Problem:** `_get_context_limit()` and `_get_n_past()` swallowed exceptions with bare `except: pass` — impossible to debug connection issues.

**Fix:** Added `logger.debug()` calls for all failure paths. Errors now appear in `~/.eaudev/eaudev.log` at DEBUG level.

**Files:** 
- `eaudev/commands/run/command.py:786-801` (_get_context_limit)
- `eaudev/commands/run/command.py:877-879` (_get_n_past)

---

### 4. Exception Swallowing in UI

**Problem:** Slash command footer render exception silently swallowed — made debugging command registry issues impossible.

**Fix:** Changed `except Exception: pass` to log the exception at DEBUG level.

**File:** `eaudev/ui/components/user_input_panel.py:133-136`

---

## Styling Consistency Audit

### Current Patterns (DO NOT CHANGE)

| Element | Style |
|---------|-------|
| Tool results | `[bright_black]...[/bright_black]` |
| Usage hints | `[bright_black]Usage: ...[/bright_black]` |
| Status messages | `[dim]...[/dim]` |
| Success | `[green]...[/green]` |
| Warnings | `[yellow]...[/yellow]` |
| Errors | `[red]...[/red]` |
| Panels | `border_style="bright_black"` |

### Inconsistencies Found (NOT YET FIXED)

| Location | Issue | Recommendation |
|----------|-------|----------------|
| Line 1129 | Prune message has leading blank line | Remove blank line for consistency |
| Line 2306 | "New session started" — no blank line | Add blank line before |
| Line 2326 | "Context cleared" — no blank line | Add blank line before |
| Line 1322 | `/shadow` uses `Rule(" ◌ shadow ")` | Consider removing or standardizing |

---

## Recommendations for Future Polish

### High Priority

1. **Add token counter to response area** — Show token delta after each agent response, not just before user input
2. **Consistent spacing** — All status messages should have same blank line treatment
3. **Error toast notifications** — Non-blocking error display for API failures

### Medium Priority

4. **Progress indicators** — Show spinning indicator during long tool calls (>5s)
5. **Command autocomplete** — Expand slash command suggestions beyond current footer
6. **Session switch confirmation** — Show brief flash when switching sessions

### Low Priority

7. **Color theme support** --dark/--light mode for terminal
8. **Compact mode** --flag to reduce vertical whitespace
9. **ASCII fallback** — For terminals without Unicode box-drawing support

---

## Testing Checklist

- [ ] Token bar appears after first agent response
- [ ] Token bar updates correctly after /prune
- [ ] Token bar updates correctly after /compact
- [ ] VoiceIO panel matches other panel styling
- [ ] VoiceIO timeout shows feedback message
- [ ] Slash command suggestions appear when typing /
- [ ] No console errors when typing invalid slash commands
- [ ] eaudev.log contains debug messages for API failures


---

## Qwen Code-Inspired Enhancements (NEW)

### 5. Multi-Stage Progress Indicator

**What:** Enhanced status bar that shows exactly what stage the agent is in.

**Stages:**
- 🧠 Analyzing request...
- 📋 Planning approach...
- 🔧 Executing tool...
- ⏳ Waiting for result...
- ✍️ Generating response...

**Why:** Users always know what's happening — no more "is it stuck?" uncertainty.

**File:** `eaudev/commands/run/command.py:693-730`

---

### 6. Session State Indicator

**What:** Shows current session name and workspace at startup.

**Example:**
```
Session: Adding voice support to EauDev | Workspace: Cluster
```

**Why:** Quick visual confirmation of which session/workspace you're in (especially useful with multiple terminal windows).

**File:** `eaudev/commands/run/command.py:2197-2202`

---

### 7. Command Suggestions (Typo Recovery)

**What:** When user types unknown command, suggests similar commands.

**Example:**
```
Unknown command: /mem
  Did you mean: /memory or /mcp?
```

**Why:** Reduces frustration from typos, helps users discover commands.

**File:** `eaudev/commands/run/command.py:2354-2365`

---

### 8. Enhanced Error Recovery Hints

**What:** Error messages now include actionable tips.

**Before:**
```
Context too large and no smaller model available. Use /prune or /compact.
```

**After:**
```
Context too large and no smaller model available. Use /prune or /compact.
  Tip: /prune removes tool results, /compact summarizes middle turns.
```

**Why:** Users know exactly how to fix the problem without consulting docs.

**File:** `eaudev/commands/run/command.py:998-1000`

---

## Summary of All UI Improvements

| Feature | Category | Impact |
|---------|----------|--------|
| Token counter timing fix | Bug fix | High |
| VoiceIO panel styling | Consistency | Medium |
| API error logging | Debugging | High |
| Exception logging in footer | Debugging | Medium |
| Multi-stage progress | UX enhancement | High |
| Session state indicator | UX enhancement | Medium |
| Command suggestions | Error recovery | High |
| Error recovery hints | Error recovery | High |

---

## Future Enhancements (Backlog)

### Inspired by Qwen Code / Cursor

1. **Inline diff preview** — Show file changes inline before accepting
2. **Tool confirmation for destructive ops** — "This will delete 5 files. Proceed?"
3. **Streaming tool output** — Show bash output as it streams, not after completion
4. **Context window visualization** — Show which parts of context are being used
5. **Recent commands history** — Show last 3 commands in footer
6. **Auto-suggest for file paths** — Tab-complete file paths in tool arguments

### EauDev-Specific

7. **Memory layer quick-stats** — One-line summary in footer (Facts: 47 | Episodes: 12 | Graph: 8 entities)
8. **VoiceIO waveform indicator** — Visual feedback when voice is detected
9. **LoRA training progress** — Show training status when session ends
10. **MCP server health** — Green/red indicator for each server in footer

