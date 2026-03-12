#!/usr/bin/env python3
"""
EauDev PostToolUse hook — Context Window Monitor.

Reads context usage metrics from a temp file written by the EauDev session
and injects warnings into the agent's context when the window is running low.

Protocol:
  stdin  → JSON: {tool_name, tool_input, tool_result, session_id}
  stdout → JSON: {"additionalContext": "..."} on warning, empty on OK
  exit 0 always (PostToolUse hooks are informational only)

Temp files:
  /tmp/eaudev-ctx-{session_id}.json       — metrics (written by EauDev)
  /tmp/eaudev-ctx-{session_id}-warned.json — debounce state (written by this hook)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
WARNING_PCT  = 35.0   # remaining_pct <= this → WARNING
CRITICAL_PCT = 20.0   # remaining_pct <= this → CRITICAL
METRICS_TTL  = 60.0   # seconds before metrics are considered stale
DEBOUNCE_N   = 5      # warn at most once every N tool calls (unless escalating)


def load_json_file(path: Path) -> dict | None:
    """Return parsed JSON from *path*, or None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_json_file(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path*, ignoring errors."""
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass


def main() -> None:
    # ── 1. Read stdin ─────────────────────────────────────────────────────────
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)

    session_id = payload.get("session_id", "")

    # ── 2. Read metrics file ──────────────────────────────────────────────────
    # session_id is a UUID — collision/symlink attack risk negligible on single-user systems
    metrics_path = Path(f"/tmp/eaudev-ctx-{session_id}.json")
    if not metrics_path.exists():
        sys.exit(0)

    metrics = load_json_file(metrics_path)
    if not metrics:
        sys.exit(0)

    # ── 3. Staleness check ────────────────────────────────────────────────────
    timestamp = metrics.get("timestamp", 0.0)
    if (time.time() - timestamp) > METRICS_TTL:
        sys.exit(0)

    # ── 4. Extract values ─────────────────────────────────────────────────────
    used_tokens   = int(metrics.get("used_tokens",   0))
    total_tokens  = int(metrics.get("total_tokens",  1))
    remaining_pct = float(metrics.get("remaining_pct", 100.0))
    used_pct      = 100.0 - remaining_pct

    # ── 5. Determine severity ─────────────────────────────────────────────────
    if remaining_pct <= CRITICAL_PCT:
        severity = "critical"
    elif remaining_pct <= WARNING_PCT:
        severity = "warning"
    else:
        sys.exit(0)  # No warning needed

    # ── 6. Debounce ───────────────────────────────────────────────────────────
    warned_path = Path(f"/tmp/eaudev-ctx-{session_id}-warned.json")
    state = load_json_file(warned_path) or {"call_count": 0, "last_severity": None}

    call_count    = int(state.get("call_count", 0)) + 1
    last_severity = state.get("last_severity")

    # Escalation: always warn if severity went from warning → critical
    escalating = (last_severity == "warning" and severity == "critical")

    # Suppress if not enough calls have passed AND not escalating
    if not escalating and (call_count % DEBOUNCE_N) != 0:
        # Update call count but don't warn
        save_json_file(warned_path, {"call_count": call_count, "last_severity": last_severity})
        sys.exit(0)

    # ── 7. Build message ──────────────────────────────────────────────────────
    if severity == "critical":
        msg = (
            f"🚨 CRITICAL: Context window at {used_pct:.0f}% used. "
            "Stop current task, run /compact NOW, save state before continuing."
        )
    else:
        msg = (
            f"⚠️ Context window at {used_pct:.0f}% used ({remaining_pct:.0f}% remaining). "
            "Consider /compact or /prune to avoid context rot."
        )

    # ── 8. Persist updated debounce state ─────────────────────────────────────
    save_json_file(warned_path, {"call_count": call_count, "last_severity": severity})

    # ── 9. Emit additionalContext ─────────────────────────────────────────────
    print(json.dumps({"additionalContext": msg}))
    sys.exit(0)


if __name__ == "__main__":
    main()
