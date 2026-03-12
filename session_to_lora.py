#!/usr/bin/env python3
"""
session_to_lora.py — EauDev session memory → LoRA delta pipeline.

Standalone script invoked by EauDev at session end (as a detached subprocess)
or run manually by the user with a --model path for actual MLX training.

Steps:
  1. export_consolidation_artefact() → alpaca JSONL (~2s)
  2. mlx_lm.lora --iters 30 --rank 8 [--warm-start existing adapter] (~15-30s)
  3. mlx_lm.fuse → updated session_lora.safetensors (~5s)
  4. Archive float16 checkpoint (merge integrity)
  5. increment_session_count() → update lifecycle state
  6. If session_count >= 20: log merge warning (no auto-merge)

Usage:
    # Full pipeline (MLX training + fuse):
    python session_to_lora.py --session-id SESSION_ID --model /path/to/mlx/model

    # Export JSONL only (no training):
    python session_to_lora.py --session-id SESSION_ID --dry-run

    # Auto mode (invoked by EauDev at session end — no model, just exports + increments):
    python session_to_lora.py --session-id SESSION_ID

LoRA stacking ceiling: 2 adapters maximum (enforced in lora_lifecycle.py).
Float16 checkpoint retained at each merge — prevents quantisation error compounding.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure EauDev package is importable when run as a standalone script
_EAUDEV_ROOT = Path(__file__).resolve().parent
if str(_EAUDEV_ROOT) not in sys.path:
    sys.path.insert(0, str(_EAUDEV_ROOT))

from eaudev.memory.consolidation import export_consolidation_artefact
from eaudev.memory.lora_lifecycle import (
    get_lora_status,
    increment_session_count,
    set_current_adapter,
)

logging.basicConfig(level=logging.INFO, format='[session_to_lora] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_CLUSTER_DIR     = Path.home() / '.cluster'
_ADAPTERS_DIR    = _CLUSTER_DIR / 'adapters'
_FLOAT16_DIR     = _ADAPTERS_DIR / 'float16'
_JSONL_PATH      = _CLUSTER_DIR / 'session_consolidation.jsonl'
_ADAPTER_NAME    = 'eaudev_session_lora'
_FUSED_ADAPTER   = _ADAPTERS_DIR / f'{_ADAPTER_NAME}.safetensors'


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EauDev session memory → LoRA delta pipeline"
    )
    parser.add_argument(
        "--session-id", required=True,
        help="Session ID to consolidate (must match an episodic record)"
    )
    parser.add_argument(
        "--model", default=None,
        help="MLX model path for training (HuggingFace or MLX format). "
             "If omitted, exports JSONL and updates session count only."
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Existing adapter path for warm-start. Defaults to the fused adapter "
             "at ~/.cluster/adapters/eaudev_session_lora.safetensors if it exists."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Export JSONL and update session count, but skip MLX training."
    )
    args = parser.parse_args()

    _ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    _FLOAT16_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Export consolidation artefact ──────────────────────────────────
    log.info("Exporting consolidation artefact for session: %s", args.session_id)
    result = export_consolidation_artefact(
        session_id=args.session_id,
        output_path=str(_JSONL_PATH),
        include_facts=True,
        min_fact_confidence=0.8,
    )

    if not result.get("success"):
        log.error("export_consolidation_artefact failed: %s", result)
        sys.exit(1)

    record_count = result["record_count"]
    log.info(
        "Exported %d training records to %s (episode_found=%s, facts=%d)",
        record_count, _JSONL_PATH,
        result.get("episode_found"), result.get("facts_count", 0),
    )

    if record_count == 0:
        log.warning("No training records — skipping LoRA training, updating session count.")
        _finalize(args.session_id)
        return

    if args.dry_run:
        log.info("Dry run — skipping MLX training. JSONL written to %s", _JSONL_PATH)
        _finalize(args.session_id)
        return

    if not args.model:
        log.info(
            "No --model specified. JSONL written to %s\n"
            "To train: python session_to_lora.py --session-id %s --model /path/to/mlx/model",
            _JSONL_PATH, args.session_id,
        )
        _finalize(args.session_id)
        return

    # ── Step 2: MLX LoRA training ──────────────────────────────────────────────
    adapter_out_dir = _ADAPTERS_DIR / _ADAPTER_NAME   # mlx_lm.lora writes here

    lora_cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",         args.model,
        "--data",          str(_JSONL_PATH.parent),
        "--iters",         "30",
        "--batch-size",    "1",
        "--lora-rank",     "8",
        "--learning-rate", "2e-4",
        "--adapter-path",  str(adapter_out_dir),
    ]

    # Warm-start from existing adapter if available
    warm_adapter = args.adapter or (str(_FUSED_ADAPTER) if _FUSED_ADAPTER.exists() else None)
    if warm_adapter and Path(warm_adapter).exists():
        lora_cmd += ["--resume-adapter-file", warm_adapter]
        log.info("Warm-starting from adapter: %s", warm_adapter)
    else:
        log.info("No existing adapter found — training fresh LoRA_A.")

    log.info("Running MLX LoRA training (%s iters, rank 8)...", 30)
    ret = subprocess.run(lora_cmd)
    if ret.returncode != 0:
        log.error("mlx_lm.lora failed (exit %d)", ret.returncode)
        sys.exit(1)

    # ── Step 3: Fuse adapter into safetensors ──────────────────────────────────
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        args.model,
        "--adapter-path", str(adapter_out_dir),
        "--save-path",    str(_FUSED_ADAPTER),
        "--de-quantize",  # keep float16 for merge chain integrity
    ]
    log.info("Fusing adapter...")
    ret = subprocess.run(fuse_cmd)
    if ret.returncode != 0:
        log.error("mlx_lm.fuse failed (exit %d)", ret.returncode)
        sys.exit(1)

    # ── Step 4: Archive float16 checkpoint ────────────────────────────────────
    # Float16 checkpoint retained at each merge — prevents quantisation error
    # compounding across merge generations. Log a warning if not present.
    status = get_lora_status()
    merge_gen = status.get("merge_generation", 0)
    float16_archive = _FLOAT16_DIR / f"{_ADAPTER_NAME}_gen{merge_gen:03d}.safetensors"

    if _FUSED_ADAPTER.exists():
        shutil.copy2(_FUSED_ADAPTER, float16_archive)
        log.info("Float16 checkpoint archived: %s", float16_archive)
    else:
        log.warning(
            "Float16 checkpoint not found at %s — merge integrity at risk. "
            "Ensure mlx_lm.fuse produced output before the next merge.",
            _FUSED_ADAPTER,
        )

    # Update lifecycle state
    set_current_adapter(str(_FUSED_ADAPTER))
    _finalize(args.session_id)


def _finalize(session_id: str) -> None:
    """Increment session_count and emit merge warnings as appropriate."""
    updated = increment_session_count()
    count   = updated.get("session_count", 0)
    log.info("Session count: %d", count)

    if updated.get("merge_required"):
        log.warning(
            "SESSION COUNT %d — MERGE REQUIRED.\n"
            "  Both LoRA adapter slots are saturated.\n"
            "  Run the merge pipeline:\n"
            "    1. mlx_lm.fuse (LoRA_A into base)\n"
            "    2. mlx_lm.fuse (LoRA_B into result of step 1)\n"
            "    3. Re-quantise to Q4_K_M → new GGUF\n"
            "    4. Retain float16 checkpoint before quantising\n"
            "    5. python session_to_lora.py --session-id <next> --model <new_base>",
            count,
        )
    elif updated.get("adapter_ceiling_warning"):
        log.warning(
            "SESSION COUNT %d — Adapter slot A saturated. "
            "LoRA_B slot now active. Consider scheduling a merge after %d more sessions.",
            count, updated.get("sessions_per_adapter", 20),
        )


if __name__ == "__main__":
    main()
