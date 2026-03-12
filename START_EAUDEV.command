#!/bin/bash
# EauDev — double-click launcher for macOS
# Uses the pytorch env which has all voice deps (torch, faster-whisper, sounddevice)

# EauDev lives inside the Cluster root — launch from the Cluster root so that
# workdir is the whole workspace, not the EauDev subdirectory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_ROOT="$(dirname "$SCRIPT_DIR")"

# cd to Cluster root so workdir auto-detection anchors correctly
cd "$CLUSTER_ROOT"

# Add EauDev/ to PYTHONPATH so `python3 -m eaudev` finds the package
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Preferred: pytorch env (has torch, faster-whisper, sounddevice for VoiceIO)
PYTHON="/Users/eaumac/.cluster/pytorch_env/bin/python3"

# Fallback chain: specialist venv → system python3
if [ ! -f "$PYTHON" ]; then
    PYTHON="/Users/eaumac/.specialist/venv/bin/python3"
fi
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

clear
"$PYTHON" -m eaudev "$@"

# EauDev has exited — drop into interactive shell so the window stays alive.
echo ""
echo "EauDev exited. Type 'exit' to close this window."
exec /bin/zsh -i
