"""EauDev constants."""

from pathlib import Path

ENDPOINT = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.5-9B-MLX-bf16"  # Updated 2026-03-07 — MLX migration (mlx_lm.server)
HOME = str(Path.home())
MLX_MODEL_PATH = str(Path.home() / ".cluster" / "MLX Models" / "Qwen3.5-9B-MLX-bf16")
MLX_ADAPTER_PATH = str(Path.home() / ".cluster" / "adapters")
CONFIG_PATH = Path.home() / ".eaudev" / "config.yml"
SESSION_DIR = Path.home() / ".eaudev" / "sessions"
LOG_PATH = Path.home() / ".eaudev" / "eaudev.log"
MCP_CONFIG_PATH = Path.home() / ".eaudev" / "mcp.json"
HOOKS_DIR = Path.home() / ".eaudev" / "hooks"
DEFAULT_PANEL_WIDTH = 100
DEFAULT_EXIT_COMMANDS = ["/exit", "/quit", "exit", "quit", "q", "/q"]
VERSION = "0.1.0"

# Memory file names — mirrors rovodev constants.py exactly
WORKSPACE_MEMORY_FILE_NAMES = [".agent.md", ".agent.local.md", "AGENTS.md", "AGENTS.local.md"]
USER_MEMORY_FILE_NAMES = [".eaudev/.agent.md"]
