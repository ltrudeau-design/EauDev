"""EauDev model registry — loads ~/.eaudev/models.yml and manages server switching."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from eaudev.constants import CLUSTER_DIR

MODELS_PATH = Path.home() / ".eaudev" / "models.yml"
console = Console()

# Tracks the server process started by EauDev (None if server was pre-existing).
# Only set when EauDev starts the server itself — never for pre-existing servers.
_active_proc: Optional[subprocess.Popen] = None


def get_active_server_proc() -> Optional[subprocess.Popen]:
    """Return the inference server process handle started by EauDev, or None."""
    return _active_proc


@dataclass
class ModelFlags:
    ngl: int = 99
    threads: int = 8
    flash_attn: bool = True
    mlock: bool = True
    no_mmap: bool = True
    cont_batching: bool = True
    batch_size: int = 2048
    ubatch_size: int = 512
    temperature: float = 0.7


@dataclass
class ServerConfig:
    name: str
    display: str
    path: str
    port: int = 8080
    context: int = 131072
    size_gb: float = 0.0
    tags: list[str] = field(default_factory=list)
    flags: ModelFlags = field(default_factory=ModelFlags)
    server_type: str = ""   # "mlx" | "llama" | "" (auto-detect from path)

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1/chat/completions"

    @property
    def resolved_path(self) -> str:
        return str(Path(self.path).expanduser())

    @property
    def size_display(self) -> str:
        return f"{self.size_gb:.0f}GB" if self.size_gb >= 1 else f"{self.size_gb:.1f}GB"

    @property
    def context_display(self) -> str:
        if self.context >= 1000:
            return f"{self.context // 1000}K"
        return str(self.context)

    @property
    def api_model_name(self) -> str:
        """Model name to send in API requests.
        mlx_lm.server uses the full model directory path as its model ID.
        llama-server uses the short name from config.
        """
        is_mlx = (
            self.server_type == "mlx"
            or (self.server_type == "" and not self.resolved_path.endswith(".gguf"))
        )
        return self.resolved_path if is_mlx else self.name

    @property
    def menu_label(self) -> str:
        return f"{self.display:<55} {self.size_display:>5}  {self.context_display:>6} ctx"


def load_model_registry(path: Path = MODELS_PATH) -> list[ServerConfig]:
    """Load models from ~/.eaudev/models.yml. Returns empty list if not found."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        configs = []
        for m in data.get("models", []):
            flags_data = m.get("flags", {})
            flags = ModelFlags(
                ngl=flags_data.get("ngl", 99),
                threads=flags_data.get("threads", 8),
                flash_attn=flags_data.get("flash_attn", True),
                mlock=flags_data.get("mlock", True),
                no_mmap=flags_data.get("no_mmap", True),
                cont_batching=flags_data.get("cont_batching", True),
                batch_size=flags_data.get("batch_size", 2048),
                ubatch_size=flags_data.get("ubatch_size", 512),
                temperature=flags_data.get("temperature", 0.7),
            )
            configs.append(ServerConfig(
                name=m["name"],
                display=m["display"],
                path=m["path"],
                port=m.get("port", 8080),
                context=m.get("context", 131072),
                size_gb=m.get("size_gb", 0.0),
                tags=m.get("tags", []),
                flags=flags,
                server_type=m.get("server_type", ""),
            ))
        return configs
    except Exception as e:
        console.print(f"[bright_black][model registry: failed to load {path}: {e}][/bright_black]")
        return []


def _is_mlx_model(cfg: ServerConfig) -> bool:
    """Return True if this model should be served via mlx_lm.server instead of llama-server."""
    if cfg.server_type == "mlx":
        return True
    if cfg.server_type == "llama":
        return False
    # Auto-detect: directory path → MLX, .gguf file → llama-server
    return not cfg.resolved_path.endswith(".gguf")


def _get_mlx_python() -> str:
    """Return the Python executable that has mlx_lm installed.

    Always prefers the pytorch_env Python (canonical MLX env on this machine).
    Falls back to sys.executable only if pytorch_env is missing.
    """
    import sys
    pytorch_python = str(CLUSTER_DIR / "pytorch_env" / "bin" / "python")
    if Path(pytorch_python).exists():
        return pytorch_python
    return sys.executable


def _build_mlx_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build the mlx_lm.server command line from a ServerConfig."""
    return [
        _get_mlx_python(), "-m", "mlx_lm.server",
        "--model", cfg.resolved_path,
        "--port", str(cfg.port),
        "--log-level", "INFO",
    ]


def _build_llama_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build the llama-server command line from a ServerConfig."""
    f = cfg.flags
    cmd = [
        "llama-server",
        "-m", cfg.resolved_path,
        "--host", "0.0.0.0",
        "--port", str(cfg.port),
        "-ngl", str(f.ngl),
        "-t", str(f.threads),
        "-c", str(cfg.context),
        "--temp", str(f.temperature),
        "-np", "1",
        "--parallel", "1",
    ]
    if f.flash_attn:
        cmd += ["--flash-attn", "on"]
    if f.mlock:
        cmd.append("--mlock")
    if f.no_mmap:
        cmd.append("--no-mmap")
    if f.cont_batching:
        cmd += ["-cb", "--cont-batching"]
    if f.batch_size != 2048:
        cmd += ["-b", str(f.batch_size)]
    if f.ubatch_size != 512:
        cmd += ["-ub", str(f.ubatch_size)]
    return cmd


def _kill_existing_server(port: int) -> None:
    """Kill any process listening on the given port."""
    from loguru import logger
    
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if pids:
            time.sleep(1)  # let it die cleanly
    except FileNotFoundError:
        logger.debug("[model_registry] lsof not found — cannot kill existing server on port {}", port)
    except Exception as e:
        logger.debug("[model_registry] _kill_existing_server failed: {}", e)


def _wait_for_server(endpoint: str, timeout: int = 60) -> bool:
    """Poll /health until the server is ready or timeout expires."""
    health_url = endpoint.replace("/v1/chat/completions", "/health")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def autostart_server(cfg: ServerConfig) -> Optional[subprocess.Popen]:
    """
    Start the inference server on EauDev launch (no pre-existing server detected).
    Returns the Popen handle on success, None on failure.
    Updates the module-level _active_proc so shutdown can terminate it cleanly.
    Does NOT kill any existing process first — caller verified the port is free.
    """
    global _active_proc

    if _is_mlx_model(cfg):
        cmd = _build_mlx_server_cmd(cfg)
        not_found_msg = "[red]mlx_lm not found. Run: pip install mlx-lm[/red]"
    else:
        cmd = _build_llama_server_cmd(cfg)
        not_found_msg = "[red]llama-server not found in PATH. Is llama.cpp installed?[/red]"

    console.print(f"[bright_black]Starting [bold]{cfg.display}[/bold]...[/bright_black]", end="")

    log_path = "/tmp/eaudev_mlx_server.log"
    try:
        log_fh = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # isolate from Ctrl+C in EauDev terminal
            )
        finally:
            log_fh.close()  # Popen duplicates the fd — safe to close our handle
    except FileNotFoundError:
        console.print(f"\n{not_found_msg}")
        return None
    except Exception:
        raise

    if _wait_for_server(cfg.endpoint, timeout=60):
        console.print(" [green]Ready.[/green]")
        _active_proc = proc
        return proc
    else:
        console.print(f"\n[red]Server failed to start within 60s.[/red]")
        console.print(f"[bright_black]Server log: {log_path}[/bright_black]")
        console.print(f"[bright_black]Command: {' '.join(cmd)}[/bright_black]")
        proc.terminate()
        return None


def switch_model(cfg: ServerConfig) -> bool:
    """
    Stop the current server, start the new one, wait for it to be ready.
    Returns True on success, False on failure.
    Updates _active_proc so atexit shutdown tracks the new process.
    """
    global _active_proc

    console.print(f"\n[bright_black]Stopping current server on port {cfg.port}...[/bright_black]")
    _kill_existing_server(cfg.port)

    console.print(f"[bright_black]Loading [bold]{cfg.display}[/bold] ({cfg.size_display})...[/bright_black]")
    if _is_mlx_model(cfg):
        cmd = _build_mlx_server_cmd(cfg)
        not_found_msg = "[red]mlx_lm not found. Run: pip install mlx-lm[/red]"
    else:
        cmd = _build_llama_server_cmd(cfg)
        not_found_msg = "[red]llama-server not found in PATH. Is llama.cpp installed?[/red]"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        console.print(not_found_msg)
        return False

    console.print(f"[bright_black]Waiting for server to be ready (up to 60s)...[/bright_black]", end="")
    if _wait_for_server(cfg.endpoint, timeout=60):
        console.print(f"\n[green]✓ Switched to {cfg.display}[/green]")
        console.print(f"[bright_black]Context: {cfg.context_display} | Endpoint: {cfg.endpoint}[/bright_black]\n")
        _active_proc = proc
        return True
    else:
        console.print(f"\n[red]Server failed to start within 60s. Check model path: {cfg.resolved_path}[/red]")
        proc.terminate()
        return False


def get_current_model_name(endpoint: str) -> str:
    """Query /props to get the current model name from a running server."""
    props_url = endpoint.replace("/v1/chat/completions", "/props")
    try:
        with urllib.request.urlopen(props_url, timeout=3) as r:
            data = json.loads(r.read())
            return data.get("default_generation_settings", {}).get("model", "unknown")
    except Exception:
        return "unknown"
