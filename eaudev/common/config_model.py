"""EauDev configuration models (pydantic v2)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from eaudev.constants import (
    CONFIG_PATH,
    ENDPOINT,
    HOOKS_DIR,
    LOG_PATH,
    MCP_CONFIG_PATH,
    MODEL,
    SESSION_DIR,
)

Permission = Literal["allow", "ask", "deny"]


class BaseConfig(BaseModel):
    model_config = {"extra": "ignore"}


class InferenceConfig(BaseConfig):
    """Local inference server configuration."""

    endpoint:        str   = Field(default=ENDPOINT, description="llama.cpp OpenAI-compatible endpoint URL")
    model:           str   = Field(default=MODEL,    description="Model name sent to the inference server")
    temperature:     float = Field(default=0.7,      description="Sampling temperature")
    top_p:           float = Field(default=0.8,      description="Top-p nucleus sampling")
    top_k:           int   = Field(default=20,       description="Top-k sampling")
    min_p:           float = Field(default=0.0,      description="Min-p sampling")
    max_tokens:      int   = Field(default=32768,    description="Maximum tokens per response")
    enable_thinking: bool  = Field(default=False,    description="Qwen3 thinking mode (slower, deeper reasoning)")


class AgentConfig(BaseConfig):
    """Agent configuration."""

    additional_system_prompt: str | None = Field(
        default=None, description="Additional system prompt appended to the default"
    )
    streaming: bool = Field(default=True, description="Enable streaming responses")
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class SessionsConfig(BaseConfig):
    """Sessions configuration."""

    auto_restore: bool = Field(default=False, description="Auto-restore last session on startup")
    persistence_dir: str = Field(
        default=str(SESSION_DIR), description="Directory where session data is stored"
    )


class ConsoleConfig(BaseConfig):
    """Console configuration."""

    output_format: Literal["markdown", "simple", "raw"] = Field(
        default="markdown", description="Output format (markdown, simple, raw)"
    )
    show_tool_results: bool = Field(default=True, description="Show tool execution results")


class LoggingConfig(BaseConfig):
    """Logging configuration."""

    path: str = Field(default=str(LOG_PATH), description="Path to the log file")


class MCPConfig(BaseConfig):
    """MCP configuration."""

    mcp_config_path: str = Field(
        default=str(MCP_CONFIG_PATH), description="Path to MCP configuration JSON file"
    )


class BashCommandConfig(BaseConfig):
    """A single bash command permission entry."""

    command: str
    permission: Permission


class BashPermissionConfig(BaseConfig):
    """Bash tool permission settings."""

    default: Permission = Field(default="ask", description="Default permission for unlisted commands")
    commands: list[BashCommandConfig] = Field(
        default_factory=lambda: [
            BashCommandConfig(command="ls.*", permission="allow"),
            BashCommandConfig(command="cat.*", permission="allow"),
            BashCommandConfig(command="echo.*", permission="allow"),
            BashCommandConfig(command="git status", permission="allow"),
            BashCommandConfig(command="git diff.*", permission="allow"),
            BashCommandConfig(command="git log.*", permission="allow"),
            BashCommandConfig(command="pwd", permission="allow"),
        ],
        description="Specific bash commands and their permissions",
    )


class ToolPermissionsConfig(BaseConfig):
    """Tool permissions configuration."""

    allow_all: bool = False
    default: Permission = Field(default="ask", description="Default for unlisted tools")
    tools: dict[str, Permission] = Field(
        default_factory=lambda: {
            "read_file": "allow",
            "list_directory": "allow",
            "write_file": "ask",
            "run_bash": "ask",
        },
        description="Per-tool permission settings",
    )
    bash: BashPermissionConfig = Field(default_factory=BashPermissionConfig)
    allowed_mcp_servers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolve_allow_all(self) -> "ToolPermissionsConfig":
        """If allow_all is True, set default to 'allow' for consistency."""
        if self.allow_all:
            self.default = "allow"
        return self


def _default_piper_model() -> str:
    """
    Auto-detect Piper model from common installation paths.
    
    Priority order (optimized for voice latency <1.5s target):
    1. British English low-latency models (original Specialist voice)
    2. Any British English model
    3. Any English model
    4. Fallback to empty string (TTS disabled)
    """
    import glob
    from pathlib import Path
    
    # Priority 1: British English low-latency (original Specialist voice)
    for pattern in [
        str(Path.home() / ".local/share/piper/voices/en_GB-*-low.onnx"),
        str(Path.home() / ".local/share/piper/voices/en_GB*.onnx"),
    ]:
        for model_path in glob.glob(pattern):
            return str(model_path)
    
    # Priority 2: Any English model
    for pattern in [
        str(Path.home() / ".local/share/piper/voices/en_US*.onnx"),
        str(Path.home() / ".local/share/piper/voices/en_*.onnx"),
    ]:
        for model_path in glob.glob(pattern):
            return str(model_path)
    
    # Priority 3: Any model
    for pattern in [
        str(Path.home() / ".local/share/piper/voices/*.onnx"),
    ]:
        for model_path in glob.glob(pattern):
            return str(model_path)
    
    return ""


class VoiceIOConfig(BaseConfig):
    """Voice I/O configuration — mirrors VoiceIOConfig dataclass in voice_io.py."""

    # ASR
    whisper_model: str = "base"
    whisper_language: str = "en"
    whisper_compute_type: str = "int8"

    # VAD (original Specialist settings for better voice detection)
    vad_threshold: float = 0.5          # Higher = less sensitive, fewer false triggers
    vad_min_silence_ms: int = 300       # Shorter = faster response
    vad_min_speech_ms: int = 250
    vad_padding_ms: int = 250

    # TTS
    piper_model: str = Field(default="", description="Path to .onnx piper model — resolved lazily via get_piper_model()")
    piper_cmd: str = "piper"
    piper_sample_rate: int = 16000      # Default for most Piper models (check .onnx.json)

    # Audio device
    sample_rate: int = 16000            # Whisper expects 16kHz
    channels: int = 1
    blocksize: int = 512                # Silero VAD requires 512 samples

    # Behaviour
    print_transcript: bool = True
    speak_responses: bool = True
    print_responses: bool = True

    def get_piper_model(self) -> str:
        """Resolve piper model path lazily — only scans filesystem when TTS is actually needed."""
        if self.piper_model:
            return self.piper_model
        return _default_piper_model()


class HookEntry(BaseConfig):
    """A single hook registration."""
    matcher: str = Field(description="Tool name or '*' to match all tools")
    command: str = Field(description="Shell command to execute as the hook")
    # Note: blocking semantics are determined by event type, not the entry.
    # PreToolUse hooks: exit 0=allow, 1=warn, 2=block (blocking by nature)
    # PostToolUse hooks: exit code informational only (non-blocking)
    # SessionStart hooks: exit code informational only (non-blocking)


class HooksConfig(BaseConfig):
    """Hooks configuration — pre/post tool call lifecycle hooks."""
    enabled: bool = Field(default=True, description="Master switch for all hooks")
    PreToolUse: list[HookEntry] = Field(
        default_factory=list,
        description="Hooks that run before a tool call. Exit 0=allow, 1=warn, 2=block."
    )
    PostToolUse: list[HookEntry] = Field(
        default_factory=lambda: [
            HookEntry(
                matcher="*",
                command="python3 ~/.eaudev/hooks/context_monitor.py",
            ),
        ],
        description="Hooks that run after a tool call. Exit code is informational only."
    )
    SessionStart: list[HookEntry] = Field(
        default_factory=lambda: [
            HookEntry(
                matcher="*",
                command="python3 ~/.eaudev/hooks/session_start.py",
            ),
        ],
        description="Hooks that run at the start of a session."
    )


class EauDevConfig(BaseConfig):
    """Root EauDev configuration."""

    version: Literal[1] = 1
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    tool_permissions: ToolPermissionsConfig = Field(default_factory=ToolPermissionsConfig)
    voice_io: VoiceIOConfig = Field(default_factory=VoiceIOConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
