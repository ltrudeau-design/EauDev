"""EauDev custom exceptions — styled Rich error panels, no cloud deps."""

from __future__ import annotations


class EauDevError(Exception):
    """Base exception for EauDev errors. Renders as a styled Rich panel."""

    def __init__(self, message: str, title: str = "Error", role: str = "error"):
        super().__init__(message)
        self.message = "\n" + message.strip() + "\n"
        self.title = title
        self.role = role


class RequestTooLargeError(EauDevError):
    """Raised when the context window is exceeded."""

    def __init__(self):
        super().__init__(
            title="Context Window Exceeded",
            message=(
                "The conversation is too large for the model's context window.\n"
                "Run /prune to remove tool results, /compact to summarise, "
                "or /clear to start fresh."
            ),
            role="error",
        )


class ServerError(EauDevError):
    """Raised when the inference server is unreachable or returns an error."""

    def __init__(self, detail: str = ""):
        msg = "Could not reach the inference server."
        if detail:
            msg += f"\n{detail}"
        msg += "\nEnsure the server is running at the configured endpoint."
        super().__init__(title="Inference Server Error", message=msg, role="error")


class ToolDeniedError(EauDevError):
    """Raised when a tool call is denied by the permission manager."""

    def __init__(self, tool_name: str):
        super().__init__(
            title="Tool Denied",
            message=f"Permission denied for tool: {tool_name}",
            role="warning",
        )
