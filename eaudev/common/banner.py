"""EauDev ASCII banner."""

from eaudev.constants import MODEL, VERSION

BANNER = f"""\
[blue]
███████  █████  ██    ██ ██████  ███████ ██    ██
██      ██   ██ ██    ██ ██   ██ ██      ██    ██
█████   ███████ ██    ██ ██   ██ █████   ██    ██
██      ██   ██ ██    ██ ██   ██ ██       ██  ██
███████ ██   ██  ██████  ██████  ███████   ████
[/blue]
Local AI coding agent — powered by {MODEL} on mlx_lm  [bright_black]v{VERSION}[/bright_black]"""
