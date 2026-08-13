import os

from rich.console import Console

console = Console()


def _info_enabled(msg) -> bool:
    text = str(msg or "")
    if text.startswith("[CAPTION HOOK]"):
        raw = os.getenv("CAPTION_HOOK_DEBUG", "true").strip().lower()
        return raw in {"1", "true", "yes", "on"}
    return True


def info(msg):
    if _info_enabled(msg):
        console.print(f"[cyan][INFO][/cyan] {msg}")


def success(msg):
    console.print(f"[green][OK][/green] {msg}")


def warning(msg):
    console.print(f"[yellow][WARN][/yellow] {msg}")


def error(msg):
    console.print(f"[red][ERROR][/red] {msg}")
