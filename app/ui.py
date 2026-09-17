"""Terminal branding, colours and structured logging for SENTINEL-X.

Deliberately dependency-free: ANSI escapes only, with a hard switch for
terminals (or piped output) that cannot render them.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime

WIDTH = 66
_SEP = "=" * WIDTH
_THIN = "-" * WIDTH

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[97m",
}

_SEVERITY_COLOR = {
    "LOW": "green",
    "MEDIUM": "yellow",
    "HIGH": "red",
    "CRITICAL": "magenta",
}

_enabled = True

# With several cameras printing from their own threads, a multi-line alert
# banner must land in one piece or it becomes unreadable.
_output_lock = threading.Lock()


def emit(text: str) -> None:
    """Print atomically with respect to other camera threads."""
    with _output_lock:
        print(text)


def _enable_windows_vt() -> bool:
    """Turn on ANSI processing for legacy Windows consoles."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 == STD_OUTPUT_HANDLE, 0x4 == ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x4))
    except Exception:
        return False


def init(use_color: bool = True) -> None:
    """Configure colour output once, at startup."""
    global _enabled
    _enabled = bool(use_color) and sys.stdout.isatty() and _enable_windows_vt()


def c(text: str, *styles: str) -> str:
    if not _enabled:
        return text
    prefix = "".join(_COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{_COLORS['reset']}" if prefix else text


def severity_color(severity: str) -> str:
    return _SEVERITY_COLOR.get(severity.upper(), "white")


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(level: str, style: str, message: str) -> None:
    emit(f"{c(_stamp(), 'dim')} {c(f'[{level}]', style)} {message}")


def info(message: str) -> None:
    _emit("INFO", "cyan", message)


def ok(message: str) -> None:
    _emit("OK", "green", message)


def warn(message: str) -> None:
    _emit("WARN", "yellow", message)


def error(message: str) -> None:
    _emit("ERROR", "red", message)


def banner() -> None:
    print()
    print(c(_SEP, "cyan"))
    print(c(" SENTINEL-X  |  INTELLIGENT BORDER SURVEILLANCE".ljust(WIDTH), "bold", "cyan"))
    print(c(" Ministry of Home Affairs | Sashastra Seema Bal".ljust(WIDTH), "dim"))
    print(c(_SEP, "cyan"))
    print()


def section(title: str) -> None:
    print()
    print(c(_THIN, "dim"))
    print(c(f" {title}", "bold"))
    print(c(_THIN, "dim"))


def field(label: str, value: str, width: int = 13) -> str:
    return f"{label.ljust(width)}: {value}"
