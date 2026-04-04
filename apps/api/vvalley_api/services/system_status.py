from __future__ import annotations

import os
import resource
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_APP_START_TIME = datetime.now(timezone.utc)
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}
_CODE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".sh",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".html",
    ".css",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_uptime_seconds(now: datetime | None = None) -> float:
    current_time = now or utc_now()
    return max((current_time - _APP_START_TIME).total_seconds(), 0.0)


def get_memory_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        rss_bytes = max_rss
    else:
        rss_bytes = max_rss * 1024
    return {
        "rss_bytes": rss_bytes,
        "ru_maxrss": max_rss,
    }


def get_python_version() -> str:
    return sys.version.split()[0]


def format_timestamp(dt: datetime | None = None) -> str:
    current = dt or utc_now()
    return current.isoformat().replace("+00:00", "Z")


def _iter_project_files(root: Path):
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in _EXCLUDED_DIR_NAMES]
        current_path = Path(current_root)
        for file_name in file_names:
            yield current_path / file_name


def _extension_key(path: Path) -> str:
    return path.suffix.lower() or "[no_extension]"


def get_file_count_by_extension(root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in _iter_project_files(root):
        counts[_extension_key(path)] += 1
    return dict(sorted(counts.items()))


def get_total_lines_of_code(root: Path) -> int:
    total = 0
    for path in _iter_project_files(root):
        if path.suffix.lower() not in _CODE_EXTENSIONS:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                total += sum(1 for _ in handle)
        except (OSError, UnicodeDecodeError):
            continue
    return total


def get_last_git_commit(root: Path) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "-1",
                "--format=%H%n%s%n%cI",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 3:
        return None

    return {
        "hash": lines[0],
        "message": lines[1],
        "timestamp": lines[2],
    }


def get_health_payload() -> dict[str, Any]:
    now = utc_now()
    return {
        "status": "ok",
        "uptime": get_uptime_seconds(now),
        "memory_usage": get_memory_usage(),
        "python_version": get_python_version(),
        "timestamp": format_timestamp(now),
    }


def get_stats_payload(root: Path) -> dict[str, Any]:
    return {
        "status": "ok",
        "root": str(root),
        "file_count_by_extension": get_file_count_by_extension(root),
        "total_lines_of_code": get_total_lines_of_code(root),
        "last_git_commit": get_last_git_commit(root),
    }
