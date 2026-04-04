from __future__ import annotations

import os
import resource
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
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


@dataclass
class EndpointMetrics:
    request_count: int = 0
    total_response_time_ms: float = 0.0
    error_count: int = 0

    def record(self, response_time_ms: float, is_error: bool) -> None:
        self.request_count += 1
        self.total_response_time_ms += response_time_ms
        if is_error:
            self.error_count += 1

    def to_payload(self, endpoint: str) -> dict[str, Any]:
        average_response_time_ms = 0.0
        if self.request_count:
            average_response_time_ms = self.total_response_time_ms / self.request_count
        error_rate = 0.0
        if self.request_count:
            error_rate = self.error_count / self.request_count
        return {
            "endpoint": endpoint,
            "request_count": self.request_count,
            "average_response_time_ms": round(average_response_time_ms, 3),
            "error_count": self.error_count,
            "error_rate": round(error_rate, 6),
        }


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, EndpointMetrics] = {}

    def record(self, endpoint: str, response_time_ms: float, status_code: int) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(endpoint, EndpointMetrics())
            metrics.record(response_time_ms=response_time_ms, is_error=status_code >= 400)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._metrics[endpoint].to_payload(endpoint)
                for endpoint in sorted(self._metrics)
            ]

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


_METRICS_COLLECTOR = MetricsCollector()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def monotonic_time() -> float:
    return time.perf_counter()


def get_metrics_collector() -> MetricsCollector:
    return _METRICS_COLLECTOR


def reset_metrics_collector_for_tests() -> None:
    _METRICS_COLLECTOR.reset()


def should_track_metrics(path: str) -> bool:
    return path not in {
        "/api/metrics",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }


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


def get_metrics_payload() -> dict[str, Any]:
    now = utc_now()
    return {
        "status": "ok",
        "timestamp": format_timestamp(now),
        "endpoints": get_metrics_collector().snapshot(),
    }
