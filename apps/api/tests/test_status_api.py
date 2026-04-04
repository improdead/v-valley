#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
TEST_DB_DIR = tempfile.TemporaryDirectory(dir=ROOT)
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_status_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
from apps.api.vvalley_api.services.system_status import (
    MetricsCollector,
    reset_metrics_collector_for_tests,
)
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend
from apps.api.vvalley_api.storage.scenarios import reset_backend_cache_for_tests as reset_scenarios_backend


class StatusApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        reset_scenarios_backend()
        reset_rate_limiter()
        reset_metrics_collector_for_tests()
        self.client = TestClient(app)

    def test_api_health_returns_extended_status(self) -> None:
        now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        with patch("apps.api.vvalley_api.main.get_health_payload") as mock_health:
            mock_health.return_value = {
                "status": "ok",
                "uptime": 12.5,
                "memory_usage": {"rss_bytes": 4096, "ru_maxrss": 4},
                "python_version": "3.12.0",
                "timestamp": now.isoformat().replace("+00:00", "Z"),
            }

            resp = self.client.get("/api/health")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        self.assertEqual(resp.json()["uptime"], 12.5)
        self.assertEqual(resp.json()["memory_usage"]["rss_bytes"], 4096)
        self.assertEqual(resp.json()["python_version"], "3.12.0")
        self.assertEqual(resp.json()["timestamp"], "2025-01-02T03:04:05Z")
        mock_health.assert_called_once_with()

    def test_api_stats_returns_project_summary(self) -> None:
        with patch("apps.api.vvalley_api.main.get_stats_payload") as mock_stats:
            mock_stats.return_value = {
                "status": "ok",
                "root": str(ROOT),
                "file_count_by_extension": {".py": 10, ".md": 2},
                "total_lines_of_code": 321,
                "last_git_commit": {
                    "hash": "abc123",
                    "message": "Add status endpoints",
                    "timestamp": "2025-01-02T03:04:05Z",
                },
            }

            resp = self.client.get("/api/stats")

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["root"], str(ROOT))
        self.assertEqual(payload["file_count_by_extension"][".py"], 10)
        self.assertEqual(payload["total_lines_of_code"], 321)
        self.assertEqual(payload["last_git_commit"]["hash"], "abc123")
        mock_stats.assert_called_once_with(ROOT)

    def test_metrics_collector_tracks_counts_average_and_errors(self) -> None:
        collector = MetricsCollector()

        collector.record("GET /api/health", response_time_ms=10.0, status_code=200)
        collector.record("GET /api/health", response_time_ms=20.0, status_code=500)
        collector.record("GET /api/stats", response_time_ms=5.0, status_code=200)

        payload = collector.snapshot()

        self.assertEqual(
            payload,
            [
                {
                    "endpoint": "GET /api/health",
                    "request_count": 2,
                    "average_response_time_ms": 15.0,
                    "error_count": 1,
                    "error_rate": 0.5,
                },
                {
                    "endpoint": "GET /api/stats",
                    "request_count": 1,
                    "average_response_time_ms": 5.0,
                    "error_count": 0,
                    "error_rate": 0.0,
                },
            ],
        )

    def test_api_metrics_reports_tracked_requests(self) -> None:
        with patch("apps.api.vvalley_api.main.monotonic_time", side_effect=[1.0, 1.010, 2.0, 2.030]):
            health_resp = self.client.get("/api/health")
            missing_resp = self.client.get("/does-not-exist")

        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(missing_resp.status_code, 404)

        resp = self.client.get("/api/metrics")

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("timestamp", payload)
        self.assertEqual(
            payload["endpoints"],
            [
                {
                    "endpoint": "GET /api/health",
                    "request_count": 1,
                    "average_response_time_ms": 10.0,
                    "error_count": 0,
                    "error_rate": 0.0,
                },
                {
                    "endpoint": "GET /does-not-exist",
                    "request_count": 1,
                    "average_response_time_ms": 30.0,
                    "error_count": 1,
                    "error_rate": 1.0,
                },
            ],
        )

    def test_api_metrics_increments_request_counts_for_repeated_endpoint_calls(self) -> None:
        with patch(
            "apps.api.vvalley_api.main.monotonic_time",
            side_effect=[1.0, 1.010, 2.0, 2.020, 3.0, 3.030],
        ):
            first = self.client.get("/api/health")
            second = self.client.get("/api/health")
            metrics = self.client.get("/api/metrics")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(
            metrics.json()["endpoints"],
            [
                {
                    "endpoint": "GET /api/health",
                    "request_count": 2,
                    "average_response_time_ms": 15.0,
                    "error_count": 0,
                    "error_rate": 0.0,
                },
            ],
        )

    def test_stats_payload_helpers_skip_excluded_directories(self) -> None:
        from apps.api.vvalley_api.services.system_status import get_file_count_by_extension, get_total_lines_of_code

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            temp_root = Path(td)
            (temp_root / "src").mkdir()
            (temp_root / ".git").mkdir()
            (temp_root / "node_modules").mkdir()

            (temp_root / "src" / "main.py").write_text("print('hi')\nprint('bye')\n", encoding="utf-8")
            (temp_root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (temp_root / ".git" / "ignored.py").write_text("raise SystemExit\n", encoding="utf-8")
            (temp_root / "node_modules" / "ignored.js").write_text("console.log('x')\n", encoding="utf-8")

            file_counts = get_file_count_by_extension(temp_root)
            total_lines = get_total_lines_of_code(temp_root)

        self.assertEqual(file_counts[".py"], 1)
        self.assertEqual(file_counts[".md"], 1)
        self.assertEqual(total_lines, 3)

    def test_last_git_commit_returns_none_when_git_fails(self) -> None:
        from apps.api.vvalley_api.services.system_status import get_last_git_commit

        with patch("apps.api.vvalley_api.services.system_status.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(get_last_git_commit(ROOT))
