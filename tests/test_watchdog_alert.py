from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from central_hub_mock.src.init_master_db import init_master_db
from central_hub_mock.src.watchdog_alert import check_watchdog, exit_code_for_status


class WatchdogAlertTest(unittest.TestCase):
    def test_healthy_telemetry_returns_zero_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, db_path = self._write_config(root)
            init_master_db(config_path, reset=True)
            now = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
            self._insert_health(db_path, "pi_01", now - timedelta(minutes=10), now)

            result = check_watchdog(config_path, now=now)

        self.assertEqual(result.status, "healthy")
        self.assertEqual(exit_code_for_status(result.status), 0)
        self.assertAlmostEqual(result.age_minutes or 0, 10.0)

    def test_stale_telemetry_returns_dummy_alert_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, db_path = self._write_config(root)
            init_master_db(config_path, reset=True)
            now = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
            self._insert_health(db_path, "pi_01", now - timedelta(minutes=90), now)

            result = check_watchdog(config_path, now=now)

        self.assertEqual(result.status, "stale")
        self.assertEqual(exit_code_for_status(result.status), 1)
        self.assertIn("DUMMY ALERT", result.message)

    def test_missing_telemetry_reports_missing_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = self._write_config(root)
            init_master_db(config_path, reset=True)

            result = check_watchdog(config_path, now=datetime(2026, 5, 6, 12, 0, tzinfo=UTC))

        self.assertEqual(result.status, "missing")
        self.assertEqual(exit_code_for_status(result.status), 1)
        self.assertIn("missing telemetry", result.message)

    def test_device_override_checks_selected_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, db_path = self._write_config(root)
            init_master_db(config_path, reset=True)
            now = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
            self._insert_health(db_path, "pi_01", now - timedelta(minutes=10), now)
            self._insert_health(db_path, "pi_02", now - timedelta(minutes=100), now)

            result = check_watchdog(config_path, device_id="pi_02", now=now)

        self.assertEqual(result.status, "stale")
        self.assertEqual(result.device_id, "pi_02")

    def _write_config(self, root: Path) -> tuple[Path, Path]:
        db_path = root / "master.sqlite"
        config_path = root / "hub_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    f"master_db_path: {db_path}",
                    "api_key: test-key",
                    "allowed_device_ids:",
                    "  - pi_01",
                    "watchdog_stale_minutes: 75",
                    "embedding_dim: 1536",
                ]
            ),
            encoding="utf-8",
        )
        return config_path, db_path

    def _insert_health(
        self,
        db_path: Path,
        device_id: str,
        timestamp: datetime,
        received_at: datetime,
    ) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO health_metrics(
                    device_id, timestamp_utc, received_at_utc, cpu_temp_c,
                    cpu_load_pct, disk_free_gb, battery_voltage, solar_amps
                )
                VALUES (?, ?, ?, 45.0, 12.5, 128.0, 12.4, 0.0);
                """,
                (
                    device_id,
                    timestamp.isoformat().replace("+00:00", "Z"),
                    received_at.isoformat().replace("+00:00", "Z"),
                ),
            )
            conn.commit()


if __name__ == "__main__":
    unittest.main()
