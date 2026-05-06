"""Phase 1 hub watchdog for stale or missing edge telemetry."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from central_hub_mock.src.init_master_db import init_master_db
from mock_common.config import load_config


WatchdogStatus = Literal["healthy", "stale", "missing"]


@dataclass(frozen=True)
class WatchdogResult:
    status: WatchdogStatus
    device_id: str
    latest_timestamp_utc: str | None
    age_minutes: float | None
    stale_after_minutes: float
    message: str


def parse_utc_timestamp(value: str) -> datetime:
    """Parse the UTC timestamp strings used by the Phase 1 mock databases."""

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def default_device_id(config: dict) -> str:
    allowed_device_ids = config.get("allowed_device_ids") or []
    if allowed_device_ids:
        return str(allowed_device_ids[0])
    return str(config.get("device_id", "pi_01"))


def latest_telemetry(db_path: str | Path, device_id: str) -> sqlite3.Row | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT *
            FROM health_metrics
            WHERE device_id = ?
            ORDER BY timestamp_utc DESC, health_id DESC
            LIMIT 1;
            """,
            (device_id,),
        ).fetchone()


def check_watchdog(
    config_path: str | Path,
    *,
    device_id: str | None = None,
    now: datetime | None = None,
) -> WatchdogResult:
    config = load_config(config_path)
    init_master_db(config_path)

    watched_device_id = device_id or default_device_id(config)
    stale_after_minutes = float(config["watchdog_stale_minutes"])
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    row = latest_telemetry(config["master_db_path"], watched_device_id)

    if row is None:
        return WatchdogResult(
            status="missing",
            device_id=watched_device_id,
            latest_timestamp_utc=None,
            age_minutes=None,
            stale_after_minutes=stale_after_minutes,
            message=f"DUMMY ALERT: missing telemetry for {watched_device_id}",
        )

    latest_timestamp = parse_utc_timestamp(str(row["timestamp_utc"]))
    age = now_utc - latest_timestamp
    age_minutes = age.total_seconds() / 60.0

    if age > timedelta(minutes=stale_after_minutes):
        return WatchdogResult(
            status="stale",
            device_id=watched_device_id,
            latest_timestamp_utc=str(row["timestamp_utc"]),
            age_minutes=age_minutes,
            stale_after_minutes=stale_after_minutes,
            message=(
                f"DUMMY ALERT: stale telemetry for {watched_device_id}; "
                f"latest={row['timestamp_utc']} age_minutes={age_minutes:.1f}"
            ),
        )

    return WatchdogResult(
        status="healthy",
        device_id=watched_device_id,
        latest_timestamp_utc=str(row["timestamp_utc"]),
        age_minutes=age_minutes,
        stale_after_minutes=stale_after_minutes,
        message=(
            f"healthy: {watched_device_id} latest={row['timestamp_utc']} "
            f"age_minutes={age_minutes:.1f}"
        ),
    )


def exit_code_for_status(status: WatchdogStatus) -> int:
    return 0 if status == "healthy" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="central_hub_mock/config/hub_config.example.yaml")
    parser.add_argument("--device-id", default=None)
    args = parser.parse_args(argv)

    result = check_watchdog(args.config, device_id=args.device_id)
    print(result.message)
    return exit_code_for_status(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
