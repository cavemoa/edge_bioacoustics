# Edge Bioacoustics

Edge Bioacoustics is a project for remote, low-bandwidth acoustic monitoring of
grey-faced petrels and other North Island New Zealand birds. The system is
designed to capture field audio at the edge, run Perch 2.0 inference locally,
save only selected audio clips, and synchronize compact metadata, logits,
telemetry, and vector embeddings back to a central hub.

The core design goal is simple: keep the field node useful even when power,
storage, weather, and cellular data are all constrained.

![Bioacoustic IoT concept diagram](diagrams/bioacoustic_IOT_concept.png)

## Project Status

The project is currently in Phase 1: desktop mock development.

Phase 1 builds and tests the full software path locally before moving to real
Raspberry Pi hardware, field microphones, Tailscale networking, 4G transport,
I2C telemetry, and systemd services.

The current implementation plan is here:

[docs/02_implementation/Phase_1_implementation_plan.md](docs/02_implementation/Phase_1_implementation_plan.md)

## System Overview

The planned system has two main sides.

The edge node is a Raspberry Pi 5 in the field. It captures 48 kHz audio,
processes it as 15-second buffers, downsamples to 32 kHz for Perch 2.0, and
evaluates the three 5-second Perch frames independently. Each buffer produces
metadata, biological and noise scores, NZ bird subset scores, and three
1536-dimensional embeddings.

The central hub is a LattePanda Alpha on a secure home network. It receives
hourly MessagePack batches from the edge node, validates them with a FastAPI
ingestion service, writes them into a WAL-enabled SQLite database, and monitors
device health with a passive watchdog.

## Planned Runtime Components

The architecture is built around four scripts:

1. `bio_capture_loop.py`: edge audio buffering, Perch inference, gating, audio retention, and edge database writes.
2. `sender_daemon.py`: edge database sync, MessagePack serialization, mock or real telemetry, and HTTP transport.
3. `ingestion_api.py`: hub-side FastAPI receiver, API key authentication, payload validation, and master database insertion.
4. `watchdog_alert.py`: hub-side stale telemetry detection and alerting.

## Data Strategy

Raw recordings are treated as read-only evidence. During desktop development,
mock 48 kHz `.wav` recordings are expected to be mounted at `/data`, with the
final path still configurable.

At the edge, the system stores every buffer event and every Perch embedding in
SQLite. Full `.flac` audio is retained only when the buffer is a biological hit
or a scheduled validation sample. This preserves a useful audit trail without
trying to send raw audio over cellular.

Configuration values such as paths, thresholds, API settings, and validation
sampling cadence should live in YAML files rather than being hardcoded into
scripts.

## Repository Map

```text
diagrams/
  bioacoustic_IOT_concept.png

docs/
  00_ideas/
  01_concept/
  02_implementation/

labels/
  perch_label.csv
  north_island_nz_bird_list.csv
  north_island_nz_perch_lablel.csv

scripts/
  label_extract.py
  match_north_island_perch_labels.py
```

## Key Documents

- [Architecture concept](docs/01_concept/architecture_concept.md)
- [Phase 1 desktop development concept](docs/01_concept/Phase_1_Desktop_Development.md)
- [Phase 1 implementation plan](docs/02_implementation/Phase_1_implementation_plan.md)
- [SQLite schema ideas](docs/00_ideas/rpi_sqlite_schema.md)
- [Sound gate notes](docs/00_ideas/sound_gate.md)

## Labels

The `labels/` directory contains the Perch label list and the North Island New
Zealand bird subset used during Phase 1.

The generated file `labels/north_island_nz_perch_lablel.csv` maps Perch label
numbers to local bird common names and scientific names. It is used for the
`nz_bird_logits` field in the planned inference pipeline.

## Development Notes

Phase 1 should prove the system locally with mock data before hardware
integration. The priority is a reliable end-to-end desktop rehearsal:

1. Read mock audio from `/data`.
2. Run Perch inference and gating.
3. Write edge SQLite rows with pending sync state.
4. Send pending rows to a localhost FastAPI hub using MessagePack.
5. Mark edge rows as synced only after hub confirmation.
6. Confirm the watchdog can detect healthy, stale, and missing telemetry.

Large audio files, local databases, model exports, secrets, virtual
environments, and machine-specific YAML config files should stay out of Git.
