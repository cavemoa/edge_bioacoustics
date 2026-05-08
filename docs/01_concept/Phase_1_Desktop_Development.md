# Phase 1: Desktop Development

Phase 1 is the local software proving ground. The aim is to build and revise the
core edge and hub logic on the Linux Mint development PC before introducing
Raspberry Pi hardware, live microphones, cellular networking, Tailscale, or
off-grid power.

This phase has deliberately treated the design as changeable. The project used
realistic mock audio, Perch inference, SQLite databases, and localhost hub
ingestion to discover what the actual system needed. That process replaced the
early fixed-retention gate with the current North Island NZ bird margin gate and
variable-length retained clips.

## Scope

Phase 1 covers:

- A mock edge workspace that reads recorded WAV files from disk.
- A mock hub workspace that receives MessagePack payloads over localhost.
- Perch CPU inference on 15-second compute buffers split into three 5-second
  frames.
- Margin-based bioacoustic gating against configured excluded labels.
- Variable 5, 10, or 15 second retained FLAC clips based on consecutive
  triggered frames.
- Edge and hub SQLite schemas for events, retained clips, embeddings, and
  telemetry.
- Teaching notebooks and test scripts that make the pipeline inspectable.
- Full six-night desktop test runs against the development recordings.

## Current Outcome

The Phase 1 system is now a working desktop mock rather than a paper design.
The edge capture loop, sender daemon, hub ingestion API, watchdog, notebooks,
single-file gate tester, and six-night test runner are all in place.

The important design decisions coming out of Phase 1 are:

- Retained audio is variable length, not a fixed 15 seconds.
- The gate is based on NZ bird logit margin over excluded noise labels.
- Every inference event has a stable `event_uuid` for long-term sync safety.
- YAML configuration is the contract between scripts rather than long command
  lines or hardcoded paths.
- The hub and edge database schemas should be expected to evolve during staged
  testing.

## Definition Of Done

Phase 1 is complete when:

1. The desktop mock can process representative audio into edge SQLite events,
   embeddings, retained clips, and local metrics.
2. Pending edge events can be synchronized to the local hub API using
   MessagePack.
3. The hub database can be inspected and the watchdog can evaluate telemetry.
4. The gate and retained clip behavior have been reviewed against known
   reference recordings.
5. The documentation records the limitations and open assumptions before moving
   onto physical Raspberry Pi testing.
