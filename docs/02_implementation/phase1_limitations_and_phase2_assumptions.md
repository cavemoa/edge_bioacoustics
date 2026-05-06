# Phase 1 Limitations And Phase 2 Assumptions

Phase 1 proves the software path on a desktop mock setup. It does not prove the
field hardware, network, power system, service management, or long-running
operational behavior.

## Confirmed Phase 1 Inputs

- The raw AR4 desktop mock audio path is confirmed as
  `/data/petrel_acoustics/raw_audio/doc_ar4/rapanui_AR4_june_2023`.
- That directory contains six nightly subfolders.
- The checked AR4 recordings are mono 32 kHz WAV files. The capture loop still
  supports the planned 48 kHz to 32 kHz downsampling path.
- A small repeatable teaching clip is tracked at
  `notebooks/example_audio/example1_120s_petrel.wav`.

## Known Phase 1 Limitations

- Audio capture is file-backed. No microphone, ALSA device, USB sound card, or
  Raspberry Pi audio path has been exercised yet.
- Perch runs through the CPU TensorFlow Hub/Kaggle model handle on the desktop.
  Raspberry Pi feasibility, model caching, startup time, memory use, and
  possible model format changes still need hardware testing.
- The biological/noise gate is a first-pass configurable logit threshold using
  selected Perch labels. It is useful for proving the data path, but threshold
  tuning and false-positive/false-negative analysis remain open.
- Telemetry is desktop mock telemetry. CPU temperature, battery voltage, solar
  current, and I2C sensor reads are not yet real field values.
- Transport is localhost HTTP. Tailscale, 4G behavior, link drops, latency,
  TLS/reverse proxy choices, and authentication hardening are Phase 2+ work.
- The sender is a one-shot desktop command. Backoff, retry scheduling,
  persistent failure accounting, and service supervision are not implemented.
- The watchdog prints dummy alerts only. Telegram, Discord, email, or other
  notification channels are intentionally deferred.
- Local API keys in example config are placeholders. Real secrets management is
  not part of Phase 1.
- SQLite vector storage has both `sqlite-vec` and BLOB fallback paths. Deployment
  should confirm which path is available on the target Raspberry Pi and hub.
- Raw audio upload is not part of the sync payload. Retained FLAC paths are
  stored as metadata; future phases need a deliberate retained-audio transfer
  policy if clips should leave the edge node.

## Phase 2 Hardware Assumptions

- The edge node will be a Raspberry Pi-class Linux host with enough CPU, RAM,
  storage, and cooling headroom to run bounded Perch inference.
- The hub will be a Linux host reachable from the edge node, initially using the
  same FastAPI ingestion contract proven in Phase 1.
- The edge node will run scripts under service management, likely `systemd`.
- The real audio input path will replace the Phase 1 file generator while
  preserving the 15-second buffer contract.
- The real telemetry layer will provide CPU, disk, battery, and solar values in
  the same shape used by the Phase 1 sender payload.
- Network transport will preserve the MessagePack batch contract and the rule
  that edge rows are marked `synced` only after hub acknowledgement.
- Configuration should remain YAML-driven, with machine-specific files ignored
  by Git and copied from the example templates.

## Move-To-Phase-2 Gate

Before moving to VS Code Remote SSH and hardware integration, keep the Phase 1
end-to-end report as the desktop baseline:

```text
docs/02_implementation/phase1_e2e_rehearsal_report.json
```

Any Phase 2 hardware change should be compared against that baseline: buffer
counts, vector counts, accepted IDs, sync state, telemetry freshness, and
watchdog status should remain internally consistent.
