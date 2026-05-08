# Phase 2: Initial Node Development

Phase 2 is the bridge from desktop mock development to a real Raspberry Pi edge
node, while keeping the hub on the Linux Mint development PC. This avoids
changing too many variables at once: the Pi becomes real, but the network stays
local, the hub stays easy to inspect, and the first audio source is still a
folder of known recordings.

The central hub mock will be adapted into a new `mint_hub` version that runs on
the Linux Mint PC. The Raspberry Pi will process the six-night test audio from a
folder first. Later in this phase, a USB microphone can be added and the same
capture loop can be switched from file replay to live audio streaming.

## Why This Phase Exists

Phase 1 showed that the design changes as soon as real evidence is available.
Phase 2 keeps iteration cheap. The Pi runs the edge workload, but the hub,
database, plots, logs, and development tools remain on the Mint PC where they
are easy to inspect and modify.

## Scope

Phase 2 covers:

- Preparing the Raspberry Pi as a local-network edge node.
- Copying or cloning the repository onto the Pi.
- Installing dependencies in a Pi virtual environment.
- Running the edge database, Perch inference, margin gate, retained clip logic,
  and sender on the Pi.
- Creating `mint_hub` from the current `central_hub_mock` code so the Mint PC
  acts as the local hub.
- Sending MessagePack batches from the Pi to the Mint PC over the home LAN.
- Processing the six-night recording folder on the Pi as the first workload.
- Adding a USB microphone only after file replay is stable.
- Continuing to revise configuration, schema, and operational scripts as
  practical issues appear.

## Explicit Non-Goals

Phase 2 does not require:

- Cellular networking.
- Tailscale deployment.
- LattePanda hub migration.
- Solar power.
- Field enclosure work.
- Full unattended field operation.

Those are later phases.

## Main Workstreams

### 1. Pi Edge Runtime

The existing edge code will be moved from desktop mock execution to Raspberry Pi
execution. The first source will be a configured folder containing the six
nights of recordings, not a live microphone.

The Pi should produce the same kinds of artifacts as Phase 1:

- Edge SQLite events.
- Perch embeddings.
- Margin-gate scores.
- Variable retained FLAC clips.
- Local metrics and plots.
- Pending sync rows.

### 2. Mint Hub

The current `central_hub_mock` will be adapted into `mint_hub`. It should still
be simple, inspectable, and friendly for development, but it will represent the
first real hub process listening to another machine on the network.

The Mint hub should provide:

- FastAPI ingestion on a LAN address.
- API key validation.
- MessagePack payload validation.
- Hub SQLite insertion.
- Watchdog checks.
- Local analysis and reporting tools.

### 3. Local Network Transport

The sender daemon on the Pi should post to the Mint PC hub over the home LAN.
This tests the real split-device boundary without adding cellular or VPN
complexity.

Useful checks include:

- Can the Pi reach the hub reliably by hostname or IP address?
- Does the API key configuration work cleanly on both machines?
- Are batches idempotent when retried?
- Are `event_uuid` values preserved from Pi to hub?
- Can the Pi resume sync after the Mint hub is stopped and restarted?

### 4. Optional Live USB Microphone

Once file replay is stable, a USB microphone can be introduced. This should be a
controlled extension of the same capture loop, not a separate architecture.

The goal is to confirm:

- The Pi can read live audio continuously.
- The sample-rate conversion path behaves as expected.
- The gate still behaves sensibly on real room/outdoor audio.
- Retained clips are correctly written from live audio.

## Definition Of Done

Phase 2 is complete when:

1. The Raspberry Pi can process a configured folder of recordings using the
   current edge pipeline.
2. The Mint PC runs `mint_hub` and receives Pi MessagePack batches over the
   local network.
3. Edge rows are marked synced only after hub confirmation.
4. Hub database rows, embeddings, retained clips, and telemetry are inspectable
   on the Mint PC.
5. At least one failure/retry scenario has been tested on the LAN.
6. The team has enough confidence to decide whether the next change should be
   live USB audio or cellular/Tailscale networking.
