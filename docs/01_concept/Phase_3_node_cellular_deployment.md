# Phase 3: Node Cellular Deployment

Phase 3 moves the Raspberry Pi from the local network onto cellular networking
and introduces Tailscale, while keeping the hub on the Linux Mint PC. The Pi is
still physically accessible at home, so network and operational issues can be
debugged without the pressure of a remote field deployment.

This phase answers the question: can the Pi behave like a remote edge node while
still being close enough to inspect, reboot, and reconfigure?

## Scope

Phase 3 covers:

- Connecting the Raspberry Pi through the cellular router or modem path.
- Installing and configuring Tailscale on the Pi and Mint PC.
- Moving hub traffic from direct LAN addressing to Tailscale addressing.
- Continuing to run `mint_hub` as the analysis and ingestion hub.
- Streaming or replaying audio on the Pi while syncing over cellular.
- Testing network dropouts, reconnects, delayed sync, and watchdog behavior.
- Measuring payload sizes, sync duration, and practical data usage.

## Explicit Non-Goals

Phase 3 does not require:

- Moving the hub to the LattePanda.
- Solar power.
- Field deployment.
- Final enclosure design.

Those remain intentionally out of scope so cellular/Tailscale behavior can be
understood on its own.

## Main Workstreams

### 1. Tailscale Connectivity

The Pi and Mint PC should be visible to each other over Tailscale. The hub API
should bind to an address/interface that allows Pi traffic while staying
controlled by API-key authentication.

Key questions:

- Can the Pi maintain a stable Tailscale connection over cellular?
- Are DNS names or static Tailscale IPs more reliable for this setup?
- Does the sender recover cleanly if the Tailscale session drops?
- Can the developer still SSH into the Pi for inspection and updates?

### 2. Cellular Sync Behavior

The sender should be exercised under real cellular conditions. The important
outcome is not raw speed; it is predictable, recoverable synchronization.

Measurements should include:

- Number of pending events before each sync.
- MessagePack payload size.
- Upload duration.
- API response status.
- Rows marked synced.
- Rows left pending after failures.
- Approximate cellular data use.

### 3. Watchdog And Delayed Sync

The Mint hub watchdog should distinguish between healthy, delayed, and missing
check-ins. The edge should continue processing audio locally when the cellular
link is unavailable.

Controlled tests should include:

- Stop the hub and confirm Pi rows remain pending.
- Restore the hub and confirm backlog sync.
- Disconnect cellular and confirm local processing continues.
- Wait long enough to confirm watchdog stale-device behavior.

## Definition Of Done

Phase 3 is complete when:

1. The Pi can stream or replay audio while connected through cellular.
2. The Pi can send batches to `mint_hub` over Tailscale.
3. Sync resumes correctly after network outages.
4. The watchdog gives useful signals for healthy and stale devices.
5. Cellular payload size and data-use estimates are understood well enough to
   plan longer unattended tests.
6. The system is stable enough to justify moving the hub role away from the Mint
   development PC.
