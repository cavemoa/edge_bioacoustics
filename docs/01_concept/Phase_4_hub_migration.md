# Phase 4: Hub Migration

Phase 4 moves the hub role from the Linux Mint development PC to the headless
LattePanda. The Raspberry Pi edge node has already been tested over local
networking, cellular, and Tailscale; this phase changes the hub hardware while
keeping the Pi accessible at home.

The goal is to make the eventual field hub real without losing the ability to
compare behavior against the known-good Mint hub.

## Scope

Phase 4 covers:

- Preparing the LattePanda operating system and Python environment.
- Migrating `mint_hub` into a LattePanda hub implementation.
- Running the FastAPI ingestion service on the LattePanda.
- Running the hub SQLite database on LattePanda storage.
- Running watchdog checks on the LattePanda.
- Configuring services so the hub starts after reboot.
- Comparing LattePanda hub behavior against previous Mint hub behavior.

## Explicit Non-Goals

Phase 4 does not require:

- Solar power on the Pi.
- Field deployment.
- Final unattended environmental testing.

This phase is about hub migration and headless operation.

## Main Workstreams

### 1. Hub Runtime Migration

The hub code should be moved to the LattePanda with minimal conceptual changes.
Any changes needed for paths, services, logging, storage, or networking should
be captured in configuration and documentation.

The LattePanda hub should provide:

- FastAPI ingestion.
- API key authentication.
- SQLite WAL database.
- Watchdog checks.
- Log files suitable for headless debugging.
- Backup/export paths for database and reports.

### 2. Service Management

The ingestion API and watchdog should run without an open terminal. The likely
shape is a `systemd` service for the API and a `systemd` timer or cron job for
the watchdog.

Tests should include:

- Clean boot.
- API starts automatically.
- Watchdog starts on schedule.
- Logs can be inspected after a failure.
- Database remains consistent after restart.

### 3. Pi To LattePanda Sync

The Pi sender should be pointed at the LattePanda hub over the chosen network
path. Initially this can be local or Tailscale, depending on what Phase 3 proves
most reliable.

The important check is that the same edge data model lands in the new hub
without changing edge inference behavior.

## Definition Of Done

Phase 4 is complete when:

1. The LattePanda receives Pi batches into its own hub database.
2. The API and watchdog survive clean reboot.
3. Logs, database files, and reports are easy to retrieve.
4. The Pi can continue processing if the LattePanda is offline, then sync after
   the hub returns.
5. The Mint hub can be retired as the default development hub, while remaining
   useful as a fallback comparison environment.
