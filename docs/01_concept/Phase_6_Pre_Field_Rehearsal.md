# Phase 6: Pre-Field Operational Rehearsal

Phase 6 is a full dress rehearsal at home before the first field trial. The
system should be as close as practical to the field configuration, but still
recoverable by walking outside, power cycling hardware, swapping storage, or
opening logs directly.

The goal is confidence through repetition: the system should run unattended long
enough to reveal operational problems before it is placed at a petrel colony.

## Scope

Phase 6 covers:

- Running the Pi in its intended enclosure or near-final physical layout.
- Running live audio capture rather than only replayed files.
- Running over the intended network path.
- Running from solar/battery power.
- Syncing to the LattePanda hub.
- Testing failure and recovery scenarios.
- Writing operational checklists for deployment, retrieval, and data review.

## Main Workstreams

### 1. End-To-End Soak Test

Run the complete system continuously at home. The emphasis is not on finding
petrels; it is on proving that the operational chain behaves correctly for days
at a time.

Track:

- Uptime.
- Missed watchdog windows.
- Network reconnects.
- Sync backlog size.
- Retained audio volume.
- Disk growth.
- Battery voltage trends.
- Hub database growth.
- Any manual intervention required.

### 2. Failure Drills

Controlled failure drills should confirm the system fails in understandable and
recoverable ways.

Examples:

- Stop the hub API.
- Disconnect the network.
- Reboot the Pi.
- Reboot the LattePanda.
- Fill or nearly fill a test storage location.
- Let the watchdog alert on a deliberately stale device.

### 3. Field Readiness Materials

By the end of this phase, the project should have practical materials rather
than only code.

Useful outputs include:

- Deployment checklist.
- Retrieval checklist.
- Known-good configuration files.
- Troubleshooting notes.
- Data export procedure.
- Backup procedure.
- Expected daily storage and bandwidth range.

## Definition Of Done

Phase 6 is complete when:

1. The complete system has run unattended at home for a representative period.
2. Known failure modes have been tested and documented.
3. The deployment and retrieval process is written down.
4. The hub database and retained audio can be backed up and inspected.
5. The remaining risks are acceptable for a first limited field trial.
