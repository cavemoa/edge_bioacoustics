# Phase 5: Solar Power Testing At Home

Phase 5 moves the Raspberry Pi edge node from bench power to off-grid solar
power while keeping it physically accessible at home. By this point the edge
software, cellular/Tailscale link, and LattePanda hub should already be working
well enough that power behavior can be tested separately.

The goal is to learn whether the Pi, microphone, storage, networking cadence,
and solar hardware can support sustained operation before taking the system into
the field.

## Scope

Phase 5 covers:

- Connecting the Pi to the intended battery and solar charging hardware.
- Reading battery and solar telemetry where available.
- Measuring power draw during inference, idle periods, and sync.
- Testing overnight operation.
- Testing multi-day operation at home.
- Confirming that data continues to sync to the LattePanda hub.
- Adjusting duty cycle, sync cadence, storage policy, or thermal management if
  needed.

## Main Workstreams

### 1. Power Telemetry

The sender should replace dummy power values with real telemetry where the
hardware exposes it. The exact fields may evolve, but the hub should be able to
track enough information to diagnose power health.

Useful values include:

- Battery voltage.
- Solar charge current.
- CPU temperature.
- CPU load.
- Disk free space.
- Uptime.
- Last successful sync.

### 2. Runtime Budget

The project should establish a practical runtime budget from observed data, not
just hardware datasheets.

Questions to answer:

- How much power does continuous Perch inference draw on the Pi?
- How much does cellular sync add?
- Does batching sync hourly still make sense?
- How much storage is retained per night with the margin gate?
- Does the Pi throttle thermally in its intended enclosure?

### 3. Home Solar Soak Tests

The Pi should run for increasingly long periods while still being easy to
retrieve and repair.

Suggested progression:

1. Several hours on battery.
2. One overnight run.
3. A full day/night cycle.
4. Several consecutive days if weather permits.

## Definition Of Done

Phase 5 is complete when:

1. The Pi can run from the planned off-grid power path at home.
2. Power telemetry reaches the hub reliably.
3. The system has survived at least one overnight run.
4. Storage growth, retained clip rate, sync size, and battery behavior are
   understood.
5. Any power-saving changes needed before field work have been identified.
