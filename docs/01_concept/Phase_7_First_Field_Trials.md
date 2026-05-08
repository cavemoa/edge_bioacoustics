# Phase 7: First Field Trials

Phase 7 is the first controlled deployment away from home. This is not yet the
final long-term deployment; it is the first real ecological and operational test
of the system in the target environment.

The goal is to learn from a limited field trial while keeping the deployment
small, observable, and reversible.

## Scope

Phase 7 covers:

- Deploying the Raspberry Pi edge node in the field.
- Running live audio capture in the target acoustic environment.
- Operating on solar/battery power.
- Syncing over the selected remote network path.
- Receiving telemetry, detections, embeddings, and retained clip metadata at
  the LattePanda hub.
- Reviewing retained audio and gate behavior after deployment.
- Recording field lessons for the next hardware/software revision.

## Main Workstreams

### 1. Limited Deployment

The first field trial should be deliberately limited. The aim is to validate the
system in real conditions, not to maximize coverage.

Good limits might include:

- One edge node.
- A known accessible location.
- A short deployment window.
- Conservative sync and storage settings.
- Clear retrieval plan.

### 2. Ecological Review

The project should compare retained clips and gate decisions against field
expectations.

Questions to answer:

- Are grey-faced petrel calls retained?
- What environmental sounds cause false positives?
- Are excluded labels still appropriate?
- Does the margin threshold need field retuning?
- Are retained clips long enough for manual interpretation?

### 3. Operational Review

The deployment should also be reviewed as an IoT system.

Questions to answer:

- Did power survive the deployment?
- Did the network behave as expected?
- Did sync backlog remain manageable?
- Did watchdog alerts provide useful signals?
- Was retrieval and data export straightforward?

## Definition Of Done

Phase 7 is complete when:

1. The system has completed a limited field deployment.
2. Field audio, retained clips, telemetry, and hub data have been reviewed.
3. Gate behavior has been assessed against real field conditions.
4. Hardware, power, network, and enclosure lessons have been documented.
5. The next revision plan is based on field evidence rather than assumptions.
