# Phase 1 Revision Implementation Plan

This revision plan locks in the margin-based NZ bird gate and variable-length
retained clips as the Phase 1 design baseline. It replaces the older idea that
a positive gate should save a fixed 15-second recording. A 15-second inference
buffer remains useful because Perch 2.0 naturally evaluates three 5-second
frames, but retained audio should now be modelled as one or more 5, 10, or
15-second child clips created from consecutive triggered frames.

The aim of this plan is to clean up the codebase, documentation, notebooks, and
test reports before moving to the next phase. It is intentionally a transition
plan rather than another gate-design experiment.

## Locked Phase 1 Design

Use this vocabulary consistently from this point forward:

```text
inference buffer
  The 15-second audio chunk passed to Perch.

Perch frame
  One 5-second Perch model window inside an inference buffer.

frame bio gate
  The per-frame margin decision:
  strongest NZ bird logit minus strongest excluded-label logit >= threshold,
  unless the overall top Perch label is itself an excluded label.

retained clip
  The saved FLAC child object. It can be 5, 10, or 15 seconds in Phase 1.

buffer event
  The database row for one Perch inference call. It stores metadata, margin
  scores, and three embeddings. It is not the saved audio object.
```

Locked gate defaults:

```yaml
bio_gate_mode: nz_bird_margin
bio_margin_threshold: 0.55
excluded_margin_labels:
  - Water
  - Train
  - Vehicle
perch_window_seconds: 5.0
max_variable_buffer_frames: 3
```

Locked retained-clip policy:

```text
1 triggered 5-second frame   -> save one 5-second clip
2 consecutive triggered frames -> save one 10-second clip
3 consecutive triggered frames -> save one 15-second clip
more than 3 consecutive frames -> split into max-15-second clips
```

## Review Findings

The current code already implements most of the target behavior:

- `edge_node_mock/src/bio_capture_loop.py` scores NZ bird margin evidence,
  applies the excluded top-label veto, groups consecutive triggered frames, and
  saves sliced FLAC clips.
- `edge_node_mock/src/init_edge_db.py` and
  `central_hub_mock/src/init_master_db.py` already include retained-clip child
  tables and margin summary fields.
- `edge_node_mock/src/sender_daemon.py` includes retained clip metadata in the
  MessagePack payload.
- `central_hub_mock/src/ingestion_api.py` validates retained clips and writes
  them to the hub.
- `scripts/run_single_file_gate_test.py` has become the most useful tuning and
  validation tool for the new gate.
- Unit tests already cover margin threshold behavior, excluded-label vetoes,
  variable clip grouping, sender payloads, hub inserts, and schema creation.

At the start of this revision, the remaining problem was consistency. Several
files still preserved the old first-pass gate vocabulary or fixed-save mental
model:

- `edge_config.example.yaml` contained legacy `bio_threshold`,
  `noise_threshold`, `noise_labels`, and `biological_labels`.
- `bio_capture_loop.py`, `run_phase1_full_test.py`, and some notebooks still
  carry compatibility parameters for the old broad biological/noise gate.
- `buffer_events.filepath` is now conceptually obsolete when saved audio lives
  in `retained_audio_clips`.
- `docs/02_implementation/Phase1_test_report.md` is explicitly an old-gate
  baseline and should be superseded by a new revision test report.
- `docs/02_implementation/phase1_limitations_and_phase2_assumptions.md` still
  describes the biological/noise threshold as the current limitation.
- `notebooks/04_gate_logic_tuning.ipynb` uses experimental names such as
  `simple_bio_gate` and has a `>` threshold comparison in one cell, while the
  codebase documents and tests `>=`.
- `notebooks/04_gate_logic_tuning.ipynb` currently contains large stored
  outputs and should be cleaned before it becomes a stable teaching notebook.
- `scripts/single_file_gate_test.yaml` is machine-specific and currently
  untracked. The repo should standardize on tracked example configs plus
  ignored local configs.

## Recommended Strategy

Because this is still a desktop mock, do not spend effort supporting migrations
from old generated Phase 1 SQLite databases. Prefer a clean schema and config
contract, then regenerate the test outputs and report.

Recommended transition:

```text
1. Keep 15-second inference buffers.
2. Remove the fixed 15-second retained recording requirement.
3. Make retained_audio_clips the only authoritative location for saved FLAC
   metadata.
4. Remove old threshold gate code paths from active runtime behavior.
5. Keep historical notes only in docs/00_ideas or clearly marked archive docs.
6. Re-run the reference-file and six-night tests after cleanup.
7. Replace the old Phase1_test_report.md with a revision report generated from
   the new gate.
```

## 1. [x] Clean Up The Configuration Contract

1.1 [x] Update `edge_node_mock/config/edge_config.example.yaml` so the active
gate section contains only the margin-gate configuration.

Target active gate block:

```yaml
bio_gate_mode: nz_bird_margin
bio_margin_threshold: 0.55
excluded_margin_labels:
  - Water
  - Train
  - Vehicle
perch_window_seconds: 5.0
max_variable_buffer_frames: 3
```

1.2 [x] Remove active use of these old fields from code and examples:

```text
bio_threshold
noise_threshold
noise_labels
biological_labels
validation_sample_interval
```

1.3 [x] If a validation-sample feature is still wanted later, reintroduce it as
a new explicit `validation_retention` section. Do not preserve the old implicit
"every N dropped buffers saves the full 15 seconds" behavior.

Suggested future shape:

```yaml
validation_retention:
  enabled: false
  mode: random_frame
  interval_dropped_buffers: 100
  clip_seconds: 5.0
```

1.4 [x] Add config validation for the new gate fields.

Validation rules:

```text
bio_gate_mode must be nz_bird_margin
bio_margin_threshold must be numeric
excluded_margin_labels must be non-empty
all excluded_margin_labels must exist in perch_label.csv
perch_window_seconds must be > 0
max_variable_buffer_frames must be >= 1
```

1.5 [x] Standardize single-file test config handling.

Recommended files:

```text
scripts/single_file_gate_test.example.yaml  # tracked
scripts/single_file_gate_test.local.yaml    # ignored
```

Implemented lookup order:

```text
default script config -> scripts/single_file_gate_test.local.yaml if present,
fallback -> scripts/single_file_gate_test.yaml if present,
fallback -> scripts/single_file_gate_test.example.yaml
```

`scripts/single_file_gate_test.local.yaml` and
`scripts/single_file_gate_test.yaml` are local working files ignored by Git.

## 2. [x] Refactor Gate Logic Into A Clear Module

2.1 [x] Move reusable margin-gate logic out of `bio_capture_loop.py` into a
small dedicated module.

Suggested file:

```text
edge_node_mock/src/gate_logic.py
```

Suggested contents:

```text
FrameScores
RetentionClip
build_margin_label_indexes
score_frames
margin_gate_scores_payload
build_variable_retention_buffers
```

2.2 [x] Rename runtime fields and function arguments away from the old broad
gate language where practical.

Preferred names:

```text
bio_margin_threshold -> margin_threshold where local context is clear
max_bio_label -> max_nz_bird_label or max_nz_bird_common_name
max_bio_logit -> max_nz_bird_logit
frame_bio_gate -> frame_margin_gate_pass or frame_bio_gate
```

Implementation note: `score_frames(...)` now accepts only NZ-bird margin inputs
and no longer accepts old broad `noise_label_indexes` or `bio_label_indexes`.
`FrameScores` exposes `max_nz_bird_common_name` and `max_nz_bird_logit`
properties, while DB-facing `max_bio_*` compatibility names remain until the
schema cleanup section.

2.3 [x] Keep payload and database field names stable only when the churn is not
worth it. If a field remains for compatibility, document its revised meaning in
one place.

Current compatibility fields that need a final decision:

```text
buffer_events.max_bio_label
buffer_events.max_bio_logit
hub_buffer_events.max_bio_label
hub_buffer_events.max_bio_logit
```

Decision: keep these names stable during the module extraction, then rename or
replace them during section 4 when the clean revision schema is updated.

2.4 [x] Remove unused old-gate arguments from `decide_buffer(...)`.

Current candidates:

```text
bio_threshold
noise_threshold
validation_sample_interval
dropped_buffer_count
```

The revised decision function should consume frame scores and margin-retention
settings only.

2.5 [x] Ensure the threshold comparison is exactly:

```python
margin_gate = nz_over_excluded_margin >= bio_margin_threshold
frame_bio_gate = (not excluded_top_label_gate) and margin_gate
```

2.6 [x] Decide and document boundary behavior across 15-second inference
buffers.

Recommended Phase 1 rule:

```text
Variable retained clips are grouped only within one 15-second inference buffer.
If a call spans a buffer boundary, Phase 1 may save adjacent clips in adjacent
buffer events. Cross-buffer clip merging is deferred until real streaming
capture work.
```

## 3. Clean Up Audio Retention Behavior

3.1 [x] Make `retained_audio_clips` the only authoritative retained-audio
metadata table.

3.2 [x] Remove the full-buffer save fallback from active margin-gate runtime
paths.

Current behavior to remove from normal runtime:

```text
if decision.audio_saved and not decision.retained_clips:
    save the full buffer
```

3.3 [x] Keep `audio_saved` only as a derived convenience flag:

```text
audio_saved = retained_clip_count > 0
```

3.4 [x] Remove or deprecate parent `buffer_events.filepath` and
`hub_buffer_events.filepath`.

Recommended clean schema decision:

```text
Do not include filepath on buffer event rows in revision-created databases.
Saved audio paths live only in retained_audio_clips.filepath.
```

3.5 [x] Add one helper for retained clip naming so scripts and notebooks do not
silently drift.

Suggested function:

```python
def retained_clip_filename(device_id, timestamp, source_stem, buffer_index, clip) -> str:
    ...
```

3.6 [x] Add tests that verify saved FLAC duration and database metadata agree
within a small tolerance.

## 4. Create A Clean Revision Schema

4.1 [x] Bump the edge and hub mock schema metadata.

Suggested metadata:

```text
phase1_revision = margin_variable_retention_v1
schema_version = 3
gate_mode = nz_bird_margin
gate_threshold = 0.55
```

4.2 [x] Treat existing generated SQLite databases as disposable. Delete and
recreate them for revision tests.

4.3 [x] Update edge `buffer_events`.

Recommended clean event fields:

```text
buffer_id INTEGER PRIMARY KEY AUTOINCREMENT
event_uuid TEXT NOT NULL UNIQUE
device_id TEXT NOT NULL
source_file TEXT
file_buffer_index INTEGER
timestamp_utc TEXT NOT NULL
inference_buffer_seconds REAL NOT NULL DEFAULT 15.0
perch_window_seconds REAL NOT NULL DEFAULT 5.0
perch_frame_count INTEGER NOT NULL DEFAULT 3
audio_saved INTEGER NOT NULL DEFAULT 0
retention_reason TEXT NOT NULL CHECK(retention_reason IN ('bio_hit', 'dropped'))
max_nz_bird_common_name TEXT
max_nz_bird_scientific_name TEXT
max_nz_bird_logit REAL
max_perch_label TEXT
max_perch_logit REAL
excluded_label_scores TEXT
nz_bird_logits TEXT
gate_mode TEXT NOT NULL
gate_threshold REAL NOT NULL
gate_trigger_count INTEGER NOT NULL DEFAULT 0
retained_clip_count INTEGER NOT NULL DEFAULT 0
margin_gate_scores TEXT NOT NULL
sync_status TEXT NOT NULL DEFAULT 'pending'
created_at_utc TEXT NOT NULL
synced_at_utc TEXT
```

4.4 [x] Keep edge `retained_audio_clips` as the saved-audio child table.

Required fields:

```text
clip_id
buffer_id
retention_index
retention_reason
filepath
start_segment_index
end_segment_index
start_offset_s
end_offset_s
duration_s
triggered_frame_count
created_at_utc
```

4.5 [x] Mirror the same cleanup in hub tables:

```text
hub_buffer_events
hub_retained_audio_clips
hub_embedding_segments
hub vector storage
```

4.6 [x] Update Pydantic payload models to match the revised schema names.

4.7 [x] Parameterize retained clip validation.

Current ingestion validation hardcodes:

```text
duration_s in {5.0, 10.0, 15.0}
segment indexes between 0 and 2
```

Recommended validation:

```text
duration_s must equal triggered_frame_count * perch_window_seconds
start/end indexes must fit the frame count declared by the buffer event
triggered_frame_count must match end_segment_index - start_segment_index + 1
```

## 5. Update Sender And Hub Sync Contracts

5.1 [x] Update `sender_daemon.py` to emit the clean revision payload.

5.2 [x] Ensure the sender still includes:

```text
one detection object per inference buffer
three embedding segments per 15-second inference buffer
zero or more retained_audio_clips per detection
telemetry once per batch
```

5.3 [x] Ensure dropped buffer rows are still synced with embeddings and margin
evidence, even when no FLAC is retained.

5.4 [x] Confirm the edge only marks rows `synced` after hub acknowledgement.

5.5 [x] Add regression tests for mixed batches:

```text
one dropped buffer
one buffer with one 5-second retained clip
one buffer with one 10-second retained clip
one buffer with one 15-second retained clip
one buffer with two retained clips from non-consecutive trigger runs
```

## 6. Update Full-Test And Single-File Scripts

6.1 [x] Keep `scripts/run_single_file_gate_test.py` as the primary local
gate-validation tool.

6.2 [x] Rename script output columns where needed to match the locked terms.

Preferred frame CSV columns:

```text
top_nz_common_name
top_nz_scientific_name
top_nz_logit
top_excluded_label
top_excluded_logit
nz_over_excluded_margin
bio_margin_threshold
excluded_top_label_gate
margin_gate
frame_bio_gate
```

6.3 [x] Keep the stacked mel plot and saved-clip bars from the single-file
script. Move shared plot style defaults into a helper if the full-test runner
uses the same plot grammar.

6.4 [x] Update `scripts/run_phase1_full_test.py` to remove old-gate metrics.

Replace or rename:

```text
noise_dominant_* -> excluded_top_label_veto_* or excluded_label_evidence_*
bio_threshold -> bio_margin_threshold
noise_threshold -> removed
bio_hit_labels -> retained_nz_bird_candidates
```

6.5 [x] Update the full-test gate plots so they use the same visual language as
the single-file script:

```text
green vertical line = retained clip start
red vertical line = retained clip end
green horizontal bar = saved clip span
colors and alpha configurable through YAML or a shared style dict
```

6.6 [x] Make full-test output naming and documentation consistent:

```text
outputs/phase1_full_test/DDMMYY/run-HHMM/
phase1_full_test_metrics.json
buffer_metrics.csv
frame_metrics.csv
retained_clips.csv or gate_plot_events.csv
gate_plots/
```

6.7 [x] Add a `retained_clips.csv` to the full-test runner if
`gate_plot_events.csv` is not enough for report writing.

6.8 [x] Update the full-test CLI documentation. The current parser uses
`--edge-config`, not `--config`; docs should match the actual command.

## 7. Update Tests

7.1 [x] Keep the existing tests that already prove the new behavior.

Current useful tests include:

```text
tests/test_bio_capture_loop.py
tests/test_sender_daemon.py
tests/test_ingestion_api.py
tests/test_phase1_setup.py
tests/test_single_file_gate_test.py
```

7.2 [x] Add focused tests for the cleaned config contract:

```text
missing excluded labels fail clearly
empty NZ bird subset fails clearly
non-numeric bio_margin_threshold fails clearly
max_variable_buffer_frames < 1 fails clearly
legacy gate fields are not required
```

7.3 [x] Add retained audio duration tests:

```text
5-second clip writes expected sample count
10-second clip writes expected sample count
15-second clip writes expected sample count
source sample rate is preserved
filename contains segment span and duration
```

7.4 [x] Add database contract tests:

```text
buffer event row count equals processed inference buffers
embedding row count equals buffer events * 3
retained clip count equals retained_audio_clips rows
parent filepath column is absent or always null, depending on final schema
foreign keys cascade from buffer events to retained clips and embeddings
```

7.5 [x] Add hub validation tests for bad retained clip metadata:

```text
retained_clip_count mismatch
audio_saved mismatch
invalid segment span
invalid duration for configured frame count
unexpected payload fields
```

7.6 [x] Add full-test runner unit coverage for retained clip CSV/plot events.

## 8. Update Teaching Notebooks

8.1 [x] Update `notebooks/01_edge_capture_walkthrough.ipynb`.

Required changes:

```text
use locked vocabulary: inference buffer, Perch frame, retained clip
show margin gate, not broad biological/noise thresholding
show retained_audio_clips as the saved-audio table
avoid saying retained audio is a saved buffer
```

8.2 [x] Update `notebooks/03_end_to_end_system_walkthrough.ipynb`.

Required changes:

```text
show edge DB buffer_events plus retained_audio_clips
show hub DB hub_buffer_events plus hub_retained_audio_clips
show one dropped event and one retained event if practical
```

8.3 [x] Update `notebooks/04_gate_logic_tuning.ipynb`.

Required changes:

```text
rename EXCLUDED_TOP3_LABELS to EXCLUDED_MARGIN_LABELS
rename simple_bio_gate to frame_bio_gate
change margin comparison from > to >=
keep BIO_MARGIN_THRESHOLD = 0.55 as the default
clear large stored outputs before committing
keep the variable-buffer subplot
```

8.4 [x] Consider adding a short notebook note that the full 15-second inference
buffer is a compute unit, while saved clips are child spans.

8.5 [ ] Run the notebooks in order after cleanup and confirm they write
inspectable artifacts under `notebooks/output/`.

8.6 [x] Strip or minimize notebook outputs before committing unless a small
teaching output is intentionally retained.

## 9. Update Documentation

9.1 [x] Update `README.md`.

Required changes:

```text
make margin gate the only current gate described
remove broad noise/bio threshold wording from active sections
explain retained clips as child rows
remove or qualify parent filepath in the schema diagram
add retained_clips.csv to full-test outputs if implemented
add single-file local/example YAML guidance
```

9.2 [x] Update `docs/01_concept/architecture_concept.md`.

Required changes:

```text
confirm 15-second ring buffer is for inference only
confirm retained audio is variable 5/10/15-second clips
describe excluded-label margin gate as the current design
```

9.3 [x] Update `docs/02_implementation/schema_decisions.md`.

Required changes:

```text
document buffer_events as inference events
document retained_audio_clips as the saved-audio authority
document margin_gate_scores
document schema version 3 if adopted
document that old duplicate filepath decisions are superseded
```

9.4 [x] Update `docs/02_implementation/phase1_limitations_and_phase2_assumptions.md`.

Required changes:

```text
remove the statement that the current gate is a first-pass biological/noise
threshold
add that field testing still needs hardware validation and real false-negative
review
add that cross-buffer retained clip merging is deferred
```

9.5 [x] Archive or clearly label `docs/02_implementation/Phase1_test_report.md`
as the old broad-label gate baseline until the new report replaces it.

Recommended approach:

```text
rename old report to Phase1_test_report_legacy_broad_gate.md
write a new Phase1_test_report.md from the revision six-night run
```

9.6 [x] Leave `docs/00_ideas/sound_gate.md` and
`docs/00_ideas/rpi_sqlite_schema.md` as historical notes, but add a short note
at the top pointing readers to the current revision docs.

## 10. Regenerate Reports And Evidence

Status note: code, schema, scripts, notebooks, and documentation have been
updated for the revision. The evidence-generation tasks below intentionally
remain unchecked until the controlled reference tests and full six-night run are
executed against the real audio/model environment.

10.1 [ ] Run a controlled single-file reference test.

Recommended first command:

```bash
.venv/bin/python scripts/run_single_file_gate_test.py \
  --config scripts/single_file_gate_test.local.yaml
```

Recommended reference sections:

```text
20230527_213004.wav, 570s to 700s
20230527_213004.wav, 300s to 700s
one or two extra files already manually reviewed by the user
```

Acceptance checks:

```text
known call spans trigger
obvious excluded-label spans are vetoed
retained clips are 5, 10, or 15 seconds only
plot bars line up with retained_clips.csv
summary.json records threshold, excluded labels, retained seconds, and counts
```

10.2 [ ] Run the full six-night revision test.

Recommended command:

```bash
.venv/bin/python scripts/run_phase1_full_test.py \
  --edge-config edge_node_mock/config/edge_config.local.yaml \
  --hub-config central_hub_mock/config/hub_config.local.yaml \
  --reset-output \
  --sync-to-hub
```

10.3 [ ] Generate the new report from the full-test output.

New report should cover:

```text
speed and performance
retained clip count and total retained seconds
storage reduction versus fixed 15-second saves
5/10/15-second clip distribution
frame gate triggers
buffer events with retained clips
excluded-label veto activations
top NZ bird candidates among retained frames
edge and hub row-count invariants
watchdog and sync status
known remaining uncertainty
```

10.4 [ ] Verify database invariants after the full run:

```text
edge buffer_events = processed inference buffers
edge embedding_segments = edge buffer_events * 3
edge vector rows = edge embedding_segments
edge retained_audio_clips = retained clip count
hub buffer rows = edge buffer rows after sync
hub retained clips = edge retained clips after sync
hub embedding rows = edge embedding rows after sync
edge pending rows = 0 after successful sync
```

10.5 [ ] Compare retained storage against the old fixed-save baseline.

Useful comparisons:

```text
old retained seconds if saving full triggered buffers
new retained seconds from variable clips
seconds saved
percent reduction
FLAC bytes saved if audio writing is enabled
```

## 11. Repository Hygiene Before Phase 2

11.1 [x] Decide which new files should be tracked.

Recommended tracked files:

```text
scripts/run_single_file_gate_test.py
scripts/single_file_gate_test.example.yaml
tests/test_single_file_gate_test.py
```

Recommended ignored local files:

```text
scripts/single_file_gate_test.local.yaml
scripts/single_file_gate_test.yaml if kept as local shorthand
```

11.2 [x] Confirm `outputs/`, generated databases, and retained FLACs remain
ignored.

11.3 [x] Confirm the bundled teaching audio remains intentionally unignored:

```text
notebooks/example_audio/example1_120s_petrel.wav
```

11.4 [x] Clear notebook outputs that make the repo unnecessarily large,
especially `notebooks/04_gate_logic_tuning.ipynb`.

11.5 [x] Run formatting or syntax checks:

```bash
.venv/bin/python -m py_compile \
  edge_node_mock/src/bio_capture_loop.py \
  edge_node_mock/src/sender_daemon.py \
  central_hub_mock/src/ingestion_api.py \
  scripts/run_single_file_gate_test.py \
  scripts/run_phase1_full_test.py

.venv/bin/python -m pytest

git diff --check
```

## 12. Completion Criteria

The Phase 1 revision is complete when all of the following are true:

```text
The active code path uses only the NZ bird margin gate.
No active runtime path requires saving a fixed 15-second retained recording.
Retained FLAC metadata lives in retained_audio_clips and hub_retained_audio_clips.
The single-file script and notebook agree on threshold behavior.
The full-test runner reports variable retained clip metrics.
The README, schema docs, limitations doc, and architecture doc use the same vocabulary.
The old Phase1_test_report is archived or replaced by a revision report.
All unit tests pass.
The controlled reference-file tests pass manual review.
The six-night revision test completes and preserves edge-to-hub row invariants.
```

## Suggested Implementation Order

Work in this order to reduce churn:

```text
1. Config cleanup and terminology decisions.
2. Gate logic module extraction and threshold alignment.
3. Runtime removal of old fixed full-buffer save paths.
4. Clean schema and payload revision.
5. Test updates.
6. Single-file and full-test runner cleanup.
7. Notebook updates and output clearing.
8. Documentation updates.
9. Reference-file validation.
10. Full six-night revision run and new report.
```

This leaves Phase 2 with a cleaner contract: 15-second inference events produce
three embeddings and zero or more variable retained clips. That contract should
be much easier to carry onto Raspberry Pi hardware than the older fixed-save
retention design.
