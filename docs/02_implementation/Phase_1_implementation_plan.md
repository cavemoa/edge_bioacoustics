# Phase 1 Implementation Plan: Desktop Mock Development

This plan translates the Phase 1 desktop concept into a checkable implementation
sequence. The aim is to prove the complete edge-to-hub data path locally before
using real microphones, Raspberry Pi hardware, 4G networking, Tailscale, I2C
telemetry, or systemd services.

Phase 1 builds four scripts:

1. `bio_capture_loop.py`: mock audio capture, Perch inference, gating, and edge DB writes.
2. `sender_daemon.py`: edge DB query, mock telemetry, MessagePack transport, and sync-state updates.
3. `ingestion_api.py`: localhost FastAPI receiver, API key auth, MessagePack validation, and master DB writes.
4. `watchdog_alert.py`: central dead-man check against hub health metrics.

## Working Assumptions

1. [x] The desktop phase uses two local workspaces or clearly separated folders:
   `edge_node_mock/` and `central_hub_mock/`.
2. [x] The local edge database is treated as the Raspberry Pi database.
3. [x] The local master database is treated as the LattePanda hub database.
4. [x] Audio files are never modified in place. Raw mock recordings mounted at `/data` are treated as read-only evidence.
5. [ ] The edge database stores every 15-second buffer event and all Perch embeddings.
6. [ ] Full `.flac` audio is retained only for biological hits and validation samples.
7. [x] Perch 2.0 embeddings are confirmed as `1536` dimensions for this project.
8. [x] Label CSVs live in the repository `labels/` directory.
9. [x] `labels/north_island_nz_perch_lablel.csv` is the local NZ bird label subset used for `nz_bird_logits` during Phase 1.
10. [x] Mock raw recording files are confirmed at `/data/petrel_acoustics/raw_audio/doc_ar4/rapanui_AR4_june_2023`, with six nightly subfolders and `**/*.wav` matching.
11. [x] User-configurable parameters should live in YAML config files wherever practical. Command-line arguments should be kept short and mostly limited to selecting a config file, bounded test runs, or dry-run behavior.

## 1. [x] Prepare The Desktop Mock Environment

1.1 [x] Create or confirm the local folder layout.

Recommended structure:

```text
edge_node_mock/
  data/
    db/
    retained_audio/
  config/
  src/
  tests/

central_hub_mock/
  data/
    db/
  config/
  src/
  tests/
```

1.2 [x] Use the repo root `.venv` for Phase 1 desktop development unless separate edge/hub environments become necessary later.

1.3 [x] Add local-only folders to `.gitignore` if they do not already exist.

Minimum ignored paths:

```text
edge_node_mock/data/
central_hub_mock/data/
*.sqlite
*.sqlite-shm
*.sqlite-wal
*.flac
*.wav
```

1.4 [x] Add repo dependency manifests for the desktop dependencies.

Initial dependency set:

```text
pandas
numpy
scipy
soundfile
tensorflow
pyyaml
sqlite-vec
msgpack
requests
fastapi
uvicorn
pydantic
psutil
pytest
```

1.5 [x] Add YAML config templates for each side.

Recommended files:

```text
edge_node_mock/config/edge_config.example.yaml
central_hub_mock/config/hub_config.example.yaml
```

Local machine-specific copies should be ignored by Git:

```text
edge_node_mock/config/edge_config.local.yaml
central_hub_mock/config/hub_config.local.yaml
```

Edge config should include:

```yaml
device_id: pi_01
edge_db_path: edge_node_mock/data/db/edge_mock.sqlite
retained_audio_dir: edge_node_mock/data/retained_audio
raw_audio_mount: /data
raw_audio_glob: "*.wav"
perch_model_path: null
perch_label_path: labels/perch_label.csv
nz_bird_label_path: labels/north_island_nz_perch_lablel.csv
embedding_dim: 1536
hub_ingest_url: http://127.0.0.1:8000/ingest_batch
api_key: replace-with-local-dev-key
bio_threshold: 2.0
noise_threshold: 5.0
validation_sample_interval: 100
```

Hub config should include:

```yaml
master_db_path: central_hub_mock/data/db/master_mock.sqlite
api_key: replace-with-local-dev-key
allowed_device_ids:
  - pi_01
watchdog_stale_minutes: 75
embedding_dim: 1536
```

1.6 [x] Gather `.wav` raw recording files for the mock data stream.

1.7 [x] Mount the raw recordings so the edge mock can read them from `/data`.

1.8 [x] Keep the final `/data` path configurable in YAML until the final mount layout is known.

1.9 [x] Add a small smoke script or test that confirms at least one configured `.wav` can be read, has one or two channels, has an accepted source sample rate, and can be converted to mono float32.

Deliverable:

1.10 [x] Both mock workspaces exist, dependency manifests exist, YAML config templates exist, and at least one configured mock audio file can be loaded.

Test:

1.11 [x] Run the environment smoke test and confirm it prints the configured audio search path, first selected file, sample rate, duration, channel count, dtype, and peak amplitude.

## 2. [x] Define And Initialize The SQLite Schemas

2.1 [x] Create a shared schema decision note in the implementation docs or in code comments.

The Phase 1 schema should resolve the duplicate `audio_saved`, `retention_reason`,
and `filepath` fields currently shown in `rpi_sqlite_schema.md`.

2.2 [x] Implement an edge DB setup script.

Suggested file:

```text
edge_node_mock/src/init_edge_db.py
```

2.3 [x] Implement the edge `buffer_events` table.

Required fields:

```text
buffer_id INTEGER PRIMARY KEY AUTOINCREMENT
device_id TEXT NOT NULL
timestamp_utc TEXT NOT NULL
audio_saved INTEGER NOT NULL DEFAULT 0
retention_reason TEXT NOT NULL
filepath TEXT
max_bio_label TEXT
max_bio_logit REAL
noise_logits TEXT
max_perch_label TEXT
max_perch_logit REAL
nz_bird_logits TEXT
sync_status TEXT NOT NULL DEFAULT 'pending'
created_at_utc TEXT NOT NULL
synced_at_utc TEXT
```

2.4 [x] Add a `CHECK` constraint or code-level validation for `retention_reason`.

Allowed values:

```text
bio_hit
validation_sample
dropped
```

2.5 [x] Add a `CHECK` constraint or code-level validation for `sync_status`.

Allowed values:

```text
pending
in_flight
synced
failed
```

2.6 [x] Implement the edge `embedding_segments` table.

Required fields:

```text
embedding_id INTEGER PRIMARY KEY AUTOINCREMENT
buffer_id INTEGER NOT NULL
segment_index INTEGER NOT NULL
FOREIGN KEY(buffer_id) REFERENCES buffer_events(buffer_id) ON DELETE CASCADE
```

2.7 [x] Implement the edge `perch_vectors` table.

Preferred form:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS perch_vectors USING vec0(
    embedding_id INTEGER PRIMARY KEY,
    embedding float[1536]
);
```

2.8 [x] Add a fallback decision for development machines without `sqlite-vec`.

Acceptable Phase 1 fallback:

```text
perch_vector_blobs(embedding_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)
```

2.9 [x] Implement a master DB setup script.

Suggested file:

```text
central_hub_mock/src/init_master_db.py
```

2.10 [x] Enable WAL mode on the master database.

Required pragma:

```sql
PRAGMA journal_mode=WAL;
```

2.11 [x] Create master tables for received buffer events, embedding segments, vector storage, ingestion batches, and health metrics.

Required `ingestion_batches` fields:

```text
batch_id INTEGER PRIMARY KEY AUTOINCREMENT
device_id TEXT NOT NULL
sent_at_utc TEXT NOT NULL
received_at_utc TEXT NOT NULL
payload_bytes INTEGER NOT NULL
detection_count INTEGER NOT NULL
status TEXT NOT NULL
```

Required `hub_buffer_events` fields:

```text
hub_buffer_id INTEGER PRIMARY KEY AUTOINCREMENT
device_id TEXT NOT NULL
source_buffer_id INTEGER NOT NULL
batch_id INTEGER NOT NULL
timestamp_utc TEXT NOT NULL
audio_saved INTEGER NOT NULL
retention_reason TEXT NOT NULL
filepath TEXT
max_bio_label TEXT
max_bio_logit REAL
noise_logits TEXT
max_perch_label TEXT
max_perch_logit REAL
nz_bird_logits TEXT
received_at_utc TEXT NOT NULL
UNIQUE(device_id, source_buffer_id)
```

Required `hub_embedding_segments` fields:

```text
hub_embedding_id INTEGER PRIMARY KEY AUTOINCREMENT
hub_buffer_id INTEGER NOT NULL
source_embedding_id INTEGER
segment_index INTEGER NOT NULL
FOREIGN KEY(hub_buffer_id) REFERENCES hub_buffer_events(hub_buffer_id) ON DELETE CASCADE
```

Required hub vector table:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS hub_perch_vectors USING vec0(
    hub_embedding_id INTEGER PRIMARY KEY,
    embedding float[1536]
);
```

Required `health_metrics` fields:

```text
health_id INTEGER PRIMARY KEY AUTOINCREMENT
device_id TEXT NOT NULL
timestamp_utc TEXT NOT NULL
received_at_utc TEXT NOT NULL
cpu_temp_c REAL
cpu_load_pct REAL
disk_free_gb REAL
battery_voltage REAL
solar_amps REAL
```

2.12 [x] Ensure the master schema preserves the original edge `buffer_id` and `device_id` so rows can be traced back to the source node.

Deliverable:

2.13 [x] Running the setup scripts creates both SQLite databases from scratch.

Test:

2.14 [x] Run a schema test that opens both databases, lists all expected tables, confirms WAL mode on the master DB, confirms vector tables use `1536` dimensions, and inserts then deletes one dummy buffer event with three embedding segments.

## 3. [x] Confirm Perch Model Inputs, Outputs, And Labels

3.1 [x] Create a model inspection script.

Suggested file:

```text
edge_node_mock/src/inspect_perch_model.py
```

3.2 [x] Load the configured Perch 2.0 TensorFlow model.

3.3 [x] Load the Perch labels CSV.

3.4 [x] Run one known 15-second mono float32 buffer through the model.

3.5 [x] Confirm the model splits the 15-second buffer into three 5-second frames.

Expected output shape:

```text
logits:     [3, label_count]
embeddings: [3, embedding_dim]
```

3.6 [x] Set `embedding_dim: 1536` in the edge and hub YAML config templates.

3.7 [x] Confirm the Perch labels can be indexed by numeric label number.

3.8 [x] Confirm `labels/north_island_nz_perch_lablel.csv` is the NZ bird label subset for `nz_bird_logits`.

3.9 [x] Confirm `labels/north_island_nz_perch_lablel.csv` can be loaded and mapped from `perch_label_number` to `common_name` and `scientific_name`.

Deliverable:

3.10 [x] A model inspection output file or console report documents `label_count`, `embedding_dim = 1536`, and the three-frame behavior.

Test:

3.11 [x] Run the inspection script and confirm the installed model output matches the YAML value `embedding_dim: 1536` before continuing.

## 4. [x] Build The Mock Capture And Gating Pipeline

4.1 [x] Implement `bio_capture_loop.py` in the edge mock workspace.

4.2 [x] Replace real microphone capture with a mock generator that reads configured `.wav` files from `/data`.

4.3 [x] Keep the mock audio input path and filename pattern in YAML rather than hardcoding them.

4.4 [x] Convert source audio to mono float32.

4.5 [x] Downsample 48 kHz source audio to 32 kHz using `scipy.signal.resample_poly` with `up=2` and `down=3`.

4.6 [x] Slice or stream the raw recordings into 15-second buffers.

4.7 [x] Pass each 15-second buffer into Perch inference.

4.8 [x] Evaluate the three 5-second frames independently.

Per-frame values to compute:

```text
max noise label and score
max biological label and score
max Perch label and score
top 3 NZ bird labels and scores
```

4.9 [x] Implement the first-pass gate.

Suggested Phase 1 logic:

```text
bio_hit if any frame has max_bio_logit >= bio_threshold
dropped if noise dominates and no biological frame passes
validation_sample every N dropped buffers
```

4.10 [x] Keep threshold values and validation-sample cadence in YAML.

4.11 [x] Save `.flac` audio only when the buffer is a `bio_hit` or `validation_sample`.

4.12 [x] Insert one `buffer_events` row for every 15-second buffer.

4.13 [x] Insert exactly three `embedding_segments` rows for every buffer.

4.14 [x] Insert exactly three vector rows for every buffer.

4.15 [x] Store all JSON-like score fields as compact JSON strings.

4.16 [x] Insert new buffer rows with `sync_status = 'pending'`.

4.17 [x] Add a short command-line override for bounded desktop testing, such as `--iterations 3`.

4.18 [x] Add a short `--config` option so the normal run command points at a YAML file instead of carrying long parameter lists.

Deliverable:

4.19 [x] The mock capture loop can process a fixed number of 15-second buffers from the configured `/data` source and populate the edge database without network access.

Test:

4.20 [x] Run `bio_capture_loop.py --config edge_node_mock/config/edge_config.local.yaml --iterations 3`.

4.21 [x] Query the edge DB and confirm:

```text
3 buffer_events rows
9 embedding_segments rows
9 vector rows
all buffer_events.sync_status = 'pending'
retained audio files exist only for bio_hit or validation_sample rows
```

## 4A. [x] Build Teaching Notebooks For The Mock Pipeline

4A.1 [x] Create teaching notebooks in the root `notebooks/` directory.

Files:

```text
notebooks/01_edge_capture_walkthrough.ipynb
notebooks/02_hub_ingestion_walkthrough.ipynb
notebooks/03_end_to_end_system_walkthrough.ipynb
```

4A.2 [x] Explain the purpose of `bio_capture_loop.py` for a new student.

4A.3 [x] Show how YAML config values drive paths, thresholds, and label groups.

4A.4 [x] Demonstrate streaming the bundled 120-second example `.wav` recording into eight 15-second buffers.

4A.5 [x] Show how a 15-second buffer becomes three 5-second Perch input windows.

4A.6 [x] Run Perch inference and display logits, embedding shapes, and per-buffer inference timing.

4A.7 [x] Demonstrate frame scoring for noise labels, biological labels, max Perch label, and top NZ bird labels.

4A.8 [x] Walk through the first-pass retention gate and its JSON fields.

4A.9 [x] Include example code for saving retained `.flac` audio.

4A.10 [x] Include scratch database and payload generation so students can inspect `buffer_events`, `embedding_segments`, vector row counts, retained `.flac` files, and MessagePack transport files without altering the main edge DB.

4A.11 [x] Add a hub walkthrough notebook that explains `ingestion_api.py`, `init_master_db.py`, and `watchdog_alert.py` using the payload generated by the edge notebook.

4A.12 [x] Add an end-to-end notebook that runs capture, starts a temporary localhost hub API, sends pending edge rows, and inspects the resulting hub database.

Deliverable:

4A.13 [x] Commented notebooks teach the capture-loop, hub-ingestion, and end-to-end flows step by step using the same functions as the production scripts.

Test:

4A.14 [x] Validate the notebook JSON and confirm the notebooks contain runnable code cells for the edge, hub, and end-to-end walkthroughs.

## 5. [x] Build The Local Hub Ingestion API

5.1 [x] Implement `ingestion_api.py` in the central hub mock workspace.

5.2 [x] Expose a localhost FastAPI endpoint:

```text
POST /ingest_batch
```

5.3 [x] Load hub settings from YAML rather than hardcoding the master DB path, API key, device allowlist, or embedding dimension.

Suggested desktop pattern:

```powershell
$env:HUB_CONFIG = "central_hub_mock/config/hub_config.local.yaml"
uvicorn ingestion_api:app --host 127.0.0.1 --port 8000
```

5.4 [x] Require an `X-API-Key` header.

5.5 [x] Reject requests with a missing or incorrect API key.

5.6 [x] Accept raw MessagePack request bodies.

5.7 [x] Decode MessagePack into a Python dictionary.

5.8 [x] Validate decoded payloads with Pydantic models before writing to SQLite.

Required top-level payload fields:

```text
device_id
sent_at_utc
detections
telemetry
```

5.9 [x] Validate each detection includes the source `buffer_id`, metadata fields, three embedding segments, and binary embeddings.

5.10 [x] Validate telemetry includes desktop mock values for:

```text
timestamp_utc
cpu_temp_c
cpu_load_pct
disk_free_gb
battery_voltage
```

5.11 [x] Insert accepted detections and telemetry into the master WAL-enabled database in one transaction.

5.12 [x] Return a response containing accepted source buffer IDs.

Suggested response:

```json
{
  "status": "ok",
  "accepted_buffer_ids": [1, 2, 3]
}
```

Deliverable:

5.13 [x] The API receives, authenticates, validates, and stores a hand-built MessagePack batch on localhost using YAML configuration.

Test:

5.14 [x] Set `HUB_CONFIG` to the local YAML file and run `uvicorn ingestion_api:app --host 127.0.0.1 --port 8000`.

5.15 [x] Send one valid test payload and confirm `200 OK`.

5.16 [x] Send one payload with a bad API key and confirm `401` or `403`.

5.17 [x] Send one malformed payload and confirm `422` or a controlled validation error.

5.18 [x] Query the master DB and confirm the valid payload inserted exactly one batch, one telemetry row, and the expected detection/vector rows.

## 6. [x] Build The Mock Sender Daemon

6.1 [x] Implement `sender_daemon.py` in the edge mock workspace.

6.2 [x] Query the edge database for `sync_status = 'pending'`.

6.3 [x] Load each pending buffer with its three embedding segments and three vector blobs.

6.4 [x] Gather desktop mock telemetry.

Suggested values:

```text
cpu_temp_c = 45.0
battery_voltage = 12.4
```

Real desktop-safe values may be gathered with `psutil` for CPU load and disk free space.

6.5 [x] Build the MessagePack payload using binary embedding data directly.

6.6 [x] POST to the configured localhost API endpoint with the `X-API-Key` header.

6.7 [x] On `200 OK`, update only the accepted source buffer IDs to `sync_status = 'synced'` and set `synced_at_utc`.

6.8 [x] On network failure, API failure, or validation failure, leave rows as `pending` or mark them `failed` with a clear logged reason.

6.9 [x] Add a `--dry-run` mode that prints payload counts without sending.

6.10 [x] Add a `--limit` option to send a small batch during desktop testing.

6.11 [x] Add a short `--config` option so API URL, API key, database path, and telemetry defaults come from YAML.

Deliverable:

6.12 [x] The sender can transmit pending edge rows to the local hub and update sync state only after successful ingestion.

Test:

6.13 [x] Start the FastAPI server in one terminal.

6.14 [x] Run `sender_daemon.py --config edge_node_mock/config/edge_config.local.yaml --limit 3` in another terminal.

6.15 [x] Confirm the edge DB now has the sent rows marked `synced`.

6.16 [x] Confirm the master DB contains matching rows for the same `device_id` and source `buffer_id` values.

6.17 [x] Stop the API server, create one new pending edge row, rerun the sender, and confirm the row is not incorrectly marked `synced`.

## 7. [x] Build The Watchdog Alert Mock

7.1 [x] Implement `watchdog_alert.py` in the central hub mock workspace.

7.2 [x] Query the master `health_metrics` table for the latest row for `device_id = 'pi_01'`.

7.3 [x] Compare the latest `timestamp_utc` with the current UTC time.

7.4 [x] Treat timestamps older than 75 minutes as stale.

7.5 [x] For Phase 1, print a dummy alert instead of sending Telegram, Discord, or email.

7.6 [x] Return a non-zero exit code when the device is stale.

7.7 [x] Return zero when the device is healthy.

7.8 [x] Add a short `--config` option so the master DB path, device ID, and stale threshold come from YAML.

Deliverable:

7.9 [x] The watchdog can distinguish healthy, stale, and missing telemetry states.

Test:

7.10 [x] Insert or update one health metric timestamp to the current UTC time and confirm the script reports healthy.

7.11 [x] Manually set the latest timestamp to more than 75 minutes old and confirm the script prints a dummy alert.

7.12 [x] Run against an empty health table and confirm it reports missing telemetry cleanly.

## 8. [x] Run The End-To-End Desktop Rehearsal

8.1 [x] Delete or archive old mock databases.

8.2 [x] Initialize fresh edge and master databases.

8.3 [x] Run the capture loop for a bounded number of buffers.

Suggested command:

```text
python bio_capture_loop.py --config edge_node_mock/config/edge_config.local.yaml --iterations 5
```

8.4 [x] Start the local hub API.

8.5 [x] Run the sender daemon once.

8.6 [x] Run the watchdog.

8.7 [x] Export a short verification report.

Suggested report values:

```text
edge buffer count
edge pending count
edge synced count
master buffer count
master vector count
latest telemetry timestamp
retained audio count
MessagePack payload byte size
```

Deliverable:

8.8 [x] One local command sequence proves the whole desktop mock pipeline from audio fixture to hub database and watchdog health check.

Test:

8.9 [x] Confirm all generated counts are internally consistent:

```text
master buffer count equals sent edge buffer count
master vector count equals master buffer count * 3
edge pending count is 0 after successful send
watchdog reports healthy after sender telemetry arrives
```

## 9. [x] Phase 1 Definition Of Done

9.1 [x] `bio_capture_loop.py` continuously or boundedly processes mock audio and populates the edge DB.

9.2 [x] Every 15-second buffer creates one metadata row and three 5-second embedding rows.

9.3 [x] The audio retention decision is visible through `audio_saved`, `retention_reason`, and `filepath`.

9.4 [x] The sender transmits pending rows using MessagePack and marks rows `synced` only after hub confirmation.

9.5 [x] The hub API authenticates, validates, and inserts batches into a WAL-enabled master DB.

9.6 [x] The watchdog identifies stale timestamps and prints a dummy alert.

9.7 [x] The model embedding dimension has been confirmed from the installed Perch model and reflected in both edge and hub DB setup.

9.8 [x] All script paths, thresholds, API settings, and mock data stream settings are read from YAML config files, with short command-line overrides only for testing.

9.9 [x] The end-to-end desktop rehearsal has been run from fresh databases.

9.10 [x] The final `/data` raw recording path has either been confirmed or is listed as a Phase 2 open decision.

9.11 [x] Known limitations and Phase 2 hardware assumptions are documented before moving to VS Code Remote SSH.

## 10. [ ] Refactor To Margin Bio Gate And Variable-Length Retention Buffers

The gate-tuning notebook showed that a margin-based NZ bird gate is a better
Phase 1 retention policy than the original broad biological/noise label gate.
This section replaces the first-pass `bio_threshold`/`noise_threshold` decision
with a gate that compares the strongest NZ bird candidate against a small set of
high-risk excluded FSD50K labels, then saves variable-length audio clips based on
consecutive triggered 5-second Perch frames.

Target policy:

```text
top_nz_logit = highest logit from labels/north_island_nz_perch_lablel.csv
top_excluded_logit = highest logit from excluded margin labels
margin = top_nz_logit - top_excluded_logit

frame_bio_gate = (
    overall_top_label not in excluded margin labels
    and margin >= bio_margin_threshold
)
```

Initial tuned values from `notebooks/04_gate_logic_tuning.ipynb`:

```yaml
bio_gate_mode: nz_bird_margin
bio_margin_threshold: 0.55
excluded_margin_labels:
  - Water
  - Train
  - Vehicle
max_variable_buffer_frames: 3
perch_window_seconds: 5.0
```

Variable-buffer retention policy:

```text
1 triggered 5-second frame  -> save 5-second clip
2 consecutive trigger frames -> save 10-second clip
3 consecutive trigger frames -> save 15-second clip
more than 3 consecutive trigger frames -> split into multiple max-15-second clips
```

10.1 [ ] Update the edge YAML config templates.

Add new config fields to `edge_node_mock/config/edge_config.example.yaml` and
the local config file:

```yaml
bio_gate_mode: nz_bird_margin
bio_margin_threshold: 0.55
excluded_margin_labels:
  - Water
  - Train
  - Vehicle
max_variable_buffer_frames: 3
perch_window_seconds: 5.0
```

Keep the old `bio_threshold`, `noise_threshold`, `noise_labels`, and
`biological_labels` fields temporarily if existing scripts or notebooks still
read them, but mark them as legacy first-pass gate settings in comments or docs.

10.2 [ ] Refactor frame scoring in `bio_capture_loop.py`.

Add a new per-frame score object or extend `FrameScores` with:

```text
top_nz_label_number
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

Use the full Perch logits for both the NZ bird subset and the excluded label
set. Do not derive these values only from the displayed top-3 labels.

10.3 [ ] Add reusable margin-gate functions.

Suggested functions:

```python
def build_margin_label_indexes(
    perch_labels: list[str],
    excluded_margin_labels: list[str],
) -> dict[str, int]:
    ...

def score_margin_gate_frame(
    frame_logits,
    *,
    perch_labels: list[str],
    nz_label_indexes: dict[int, NzBirdLabel],
    excluded_label_indexes: dict[str, int],
    threshold: float,
) -> MarginGateFrameScore:
    ...

def score_margin_gate_buffer(...) -> list[MarginGateFrameScore]:
    ...
```

Validation rules:

```text
all configured excluded labels must exist in perch_label.csv
the NZ bird subset must not be empty
bio_margin_threshold must be numeric
max_variable_buffer_frames must be >= 1
```

10.4 [ ] Implement variable-buffer grouping.

Add a function that groups consecutive triggered 5-second frames into proposed
audio saves:

```python
def build_variable_retention_buffers(
    frame_scores: list[MarginGateFrameScore],
    *,
    max_frames: int = 3,
) -> list[RetentionBuffer]:
    ...
```

Each returned retention buffer should include:

```text
retention_index
start_segment_index
end_segment_index
start_offset_s
end_offset_s
duration_s
triggered_frame_count
retention_reason
```

For Phase 1, use `retention_reason = 'bio_hit'` for all margin-triggered
variable clips. Keep `validation_sample` behavior as a separate later decision;
do not mix validation sampling into the first implementation of the new gate.

10.5 [ ] Refactor audio saving.

Change `save_retained_audio(...)` so it can save a slice of the current source
buffer rather than always saving the full 15 seconds.

Required behavior:

```text
save 5, 10, or 15 seconds according to the retention buffer span
preserve source sample rate in the saved FLAC
include duration or segment span in the filename
store a filepath only for saved variable clips
```

Suggested filename pattern:

```text
{device_id}_{timestamp}_{reason}_{source_stem}_{file_buffer_index:03d}_seg{start}-{end}_{duration}s.flac
```

10.6 [ ] Decide whether `buffer_events` represents inference buffers or saved clips.

Recommended Phase 1 refactor: keep `buffer_events` as the inference event table
and add a new child table for variable saved clips. This preserves one row per
15-second Perch inference call and exactly three embeddings per inference event,
while allowing zero, one, or more saved audio clips per event.

Add an edge table:

```sql
CREATE TABLE IF NOT EXISTS retained_audio_clips (
    clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
    buffer_id INTEGER NOT NULL,
    retention_index INTEGER NOT NULL,
    retention_reason TEXT NOT NULL CHECK(retention_reason IN ('bio_hit', 'validation_sample')),
    filepath TEXT NOT NULL,
    start_segment_index INTEGER NOT NULL,
    end_segment_index INTEGER NOT NULL,
    start_offset_s REAL NOT NULL,
    end_offset_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    triggered_frame_count INTEGER NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(buffer_id) REFERENCES buffer_events(buffer_id) ON DELETE CASCADE,
    UNIQUE(buffer_id, retention_index)
);
```

Update `buffer_events` with margin-gate summary fields:

```text
gate_mode TEXT
gate_threshold REAL
gate_trigger_count INTEGER
retained_clip_count INTEGER
margin_gate_scores TEXT
```

Keep existing fields for compatibility during the transition, but define their
new meaning clearly:

```text
audio_saved = 1 when retained_clip_count > 0
filepath = null once retained_audio_clips is authoritative
max_bio_label/max_bio_logit = strongest NZ bird candidate for the 15-second inference event
noise_logits = compact JSON of excluded-label evidence, not broad environmental scoring
nz_bird_logits = compact JSON of top NZ birds per 5-second frame
```

10.7 [ ] Update the hub schema.

Add a matching hub child table:

```sql
CREATE TABLE IF NOT EXISTS hub_retained_audio_clips (
    hub_clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hub_buffer_id INTEGER NOT NULL,
    source_clip_id INTEGER,
    retention_index INTEGER NOT NULL,
    retention_reason TEXT NOT NULL,
    filepath TEXT NOT NULL,
    start_segment_index INTEGER NOT NULL,
    end_segment_index INTEGER NOT NULL,
    start_offset_s REAL NOT NULL,
    end_offset_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    triggered_frame_count INTEGER NOT NULL,
    received_at_utc TEXT NOT NULL,
    FOREIGN KEY(hub_buffer_id) REFERENCES hub_buffer_events(hub_buffer_id) ON DELETE CASCADE,
    UNIQUE(hub_buffer_id, retention_index)
);
```

Add matching margin-gate summary columns to `hub_buffer_events`:

```text
gate_mode TEXT
gate_threshold REAL
gate_trigger_count INTEGER
retained_clip_count INTEGER
margin_gate_scores TEXT
```

10.8 [ ] Update sender payload construction.

Extend each detection payload with:

```text
gate_mode
gate_threshold
gate_trigger_count
retained_clip_count
margin_gate_scores
retained_audio_clips[]
```

Each retained clip payload should include:

```text
source_clip_id
retention_index
retention_reason
filepath
start_segment_index
end_segment_index
start_offset_s
end_offset_s
duration_s
triggered_frame_count
```

For Phase 1, continue sending file paths rather than binary audio. The hub mock
is validating metadata movement, not transferring FLAC bytes.

10.9 [ ] Update hub ingestion validation and inserts.

Extend the Pydantic detection model in `ingestion_api.py` and insert retained
clip rows inside the same transaction as the parent buffer event and embeddings.

Validation rules:

```text
retained_clip_count must equal len(retained_audio_clips)
audio_saved must equal retained_clip_count > 0
clip duration must be 5, 10, or 15 seconds for the current Perch window size
clip segment indexes must be between 0 and 2 for a single 15-second inference buffer
```

10.10 [ ] Update tests for the new gate.

Add focused unit tests for:

```text
highest NZ bird candidate is selected from the full NZ subset
highest excluded label is selected from the configured excluded labels
overall top excluded label veto wins even when margin passes
margin equal to threshold passes because the documented comparison is >= threshold
one triggered frame produces one 5-second retained clip
two consecutive triggered frames produce one 10-second retained clip
three consecutive triggered frames produce one 15-second retained clip
four consecutive triggered frames produce one 15-second clip and one 5-second clip
non-consecutive triggered frames produce separate 5-second clips
```

Add DB/schema tests for:

```text
edge retained_audio_clips table exists
hub_retained_audio_clips table exists
buffer_events summary fields exist
hub_buffer_events summary fields exist
foreign keys cascade from buffer event to retained clips
```

Add sender/ingestion tests for:

```text
retained clip metadata is included in MessagePack payloads
hub rejects inconsistent retained_clip_count values
hub inserts retained clips linked to the correct source buffer
edge rows still transition to synced only after accepted hub ingest
```

10.11 [ ] Update the full six-night test runner.

Modify `scripts/run_phase1_full_test.py` so the metrics report includes:

```text
margin threshold used
excluded margin labels used
frame_bio_gate trigger count
retained variable clip count
retained seconds
5-second / 10-second / 15-second retained clip counts
top NZ bird labels among retained clips
top excluded labels among vetoed frames
margin distribution summary
```

Update `outputs/phase1_full_test/latest/frame_metrics.csv` or add a companion
CSV to include the new per-frame margin fields.

10.12 [ ] Update notebooks.

Update:

```text
notebooks/01_edge_capture_walkthrough.ipynb
notebooks/03_end_to_end_system_walkthrough.ipynb
notebooks/04_gate_logic_tuning.ipynb
```

Notebook goals:

```text
show the new margin gate logic
show variable retained clips in the notebook output directory
show retained_audio_clips rows alongside buffer_events rows
show how one 15-second inference event can create a 5, 10, or 15 second saved FLAC
```

10.13 [ ] Update documentation.

Update:

```text
README.md
docs/01_concept/architecture_concept.md
docs/02_implementation/Phase1_test_report.md
```

Documentation should explain:

```text
why the margin gate replaced the original broad biological/noise threshold
why Water, Train, and Vehicle are the initial excluded labels
how the NZ bird label subset is used
how variable retained clips relate to fixed 15-second Perch inference events
what metadata is stored at the edge and hub
which threshold value was used for the current test report
```

10.14 [ ] Run a controlled regression test on the known reference file.

Use:

```text
/data/petrel_acoustics/raw_audio/doc_ar4/rapanui_AR4_june_2023/20230527/20230527_213004.wav
```

Start with the notebook section:

```text
570s to 700s
BIO_MARGIN_THRESHOLD = 0.55
excluded_margin_labels = Water, Train, Vehicle
```

Confirm:

```text
known petrel call frames trigger
obvious excluded-label frames are vetoed
variable saved clips match the notebook visualization
saved FLAC durations match the retained clip metadata
```

10.15 [ ] Run the full six-night test again after the refactor.

Command target:

```text
python scripts/run_phase1_full_test.py --config edge_node_mock/config/edge_config.local.yaml
```

Report outputs:

```text
outputs/phase1_full_test/latest/
docs/02_implementation/Phase1_test_report.md
```

Acceptance criteria:

```text
script completes without schema or payload errors
all retained clips have valid file paths and expected durations
edge retained clip counts match hub retained clip counts after sending
embedding row count remains three per 15-second inference event
margin-gate metrics are present in the report
```

10.16 [ ] Decide how to handle validation samples under variable retention.

Open decision for after the first refactor:

```text
Option A: validation samples remain full 15-second inference buffers
Option B: validation samples are fixed 5-second clips from randomly selected frames
Option C: validation samples are disabled during gate tuning runs
```

Default for the initial implementation should be Option C unless the test report
shows we need retained negatives for reviewing false rejections.

Deliverable:

10.17 [ ] The edge mock uses the margin-based NZ bird gate, writes variable
retained clips, and still preserves one inference event with three embeddings
for each 15-second Perch call.

Test:

10.18 [ ] Unit, schema, sender, ingestion, notebook, reference-file, and
six-night regression tests all pass with `BIO_MARGIN_THRESHOLD = 0.55`.
