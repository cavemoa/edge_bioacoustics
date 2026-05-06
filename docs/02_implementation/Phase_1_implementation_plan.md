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

## 4A. [x] Build A Teaching Notebook For The Capture Pipeline

4A.1 [x] Create a notebook in the root `notebooks/` directory.

Suggested file:

```text
notebooks/04A_bio_capture_loop_walkthrough.ipynb
```

4A.2 [x] Explain the purpose of `bio_capture_loop.py` for a new student.

4A.3 [x] Show how YAML config values drive paths, thresholds, and label groups.

4A.4 [x] Demonstrate streaming the bundled 120-second example `.wav` recording into eight 15-second buffers.

4A.5 [x] Show how a 15-second buffer becomes three 5-second Perch input windows.

4A.6 [x] Run Perch inference and display logits, embedding shapes, and per-buffer inference timing.

4A.7 [x] Demonstrate frame scoring for noise labels, biological labels, max Perch label, and top NZ bird labels.

4A.8 [x] Walk through the first-pass retention gate and its JSON fields.

4A.9 [x] Include example code for saving retained `.flac` audio.

4A.10 [x] Include an optional scratch database insert so students can inspect `buffer_events`, `embedding_segments`, and vector row counts without altering the main edge DB.

Deliverable:

4A.11 [x] A commented notebook teaches the capture-loop flow step by step using the same functions as the production script.

Test:

4A.12 [x] Validate the notebook JSON and confirm it contains runnable code cells for the capture-loop walkthrough.

## 5. [ ] Build The Local Hub Ingestion API

5.1 [ ] Implement `ingestion_api.py` in the central hub mock workspace.

5.2 [ ] Expose a localhost FastAPI endpoint:

```text
POST /ingest_batch
```

5.3 [ ] Load hub settings from YAML rather than hardcoding the master DB path, API key, device allowlist, or embedding dimension.

Suggested desktop pattern:

```powershell
$env:HUB_CONFIG = "central_hub_mock/config/hub_config.local.yaml"
uvicorn ingestion_api:app --host 127.0.0.1 --port 8000
```

5.4 [ ] Require an `X-API-Key` header.

5.5 [ ] Reject requests with a missing or incorrect API key.

5.6 [ ] Accept raw MessagePack request bodies.

5.7 [ ] Decode MessagePack into a Python dictionary.

5.8 [ ] Validate decoded payloads with Pydantic models before writing to SQLite.

Required top-level payload fields:

```text
device_id
sent_at_utc
detections
telemetry
```

5.9 [ ] Validate each detection includes the source `buffer_id`, metadata fields, three embedding segments, and binary embeddings.

5.10 [ ] Validate telemetry includes desktop mock values for:

```text
timestamp_utc
cpu_temp_c
cpu_load_pct
disk_free_gb
battery_voltage
```

5.11 [ ] Insert accepted detections and telemetry into the master WAL-enabled database in one transaction.

5.12 [ ] Return a response containing accepted source buffer IDs.

Suggested response:

```json
{
  "status": "ok",
  "accepted_buffer_ids": [1, 2, 3]
}
```

Deliverable:

5.13 [ ] The API receives, authenticates, validates, and stores a hand-built MessagePack batch on localhost using YAML configuration.

Test:

5.14 [ ] Set `HUB_CONFIG` to the local YAML file and run `uvicorn ingestion_api:app --host 127.0.0.1 --port 8000`.

5.15 [ ] Send one valid test payload and confirm `200 OK`.

5.16 [ ] Send one payload with a bad API key and confirm `401` or `403`.

5.17 [ ] Send one malformed payload and confirm `422` or a controlled validation error.

5.18 [ ] Query the master DB and confirm the valid payload inserted exactly one batch, one telemetry row, and the expected detection/vector rows.

## 6. [ ] Build The Mock Sender Daemon

6.1 [ ] Implement `sender_daemon.py` in the edge mock workspace.

6.2 [ ] Query the edge database for `sync_status = 'pending'`.

6.3 [ ] Load each pending buffer with its three embedding segments and three vector blobs.

6.4 [ ] Gather desktop mock telemetry.

Suggested values:

```text
cpu_temp_c = 45.0
battery_voltage = 12.4
```

Real desktop-safe values may be gathered with `psutil` for CPU load and disk free space.

6.5 [ ] Build the MessagePack payload using binary embedding data directly.

6.6 [ ] POST to the configured localhost API endpoint with the `X-API-Key` header.

6.7 [ ] On `200 OK`, update only the accepted source buffer IDs to `sync_status = 'synced'` and set `synced_at_utc`.

6.8 [ ] On network failure, API failure, or validation failure, leave rows as `pending` or mark them `failed` with a clear logged reason.

6.9 [ ] Add a `--dry-run` mode that prints payload counts without sending.

6.10 [ ] Add a `--limit` option to send a small batch during desktop testing.

6.11 [ ] Add a short `--config` option so API URL, API key, database path, and telemetry defaults come from YAML.

Deliverable:

6.12 [ ] The sender can transmit pending edge rows to the local hub and update sync state only after successful ingestion.

Test:

6.13 [ ] Start the FastAPI server in one terminal.

6.14 [ ] Run `sender_daemon.py --config edge_node_mock/config/edge_config.local.yaml --limit 3` in another terminal.

6.15 [ ] Confirm the edge DB now has the sent rows marked `synced`.

6.16 [ ] Confirm the master DB contains matching rows for the same `device_id` and source `buffer_id` values.

6.17 [ ] Stop the API server, create one new pending edge row, rerun the sender, and confirm the row is not incorrectly marked `synced`.

## 7. [ ] Build The Watchdog Alert Mock

7.1 [ ] Implement `watchdog_alert.py` in the central hub mock workspace.

7.2 [ ] Query the master `health_metrics` table for the latest row for `device_id = 'pi_01'`.

7.3 [ ] Compare the latest `timestamp_utc` with the current UTC time.

7.4 [ ] Treat timestamps older than 75 minutes as stale.

7.5 [ ] For Phase 1, print a dummy alert instead of sending Telegram, Discord, or email.

7.6 [ ] Return a non-zero exit code when the device is stale.

7.7 [ ] Return zero when the device is healthy.

7.8 [ ] Add a short `--config` option so the master DB path, device ID, and stale threshold come from YAML.

Deliverable:

7.9 [ ] The watchdog can distinguish healthy, stale, and missing telemetry states.

Test:

7.10 [ ] Insert or update one health metric timestamp to the current UTC time and confirm the script reports healthy.

7.11 [ ] Manually set the latest timestamp to more than 75 minutes old and confirm the script prints a dummy alert.

7.12 [ ] Run against an empty health table and confirm it reports missing telemetry cleanly.

## 8. [ ] Run The End-To-End Desktop Rehearsal

8.1 [ ] Delete or archive old mock databases.

8.2 [ ] Initialize fresh edge and master databases.

8.3 [ ] Run the capture loop for a bounded number of buffers.

Suggested command:

```text
python bio_capture_loop.py --config edge_node_mock/config/edge_config.local.yaml --iterations 5
```

8.4 [ ] Start the local hub API.

8.5 [ ] Run the sender daemon once.

8.6 [ ] Run the watchdog.

8.7 [ ] Export a short verification report.

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

8.8 [ ] One local command sequence proves the whole desktop mock pipeline from audio fixture to hub database and watchdog health check.

Test:

8.9 [ ] Confirm all generated counts are internally consistent:

```text
master buffer count equals sent edge buffer count
master vector count equals master buffer count * 3
edge pending count is 0 after successful send
watchdog reports healthy after sender telemetry arrives
```

## 9. [ ] Phase 1 Definition Of Done

9.1 [ ] `bio_capture_loop.py` continuously or boundedly processes mock audio and populates the edge DB.

9.2 [ ] Every 15-second buffer creates one metadata row and three 5-second embedding rows.

9.3 [ ] The audio retention decision is visible through `audio_saved`, `retention_reason`, and `filepath`.

9.4 [ ] The sender transmits pending rows using MessagePack and marks rows `synced` only after hub confirmation.

9.5 [ ] The hub API authenticates, validates, and inserts batches into a WAL-enabled master DB.

9.6 [ ] The watchdog identifies stale timestamps and prints a dummy alert.

9.7 [ ] The model embedding dimension has been confirmed from the installed Perch model and reflected in both edge and hub DB setup.

9.8 [ ] All script paths, thresholds, API settings, and mock data stream settings are read from YAML config files, with short command-line overrides only for testing.

9.9 [ ] The end-to-end desktop rehearsal has been run from fresh databases.

9.10 [ ] The final `/data` raw recording path has either been confirmed or is listed as a Phase 2 open decision.

9.11 [ ] Known limitations and Phase 2 hardware assumptions are documented before moving to VS Code Remote SSH.
