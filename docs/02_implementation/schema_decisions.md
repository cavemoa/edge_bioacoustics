# Phase 1 Schema Decisions

Phase 1 revision databases use schema version 2:

```text
phase1_revision = margin_variable_retention_v1
schema_version = 2
gate_mode = nz_bird_margin
gate_threshold = 0.55
```

## Core Contract

`buffer_events` and `hub_buffer_events` are inference-event tables. Each row
represents one 15-second compute unit and always has three 5-second Perch frame
embeddings.

Saved audio is represented only by child rows:

```text
retained_audio_clips
hub_retained_audio_clips
```

The parent `filepath`, `max_bio_*`, and `noise_logits` fields from the first
mock design are superseded. New revision-created databases do not include those
columns.

## Retention Fields

Parent event rows keep the retained-audio summary:

```text
audio_saved INTEGER NOT NULL DEFAULT 0
retention_reason TEXT NOT NULL CHECK(retention_reason IN ('bio_hit', 'dropped'))
retained_clip_count INTEGER NOT NULL DEFAULT 0
```

`audio_saved` is a derived convenience flag:

```text
audio_saved = retained_clip_count > 0
```

Retained clip child rows store the actual FLAC metadata:

```text
retention_index
retention_reason = bio_hit
filepath
start_segment_index
end_segment_index
start_offset_s
end_offset_s
duration_s
triggered_frame_count
```

The active variable-retention design saves 5, 10, or 15 seconds inside the
current 15-second inference event. Cross-buffer clip merging is deferred.

## Gate Evidence

The active gate is the North Island NZ bird margin gate:

```text
frame_bio_gate =
    overall top Perch label is not an excluded label
    AND top NZ bird logit - top excluded-label logit >= bio_margin_threshold
```

Current defaults:

```text
bio_margin_threshold = 0.55
excluded_margin_labels = Water, Train, Vehicle
```

Event rows store:

```text
max_nz_bird_common_name
max_nz_bird_scientific_name
max_nz_bird_logit
excluded_label_scores
nz_bird_logits
margin_gate_scores
```

`margin_gate_scores` is the compact per-frame audit trail used by reports and
debugging tools to explain why a frame was retained or dropped.

## Sync State

The edge database stores sync state on `buffer_events.sync_status`:

```text
pending
in_flight
synced
failed
```

Phase 1 marks rows `synced` only after the hub accepts their source
`buffer_id`.

## Vector Storage

The preferred vector storage is `sqlite-vec` with `float[1536]` vector columns.
Development machines without `sqlite-vec` use BLOB fallback tables. Both
databases record `embedding_dim`, `vector_storage_mode`, and `vector_table` in
`schema_metadata`.
