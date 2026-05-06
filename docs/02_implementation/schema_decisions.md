# Phase 1 Schema Decisions

Phase 1 uses two SQLite databases: one edge-node mock database and one
central-hub mock database. The edge database is treated as the future Raspberry
Pi database, and the master database is treated as the future LattePanda hub
database.

## Audio Retention Fields

The duplicate retention fields from `docs/00_ideas/rpi_sqlite_schema.md` are
resolved into one set of fields on each buffer event:

```text
audio_saved INTEGER NOT NULL DEFAULT 0
retention_reason TEXT NOT NULL
filepath TEXT
```

`audio_saved` is constrained to `0` or `1`. `retention_reason` is constrained to
one of:

```text
bio_hit
validation_sample
dropped
```

`filepath` is nullable and should only be populated when retained audio exists.

## Sync State

The edge database stores sync state on `buffer_events.sync_status`. The allowed
values are:

```text
pending
in_flight
synced
failed
```

New capture rows start as `pending`. Sender logic may use `in_flight` later, but
Phase 1 only marks rows `synced` after the hub accepts their source
`buffer_id`.

## Vector Storage

The preferred vector storage is `sqlite-vec` with `float[1536]` vector columns:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS perch_vectors USING vec0(
    embedding_id INTEGER PRIMARY KEY,
    embedding float[1536]
);
```

The central hub mirrors this with `hub_perch_vectors`.

Development machines without `sqlite-vec` use BLOB fallback tables:

```text
perch_vector_blobs(embedding_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)
hub_perch_vector_blobs(hub_embedding_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)
```

Both databases record `embedding_dim`, `vector_storage_mode`, and `vector_table`
in `schema_metadata` so tests and later scripts can discover the active storage
mode without guessing.
