## Ideas on the strcuture of the SQLite database for both relational data and vector embeddings


Planned highly robust strategy. By saving **all** embeddings but gating the actual `.flac` audio files, you are building a complete, mathematically continuous baseline of the night's acoustic environment while still aggressively protecting your edge storage capacity. Creating a new SQLite database file per night (e.g., `perch_data_YYYY_MM_DD.sqlite`) is also a great practice for manageable file sizes and easy backup.

To support this new logic, we need to modify the previous schema. We will add specific fields to track whether the audio was saved and *why* it was saved (to differentiate between a biological hit and your 5-minute validation sample). 

Here is the updated `sqlite-vec` schema encompassing both the relational metadata and the vector embeddings:

### **The Updated Nightly SQLite-Vec Schema**

```sql
-- ==========================================
-- 1. Relational Data: Core Metadata & Queue
-- ==========================================
CREATE TABLE IF NOT EXISTS buffer_events (
    buffer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Audio Retention Tracking
    audio_saved INTEGER DEFAULT 0,  -- 1 (True) if audio was saved, 0 (False) if dropped
    retention_reason TEXT,          -- 'bio_hit', 'validation_sample', or 'dropped'
    filepath TEXT,                  -- Path to the .flac file (NULL if audio_saved is 0)
    
    -- Audio Retention Tracking
    audio_saved INTEGER DEFAULT 0,     -- 1 (True) if audio was saved, 0 (False) if dropped [cite: 95]
    retention_reason TEXT,             -- 'bio_hit', 'validation_sample', or 'dropped' [cite: 95]
    filepath TEXT,                     -- Path to the .flac file (NULL if audio_saved is 0) [cite: 95]
    
    -- Inference Metadata (FSD50K Gating)
    max_bio_label TEXT,                -- The highest biological FSD50K label from FSD50K subset contained in FSD50K_subset_labels.csv
    max_bio_logit REAL,                -- The highest biological FSD50K score
    noise_logits JSON,                 -- A JSON object of the top FSD50K noise scores from FSD50K noise list in script yaml file
    
    -- Inference Metadata (Perch Specifics)
    max_perch_label TEXT,              -- The single highest scoring Perch label overall
    max_perch_logit REAL,              -- The certainty (logit) of that highest Perch label
    nz_bird_logits JSON,               -- A JSON object of the top 3 NZ-specific bird labels and their scores from north_island_nz_perch_lablel.csv list
    
    -- Transport State
    sync_status TEXT DEFAULT 'pending' -- Tracks if the row has been sent to the LattePanda
);

-- ==========================================
-- 2. Relational Data: Embedding Mapping
-- ==========================================
-- Maps the 3 distinct 5-second embeddings Perch generates back to the single 15s buffer
CREATE TABLE IF NOT EXISTS embedding_segments (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    buffer_id INTEGER,           
    segment_index INTEGER,       -- 0, 1, or 2 (representing the 1st, 2nd, or 3rd 5-second chunk)
    FOREIGN KEY(buffer_id) REFERENCES buffer_events(buffer_id) ON DELETE CASCADE
);

-- ==========================================
-- 3. Vector Data: The sqlite-vec Table
-- ==========================================
-- Specialized virtual table for the raw 32-bit float BLOBs. 
-- The rowid matches the embedding_id from embedding_segments.
CREATE VIRTUAL TABLE IF NOT EXISTS perch_vectors USING vec0(
    embedding_id INTEGER PRIMARY KEY, 
    embedding float[1536]        -- Perch 2.0's 1536-dimensional vector
);
```

### **How your Python logic changes with this schema:**

Since you are now saving embeddings for *all* audio streams, the database insertion sequence in your `bio_capture_loop.py` script will look like this for **every single 15-second buffer**:

1.  **Evaluate the Gate:** Check the logits. Did it pass the biological threshold? 
    * *If no:* Has it been 5 minutes since the last validation sample?
2.  **Determine Status:** * Set `audio_saved = 1` and `retention_reason = 'bio_hit'` (Save the `.flac`)
    * *OR* set `audio_saved = 1` and `retention_reason = 'validation_sample'` (Save the `.flac`)
    * *OR* set `audio_saved = 0`, `retention_reason = 'dropped'`, and `filepath = NULL` (Discard the audio).
3.  **Insert Relational Data:** Insert the record into `buffer_events` and grab the returned `buffer_id`.
4.  **Insert Vector Data (Always happens now):** Loop through your 3 Perch embeddings. Insert a mapping row into `embedding_segments` for each one, grab the `embedding_id`, and insert the raw float array into the `perch_vectors` virtual table.

When your `sender_daemon.py` wakes up hourly, it will just blindly grab everything where `sync_status = 'pending'`, package it up, and send it to the LattePanda. Because you are sending *all* embeddings now, expect your hourly Tailscale payload to be around **3.5 to 4.5 MB**.