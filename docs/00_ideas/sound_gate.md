





### The "5-Second Rule"
Perch 2.0 is hardcoded to analyze audio in **5-second windows**. 
When you pass your 15-second `audio_32k` array to the full TF2 model, it does not output just one single prediction. It automatically chops the 15 seconds into three separate 5-second frames. 

The model will output an array of shape `[3, 14795]` for the logits, and `[3, embedding_size]` for the embeddings. 

This is actually a **massive advantage** for your noise gate. A strong gust of wind might ruin the first 5 seconds of the clip, but a Grey-faced Petrel might call perfectly in the final 5 seconds. By evaluating each 5-second chunk individually, you won't throw away a positive detection just because part of the buffer was noisy.

### The Full Pipeline Code
Here is how you combine the Rode AI-Micro downsampling, the native Kaggle TF2 model, and the noise gate logic.

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.io import wavfile

# --- 1. Initialization (Runs Once on Boot) ---
print("Loading TensorFlow Model...")
# Load the unzipped folder you downloaded from Kaggle
model = tf.saved_model.load('/path/to/perch_v2_cpu')

print("Loading Labels...")
labels_df = pd.read_csv('/path/to/perch_v2_cpu/assets/labels.csv')

# Setup Noise Gate subsets
noise_labels = ['Wind', 'Ocean', 'Waves, surf', 'Rain', 'Thunderstorm']
positive_labels = ['Bird', 'Bird vocalization, bird call, bird song', 'Animal']

noise_idx = labels_df[labels_df['name'].isin(noise_labels)].index.to_numpy()
positive_idx = labels_df[labels_df['name'].isin(positive_labels)].index.to_numpy()


# --- 2. The Inference Loop ---
def process_15s_window(audio_48k_buffer):
    # 1. Downsample from 48kHz (Rode) to 32kHz (Perch)
    audio_32k = resample_poly(audio_48k_buffer, up=2, down=3)
    
    # Ensure it's 32-bit float for TensorFlow
    audio_32k = audio_32k.astype(np.float32)

    # 2. Run Inference
    # The model expects a 1D tensor. We use infer_tf (standard for this model)
    output = model.infer_tf(tf.constant(audio_32k))
    
    logits = output['predictions'].numpy() # Shape: (3, 14795)
    embeddings = output['embeddings'].numpy() # Shape: (3, 1280)

    is_petrel = False

    # 3. Analyze each 5-second frame individually
    for frame_index in range(3):
        frame_logits = logits[frame_index]
        
        # Slicing the scores
        noise_scores = frame_logits[noise_idx]
        positive_scores = frame_logits[positive_idx]
        
        max_noise = np.max(noise_scores)
        max_positive = np.max(positive_scores)
        
        # Check the gate for this specific 5 seconds
        if max_noise > 5.0 and max_positive < 2.0:
            print(f"Frame {frame_index}: Rejected by noise gate (Noise: {max_noise:.2f})")
            continue # Skip to the next frame
            
        elif max_positive >= 2.0:
            print(f"Frame {frame_index}: Biological sound detected! Passing to custom classifier...")
            
            # Grab the specific embedding for this 5-second frame
            frame_embedding = embeddings[frame_index]
            
            # -> RUN YOUR CUSTOM SCIKIT-LEARN PETREL MODEL HERE <-
            # if custom_model.predict([frame_embedding]) == "Petrel":
            #     is_petrel = True
            #     break # We found one, no need to check the rest of the 15s window

    # 4. Save to SSD if triggered
    if is_petrel:
        filename = "petrel_detection.wav"
        # Scale float32 back to standard 16-bit PCM for the .wav file
        audio_16bit = np.int16(audio_32k * 32767)
        wavfile.write(filename, 32000, audio_16bit)
        print(f"Saved 15-second clip to SSD.")
```


**SQLite combined with the `sqlite-vec` extension is the absolute best option** for this edge deployment. 

You do not need a separate "conventional" database for your metadata. The beauty of SQLite is that it is a world-class relational database first and foremost. By adding the `sqlite-vec` extension, you give standard SQLite the superpower to store and query high-dimensional arrays alongside standard text and float columns.

Moving to a "bigger" vector database like ChromaDB, Qdrant, or running PostgreSQL with `pgvector` on a Raspberry Pi 5 would introduce unnecessary background services, memory overhead, and complexity. SQLite runs as a single `.db` file on your SSD with zero background processes.

Here is how you architect this database beautifully to handle both your vector embeddings and your rich diagnostic metadata.

### 1. The Database Architecture
Because of how `sqlite-vec` works under the hood, the most efficient way to store this data is using a **two-table relational model** linked by a shared ID:

1.  **The Metadata Table (Standard SQLite):** Stores your timestamp, file paths, and the specific logits (noise, general bird, and *Pterodroma gouldi*).
2.  **The Vector Table (Virtual `vec0` Table):** A highly optimized virtual table dedicated purely to searching the 1280-dimensional Perch embeddings.

### 2. Setting up the Schema in Python
Here is the exact Python code to initialize this structure. You will need to install the extension via pip (`pip install sqlite-vec sqlite3`).

```python
import sqlite3
import sqlite_vec
import numpy as np
import datetime

def init_database(db_path="petrel_detections.db"):
    # Connect and load the sqlite-vec extension
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()

    # 1. Create the Standard Metadata Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            file_path TEXT,
            max_noise_score REAL,
            max_positive_score REAL,
            gouldi_logit REAL
        )
    """)

    # 2. Create the Virtual Vector Table for Perch Embeddings
    # Perch v2 embeddings are 1280 dimensions
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS detection_embeddings USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[1280]
        )
    """)
    
    conn.commit()
    return conn

# Initialize the DB when the script starts
db_conn = init_database()
```

### 3. Inserting a Detection
When your 15-second loop triggers a positive detection, you package all those diagnostic logits you sliced out and save them in one atomic transaction.

```python
def save_detection_to_db(conn, file_path, noise_score, positive_score, gouldi_score, embedding_array):
    cursor = conn.cursor()
    
    # Current time in ISO format
    now = datetime.datetime.now().isoformat()
    
    # 1. Insert Metadata
    cursor.execute("""
        INSERT INTO detection_metadata 
        (timestamp, file_path, max_noise_score, max_positive_score, gouldi_logit)
        VALUES (?, ?, ?, ?, ?)
    """, (now, file_path, float(noise_score), float(positive_score), float(gouldi_score)))
    
    # Grab the ID that SQLite just generated for this row
    row_id = cursor.lastrowid
    
    # 2. Insert the Embedding into the Virtual Table using the same ID
    # sqlite-vec expects the embedding as bytes, which NumPy can handle easily
    cursor.execute("""
        INSERT INTO detection_embeddings (id, embedding)
        VALUES (?, ?)
    """, (row_id, embedding_array.tobytes()))
    
    conn.commit()
    print(f"Logged detection {row_id} to database.")

# Example usage inside your audio loop:
# save_detection_to_db(db_conn, "petrel_clip_01.wav", max_noise, max_positive, petrel_score, frame_embedding)
```

### Why this setup is incredibly powerful for Diagnostics
By linking the metadata and the vector embeddings via the `id`, you can perform incredibly advanced diagnostic queries later when you retrieve the Pi from the field. 

For example, using a standard SQL `JOIN`, you can ask the database:
*"Find me 10 clips that sound exactly like this verified petrel call (Vector Search), but ONLY show me clips where the wind noise logit was less than 3.0 (Metadata Filter)."*

```sql
SELECT 
    m.file_path, 
    m.timestamp, 
    m.gouldi_logit,
    vec_distance_cosine(e.embedding, :target_vector) as similarity
FROM detection_embeddings e
JOIN detection_metadata m ON e.id = m.id
WHERE m.max_noise_score < 3.0
ORDER BY similarity LIMIT 10;
```

This single `.db` file will hold hundreds of thousands of detections flawlessly, take up very little space, and give you robust, crash-proof logging for your Taranaki deployment. 

Are you comfortable with SQL, or would you like to build out a few specific Python helper functions to query this diagnostic data later?


## Concerns around "Blind Gating"

This is an excellent engineering question. Your hesitation about "blind gating" is entirely justified. In machine learning deployments, if you silently drop data at the edge, you lose the ability to audit why your model made that decision, making it incredibly difficult to tune your gating thresholds later.

However, choosing what to send back to your LattePanda requires balancing your need for validation against the core constraints of your system: cellular bandwidth and edge storage. 

Here is the straightforward breakdown of your two options, along with the recommended approach for your testing phase.

### **Option A: Saving the Full Embedding for Dropped Audio**
This means every time your Pi captures a 15-second buffer of wind or ocean noise, it saves the full 1536-dimensional EfficientNet vector to the SQLite database and transmits it via MessagePack.

* **The Pros:** Total retrospective visibility. Using the "Agile Modeling" advantage we discussed previously, you could later run these "noise" embeddings through a new classifier on your LattePanda to see exactly what sounds were present, or verify if a faint grey-faced petrel call was masked by the wind. 
* **The Cons (The Bandwidth Trap):** Your system processes four 15-second buffers every minute, totaling 5,760 buffers a day. A single 1536-dimensional 32-bit float vector is about 6KB. If it’s a stormy night and 90% of your audio is dropped, sending all those discarded embeddings will add roughly ~30MB of daily data to your cellular payload. While manageable on some 4G plans, it directly contradicts your goal of keeping the daily transmission extremely lean.

### **Option B: Saving a Subset of Classified Scores (Logits)**
Instead of the heavy 1536-float embedding, you only save and transmit the specific FSD50K classification scores (logits) that triggered the gate to close. 

* **The Pros:** Extreme efficiency. You are only sending a few bytes of data (e.g., `{"wind": 0.85, "ocean": 0.60, "max_bio": 0.12}`). This perfectly preserves your bandwidth and database size while giving you the exact mathematical reason the audio was dropped.
* **The Cons:** You cannot retroactively re-analyze the audio for new, unexpected biological classes. You only know what the model thought it heard at that exact moment.

---

### **The Verdict: A Hybrid Testing Strategy**

For a robust, professional IoT testing phase, **Option B (Subset of Scores) is the better architectural choice**, but it needs a slight modification to solve your "blind spot" concern. 

I recommend implementing a **"Random Sampling" (or Decimation) strategy** alongside your subset scores. Here is how you should structure your `bio_capture_loop.py` script:

1.  **For Positive Biological Hits:** Status quo. Save the `.flac` locally, and send the full 1536-dimensional embedding + metadata to the LattePanda.
2.  **For Dropped Audio (Standard):** Discard the audio. Save **only a subset of scores** (the top 3 FSD50K noise logits and the highest biological logit) to your SQLite database to be synced to the hub. This acts as a lightweight audit log.
3.  **The Validation Catch (1 in 100):** Add a simple counter or randomizer to your script. For every 100th buffer that gets classified as "noise" and dropped, **override the gate**. Save the `.flac` locally and send the full 1536-dimensional embedding anyway, tagging it as a `validation_sample`. 

**Why this works:**
It keeps your cellular payload incredibly light and respects your power/storage constraints, while providing a mathematically sound statistical sample of your "dropped" audio. When you review the data on your LattePanda, you can look at the validation samples to ensure your wind and rain thresholds are tuned perfectly, without having to transmit thousands of useless wind embeddings.