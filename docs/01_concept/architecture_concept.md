# Technical Specification: Edge-Native Bioacoustic Monitoring System

## 1. System Overview
This document outlines the architecture for a remote, power-efficient, edge-machine-learning bioacoustic monitoring system. The system captures environmental audio, performs on-device inference to isolate biological sounds, and synchronizes metadata, telemetry, retained-clip metadata, and vector embeddings to a hub database. The system is primarily designed for use on grey-faced petrels at a colony on the North Island of New Zealand. The petrels are present at their burrows in the colony at night.

The architecture is explicitly designed to solve three major challenges of remote AI IoT deployments: **Carrier-Grade NAT (CGNAT) traversal**, **cellular bandwidth limitations**, and **power/thermal management** on battery/solar systems.

The implementation roadmap is intentionally staged. Phase 1 proved the software
locally on the Linux Mint development PC. Phase 2 moves the edge workload to the
Raspberry Pi while keeping the hub on the Mint PC as `mint_hub`. Phase 3 moves
the Pi onto cellular and Tailscale while it is still physically accessible at
home. Phase 4 migrates the hub to the headless LattePanda. Phase 5 adds solar
power testing at home. Phase 6 is a full pre-field rehearsal, and Phase 7 is the
first limited field trial.

---

## 2. Hardware Infrastructure
*   **Edge Compute Node:** Raspberry Pi 5 with active cooling.
*   **Edge Power:** 40W 12v Solar Panel, 20Ah LiFePO4 battery with BMS
*   **Solar Charging Management:** PV Pi: Solar Charging Hat for Raspberry Pi
*   **Edge Network Gateway:** TP-Link Archer MR100-Outdoor 4G LTE Router.
*   **Development Hub:** Linux Mint PC running `mint_hub` during Phases 2 and 3.
*   **Central Ingestion Hub:** LattePanda Alpha, introduced after the Pi and network path are stable.
*   **Microphones:** Primo EM272Z1 Omni Electret Condenser Microphone
*   **USB Sound card:** Rode AI-Micro Compact Audio Interface

---

## 3. Network Architecture & Security
*   **Phase 2 Local LAN:** The Raspberry Pi first sends batches to `mint_hub` on the Linux Mint PC over the home local network.
*   **Phase 3 Mesh VPN (Tailscale):** Tailscale is then deployed on the Raspberry Pi and Mint PC while the Pi is still accessible at home.
*   **CGNAT Bypass:** Once cellular networking is introduced, Tailscale's static `100.x.x.x` IP addresses bypass the cellular provider's CGNAT constraints without port forwarding.
*   **Phase 4 Hub Migration:** After Pi cellular/Tailscale behavior is understood, the hub role migrates from Mint PC to LattePanda.
*   **Application Security:** The central API server enforces an application-level API key (`X-API-Key` header) to prevent unauthorized internal network requests.

---

## 4. Edge Processing Pipeline (Raspberry Pi 5)
The audio capture and inference pipeline operates asynchronously and is completely decoupled from the network transport layer.

### 4.1 Audio Capture & Inference
*   **Buffer:** Audio is captured into a continuous 15-second ring buffer.
*   **Sample Rate:** Native 32kHz (downsampled from standard 48kHz to optimize storage and perfectly match inference requirements).
*   **Inference Engine:** Perch 2.0.
*   **Gating & Filtering:** Perch logits are evaluated per 5-second frame. The current Phase 1 gate compares the strongest North Island NZ bird candidate against the strongest configured excluded FSD50K label (`Water`, `Train`, or `Vehicle`). A frame is biological when the NZ bird logit beats the excluded-label logit by the configured margin and the overall top label is not one of the excluded labels.

### 4.2 Edge Storage
*   **Audio Storage:** If biological activity is detected, only the triggered span is compressed to a lossless `.flac` file and saved to the Pi's local storage for future retrieval or manual validation. One triggered 5-second frame saves 5 seconds, two consecutive frames save 10 seconds, and three consecutive frames save the full 15-second inference buffer. **Audio files are never transmitted over 4G.**
*   **Database Storage:** Metadata (timestamp, gating scores, retained clip metadata) and the resulting three 1536-dimensional float vector embeddings are written to a local `sqlite-vector` database.
*   **Vector Format:** Embeddings are stored natively as binary `BLOB`s (raw 32-bit floats) to preserve absolute precision.
*   **State Tracking:** Newly inserted rows are flagged with the column `sync_status = 'pending'`.

---

## 5. Transport Layer (The Sender Daemon)
Data is moved from the Edge to the Central Hub via a highly optimized, asynchronous transport pipeline.

### 5.1 The Sender Daemon
*   A standalone Python script, initially run manually or by a simple schedule during local-network testing, then promoted to a background `systemd` service or timer.
*   **Cadence:** The daemon wakes up exactly once per hour. 
*   **Query:** It queries the local SQLite database for all rows where `sync_status = 'pending'`.

### 5.2 Serialization (MessagePack)
*   To drastically reduce CPU overhead, RAM spikes, and transmission payload size, JSON is discarded in favor of **MessagePack**.
*   **Zero-Copy Serialization:** Because SQLite stores the embeddings as `BLOB`s, and MessagePack natively understands binary data, the float arrays are packed directly into the payload without undergoing expensive text/string conversion.

### 5.3 Piggybacked Telemetry
*   Before transmitting, the daemon utilizes `psutil` and local system files to gather health metrics: CPU load, CPU temperature, disk space, and battery voltage / solar amperage (via I2C).
*   This telemetry data is appended to the hourly MessagePack payload, achieving comprehensive hardware monitoring with zero extra network requests.

### 5.4 Payload Schema (Conceptual)
```javascript
{
  "device_id": "pi_01",
  "detections": [
    {
      "id": 1042,
      "timestamp": "2026-05-05T06:15:30",
      "gate_mode": "nz_bird_margin",
      "margin_gate_scores": "[...]",
      "retained_audio_clips": [
        {"start_offset_s": 5.0, "end_offset_s": 10.0, "duration_s": 5.0}
      ],
      "embedding_segments": [
        {"segment_index": 0, "embedding": <Binary BLOB Data>},
        {"segment_index": 1, "embedding": <Binary BLOB Data>},
        {"segment_index": 2, "embedding": <Binary BLOB Data>}
      ]
    }
  ],
  "telemetry": {
    "timestamp": "2026-05-05T07:00:00",
    "cpu_temp_c": 45.2,
    "cpu_load_pct": 12.5,
    "disk_free_gb": 42.1,
    "battery_volts": 12.4,
    "solar_amps": 0.8
  }
}
```

---

## 6. Central Ingestion Server
The central hub processes incoming telemetry and ML data, making it immediately available for offline analysis. The hub has two planned forms: `mint_hub` on the Linux Mint development PC during Phases 2 and 3, then the LattePanda hub from Phase 4 onward.

### 6.1 FastAPI Service
*   A lightweight FastAPI server listens first on the Mint PC local-network or Tailscale address, then later on the LattePanda's selected network address.
*   It exposes a secure `/ingest_batch` POST endpoint.
*   **Validation:** Incoming MessagePack data is unpacked and strictly validated against Pydantic schemas.

### 6.2 Master Database Insertion
*   **Concurrency:** The master `sqlite-vector` database runs with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) enabled to allow the FastAPI server to insert incoming batches while data scientists concurrently query the database for clustering and review.
*   **Insertion:** Validated metadata, binary embeddings, and piggybacked telemetry are inserted into their respective master tables.
*   **Confirmation:** Upon successful insertion, the API returns a `200 OK`, prompting the Pi's Sender Daemon to mark the edge rows as `sync_status = 'synced'`.

---

## 7. Passive Watchdog & Alerting
To prevent the "Observer Effect" (waking the 4G modem to check system health, thereby draining the battery), the system relies on a passive "Dead Man's Switch".

*   **Implementation:** A scheduled watchdog runs first on the Mint PC hub, then later on the LattePanda Alpha.
*   **Logic:** It checks the `health_metrics` SQLite table for the most recent check-in from the Pi.
*   **Alerting:** If the latest timestamp is older than 75 minutes (accounting for the 60-minute sync cycle + 15 minutes of network grace time), the current hub triggers an automated alert (e.g., via Telegram or Email) indicating that the edge node has likely suffered a power or hardware failure.

---

## 8. Summary of Optimizations
1.  **Bandwidth:** Leaving `.flac` files on the edge reduces daily data transmission from gigabytes to under 20 Megabytes.
2.  **Power:** Hourly batched syncs allow the 4G LTE modem to remain in a low-power state for 98% of the day.
3.  **Compute:** Utilizing MessagePack and binary BLOBs prevents CPU/RAM spikes on the Raspberry Pi during the serialization phase.
4.  **Resilience:** Decoupling the audio inference loop from the network transport ensures that the Pi continues recording and logging data perfectly even during extended cellular network outages.

Based on the edge-native architecture we have designed, you will need to develop **four distinct Python scripts** across the two devices. 

Here is the comprehensive list of the scripts, including their core responsibilities and the primary libraries they will rely on.

## 9. Summary of Software Architecture 

### **Part 1: Raspberry Pi 5 (The Edge Node)**

You will need two separate scripts running on the Pi. Decoupling them is what makes the system resilient to network drops.

#### **1. `bio_capture_loop.py` (The Audio & Inference Engine)**
This is your continuous, high-priority loop. It handles the actual listening and machine learning, completely ignoring the network.
*   **Run Method:** Started on boot via `systemd` (runs continuously).
*   **Primary Libraries:** `sounddevice` or `pyaudio` (for the ring buffer), `numpy`, `librosa` (for 32kHz downsampling/formatting), `sqlite3`, and the Perch 2.0 / FSD50K inference libraries.
*   **Core Responsibilities:**
    *   Maintain a 15-second rolling audio buffer at 32kHz.
    *   Pass the buffer to Perch 2.0 to generate embeddings and logits.
    *   Gate the results: compare the strongest NZ bird label against the strongest excluded label using a margin threshold.
    *   If positive: compress the triggered 5/10/15-second span to `.flac` and save it to the local SD card/USB drive.
    *   Insert margin scores, retained clip metadata, timestamp, and raw binary embeddings into the local `sqlite-vector` database with `sync_status = 'pending'`.

#### **2. `sender_daemon.py` (The Transport & Telemetry Worker)**
This script acts as your courier. It wakes up, gathers the data, checks the Pi's health, and ships it all to the server.
*   **Run Method:** Triggered hourly via a `cron` job (or a looping `systemd` timer).
*   **Primary Libraries:** `sqlite3`, `msgpack`, `requests`, `psutil`, `smbus2` (for I2C solar/battery reading).
*   **Core Responsibilities:**
    *   Gather system telemetry (CPU temp, disk space, battery voltage via I2C).
    *   Query the local SQLite database for all rows where `sync_status = 'pending'`.
    *   Bundle the telemetry and the database rows into a single dictionary.
    *   Serialize the dictionary into raw binary using `msgpack.packb()`.
    *   Send an HTTP POST request to the configured hub API using the API Key header.
    *   If a `200 OK` is received, update the local database rows to `sync_status = 'synced'`.

---

### **Part 2: The Hub**

The hub requires two scripts: one to passively receive the data, and one to actively monitor the health of the system. During Phases 2 and 3 these run as `mint_hub` on the Linux Mint PC. During Phase 4 they migrate to the LattePanda.

#### **3. `ingestion_api.py` (The FastAPI Server)**
This is the central nervous system of your database. It must be robust, strictly validated, and always online.
*   **Run Method:** Started manually during early Mint hub testing, then via `systemd` on the production hub, running through an ASGI server like `uvicorn`.
*   **Primary Libraries:** `fastapi`, `uvicorn`, `pydantic`, `sqlite3`, `msgpack`.
*   **Core Responsibilities:**
    *   Listen on the configured hub interface for the current phase.
    *   Authenticate incoming POST requests using a static `X-API-Key` header.
    *   Unpack the incoming MessagePack binary payload.
    *   Strictly validate the data structure and data types using Pydantic models.
    *   Open a fast, WAL-enabled connection to the master `sqlite-vector` database.
    *   Insert the biological data and the telemetry data into their respective tables safely.

#### **4. `watchdog_alert.py` (The Dead Man's Switch)**
This is a lightweight script that ensures you know if the Pi dies, without wasting the Pi's battery to find out.
*   **Run Method:** Triggered every 15 minutes via a `cron` job.
*   **Primary Libraries:** `sqlite3`, `requests` (for Telegram/Discord webhooks) or `smtplib` (for email alerts).
*   **Core Responsibilities:**
    *   Query the master `health_metrics` table on the current hub.
    *   Retrieve the most recent `timestamp` for the `pi_01` device.
    *   Compare the timestamp against the current system time.
    *   If the difference exceeds 75 minutes, fire an alert via webhook or email stating that the edge node has missed its check-in window.
