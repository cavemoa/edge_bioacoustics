### **Phase 1: Local Development & Mocking (The Desktop Phase)**

Your Linux Mint PC is your sandbox. The goal of this phase is to build the core logic of all four scripts without relying on physical sensors, 4G networks, or actual microcontrollers.

#### **1. Preparation (Setting the Stage)**
Before writing any script logic, you need to prepare your local Linux environment to mimic the dual-device architecture:
* **Workspace Separation:** Create two distinct root folders in VS Code (e.g., `edge_node_mock` and `central_hub_mock`) to prevent accidental cross-contamination of dependencies.
* **Virtual Environments:** Initialize a separate Python virtual environment (`venv`) in each folder. 
* **Acquire Test Assets:** Find or generate a pre-recorded 15-second, 48kHz `.wav` file. This will act as your "fake microphone" for the capture loop.

#### **2. Execution Steps**

**Step 2.1: Database Initialization**
* **Action:** Run a setup script to create both the edge SQLite database and the master WAL-enabled SQLite database on your local hard drive.
* **Details:** Ensure the schema strictly matches the final design. The edge database needs a table for embeddings (stored as raw 32-bit float BLOBs) with a `sync_status` column. 

**Step 2.2: Mocking the Capture Loop (`bio_capture_loop.py`)**
* **Action:** Write the core inference script, but bypass hardware inputs.
* **Details:** Do not use a real microphone yet. Write a mock generator that feeds your pre-recorded `.wav` file into the buffer. Pass this to Perch 2.0, generate the embeddings, and insert them into your local edge SQLite database with `sync_status = 'pending'`. 

**Step 2.3: Building the Hub API (`ingestion_api.py`)**
* **Action:** Develop the central receiver on the "server" side.
* **Details:** Spin up the `ingestion_api.py` FastAPI server on `localhost`[cite: 32]. Implement your API key authentication and write the Pydantic models to strictly validate incoming MessagePack payloads before inserting them into your master database.

**Step 2.4: Mocking the Sender Daemon (`sender_daemon.py`)**
* **Action:** Build the transport layer to connect your two local components.
* **Details:** Hardcode dummy telemetry data (e.g., `cpu_temp = 45.0`, `battery_voltage = 12.4`) instead of trying to read from the I2C bus. Have the script query the local edge database for pending rows, bundle them with the dummy telemetry, pack it using MessagePack, and POST it to your `localhost` FastAPI server. 
* **Testing:** You can trigger the sender script in one terminal and watch the FastAPI server receive and unpack it in another.

**Step 2.5: Mocking the Watchdog (`watchdog_alert.py`)**
* **Action:** Build the dead man's switch.
* **Details:** Write a script that queries your local master database's health metrics table. Manually manipulate the timestamps in your local database to be older than 75 minutes, and verify that the script successfully triggers a dummy alert printout to your terminal.

#### **3. Definition of Done**
You are ready to move to Phase 2 (VS Code Remote - SSH) when:
1. Your mock audio loops continuously and populates the local edge database.
2. Running the sender daemon successfully transmits the data locally, updates the edge database to `sync_status = 'synced'`, and populates the master database.
3. The watchdog correctly identifies stale timestamps.