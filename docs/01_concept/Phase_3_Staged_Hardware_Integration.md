### **Phase 3: Staged Hardware Integration**

Now that your code is running on the actual edge devices via SSH, you start swapping out your "mocks" for physical reality. The goal of this phase is to integrate the real sensors, establish live environmental monitoring, and make the deployment resilient to power cycles.

#### **1. Preparation (Setting the Stage)**

Before modifying your scripts, ensure all physical peripherals are securely hooked up to the Raspberry Pi 5.
* **Audio Hardware:** Connect the Primo EM272Z1 Omni Electret Condenser Microphone and the Rode AI-Micro Compact Audio Interface to the Pi.
* **Power Hardware:** Ensure the 40W 12v Solar Panel, 20Ah LiFePO4 battery, and Solar Charging Hat are properly connected to supply power and I2C data.

#### **2. Execution Steps**

**Step 3.1: Live Audio & AI Integration (`bio_capture_loop.py`)**
* **Action:** Transition from the mock `.wav` file to the live microphone feed.
* **Details:** Replace your mock audio generator with actual `sounddevice` or `pyaudio` capture streams. Ensure this stream properly feeds the continuous 15-second, 32kHz ring buffer to Perch 2.0 for inference.
* **Validation:** Test the microphone. Verify that positive FSD50K biological detections correctly trigger the buffer to compress to a lossless `.flac` file on local storage, and that the database logs the raw 32-bit float vector embeddings.

**Step 3.2: Sensor Integration (`sender_daemon.py`)**
* **Action:** Hook into the physical hardware telemetry.
* **Details:** Replace your hardcoded telemetry with real `psutil` and `smbus2` calls[cite: 45]. Write the logic to accurately read CPU loads and temperatures via `psutil`, and query the Solar Charging Hat over the I2C bus for actual battery voltage and solar amperage.

**Step 3.3: Daemonization (Systemd & Cron)**
* **Action:** Make the system autonomous, persistent, and self-healing.
* **Details:** Wrap your continuous scripts (`bio_capture_loop.py` on the Pi, and `ingestion_api.py` on the LattePanda) in `systemd` services. Schedule your periodic scripts (`sender_daemon.py` hourly, `watchdog_alert.py` every 15 minutes) using `cron` jobs.
* **Purpose:** This ensures all system components survive power cycles and reboot automatically[cite: 47]. 

#### **3. Definition of Done**

You are ready to move to Phase 4 (End-to-End & Chaos Testing) when:
1. The Pi successfully captures live environmental audio, scores it, and correctly saves `.flac` files for biological hits.
2. The sender daemon accurately packs real hardware telemetry into the hourly MessagePack payload.
3. You can manually restart both the Raspberry Pi and the LattePanda, and observe all four scripts resuming their duties automatically without human intervention.