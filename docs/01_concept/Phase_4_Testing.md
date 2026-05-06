### **Phase 4: End-to-End & Chaos Testing**

With everything running on the hardware, you test the resilience of the architecture. This final iterative step ensures you are never fighting a hardware bug, a network bug, and a software bug all at the same time. The goal here is to intentionally break the system's environment to prove that the software behaves exactly as designed under adverse conditions.

#### **1. Preparation (Setting the Stage)**

Before initiating chaos tests, ensure your system is currently in a stable, running state.
* **Autonomous State:** Verify that all `systemd` services and `cron` jobs on both the Raspberry Pi 5 and LattePanda Alpha are active and have survived at least one clean reboot.
* **Baseline Health:** Check the LattePanda's master database to ensure you are actively receiving hourly MessagePack payloads containing real telemetry and any biological hits. 

#### **2. Execution Steps**

**Step 4.1: The Network Drop (Simulating Cellular Outage)**
* **Action:** Physically disconnect the Pi from the network or disable the 4G modem[cite: 49].
* **Details:** Leave the Pi offline for several hours, optionally generating artificial noise or biological sounds to trigger the microphone.
* **Validation:** Verify that `bio_capture_loop.py` continues to compress `.flac` files and insert raw 32-bit float vector embeddings into the database with `sync_status = 'pending'`[cite: 50]. This confirms that decoupling the audio inference loop from the network transport ensures that the Pi continues recording and logging data perfectly even during extended cellular network outages.

**Step 4.2: The Reconnect (Simulating Recovery)**
* **Action:** Turn the network back on[cite: 51].
* **Details:** Wait for the next hourly execution of the `sender_daemon.py` `cron` job.
* **Validation:** Verify that `sender_daemon.py` wakes up, packs the pending rows via MessagePack, and securely POSTs them to the LattePanda's `ingestion_api.py`[cite: 51]. Confirm on the LattePanda that the large batch of data was successfully un-packed, validated by Pydantic, inserted into the WAL-enabled master database, and that the edge rows were correctly updated to `sync_status = 'synced'`.

**Step 4.3: The Dead Man's Switch (Simulating Power Failure)**
* **Action:** Physically remove power from the Raspberry Pi. 
* **Details:** Finally, turn the Pi completely off and wait 75 minutes[cite: 52]. This 75-minute threshold accounts for the standard 60-minute sync cycle plus 15 minutes of network grace time.
* **Validation:** Ensure the `watchdog_alert.py` script on the LattePanda correctly queries the master health metrics table and fires your automated Telegram/email alert indicating that the edge node has missed its check-in window.

#### **3. Definition of Done**

You are officially ready to deploy the system to the grey-faced petrel colony in New Zealand when:
1. The system effortlessly handles unexpected network disconnections and seamlessly syncs all backlogged data upon reconnection.
2. Complete power loss safely triggers the passive remote alerts without relying on the dead node to report its own failure.
3. You have absolute confidence in the system's ability to run asynchronously and autonomously.