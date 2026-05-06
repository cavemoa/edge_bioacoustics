### **Phase 2: The Bridge (VS Code Remote - SSH)**

The goal of this phase is to migrate your locally tested, mocked scripts onto the actual physical devices and establish the secure communication pipeline. Crucially, you will still be using your mocked audio and telemetry data—we are only testing the network and deployment environments here.

#### **1. Preparation (Setting the Stage)**

Before moving any code, you need to establish the secure network and developer tooling that will allow you to work seamlessly across the devices.

* [cite_start]**Tailscale Integration:** Ensure Tailscale is running on your Mint PC, the Raspberry Pi 5, and the LattePanda Alpha[cite: 36]. 
* [cite_start]**Networking Advantage:** Because Tailscale bypasses CGNAT and assigns static 100.x.x.x IPs, your devices are effectively on the same local network regardless of where they are[cite: 37]. You will use these IPs for both SSH and application traffic.
* [cite_start]**IDE Setup:** Install and enable the VS Code Remote - SSH extension on your Linux Mint PC[cite: 38]. Establish SSH key authentication from your Mint PC to both the Raspberry Pi and the LattePanda so you can connect without typing passwords.

#### **2. Execution Steps**

**Step 2.1: Migrating the Hub (LattePanda Alpha)**
* **Action:** Deploy the central receiver to the physical hub.
* **Details:** Using VS Code Remote - SSH, connect to the LattePanda's Tailscale IP. Recreate your Python virtual environment and transfer your mocked `ingestion_api.py` and `watchdog_alert.py` scripts. 
* **Configuration:** Update the `ingestion_api.py` to listen specifically on the LattePanda's static Tailscale IP rather than `localhost`. 

**Step 2.2: Migrating the Edge (Raspberry Pi 5)**
* **Action:** Deploy the edge node logic to the physical Pi.
* [cite_start]**Details:** You SSH into the Raspberry Pi directly from VS Code[cite: 40]. Recreate the virtual environment and transfer `bio_capture_loop.py` and `sender_daemon.py`. 
* [cite_start]**The "Secret Weapon" Workflow:** At this point, your editor, terminal, and AI coding assistants (Gemini/Codex) run on your Mint PC, but the code execution and file saving happen natively on the Pi[cite: 41]. Keep your mock `.wav` file and hardcoded telemetry in place.

**Step 2.3: Cross-Device Network Testing**
* **Action:** Test the transport layer across the actual Tailscale mesh.
* **Details:** Start the FastAPI server on the LattePanda. Then, from your VS Code window connected to the Pi, manually trigger the mocked `sender_daemon.py`. 
* **Validation:** Verify that the Pi successfully reaches the LattePanda over the 100.x.x.x network, authenticates via the API key, and that the LattePanda strictly validates the unpacked MessagePack data with Pydantic before inserting it into the master WAL-enabled database.

#### **3. Definition of Done**

You are ready to move to Phase 3 (Staged Hardware Integration) when:
1. You can instantly open a VS Code workspace on either the Pi or the LattePanda without leaving your Mint PC.
2. The mocked `sender_daemon.py` running natively on the Pi successfully transmits its MessagePack payload to the LattePanda over the Tailscale network.
3. The `watchdog_alert.py` running on the LattePanda successfully monitors the newly integrated master database.