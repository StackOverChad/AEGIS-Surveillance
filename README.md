# Aegis Surveillance

Aegis Surveillance is an advanced, AI-powered home security and CCTV monitoring system. The project is named after the "Aegis"—the mythological shield carried by Athena and Zeus, symbolizing ultimate protection, security, and defense for your home.

## Features
- **Real-Time Anomaly Detection**: Uses a PyTorch SlowFast model to analyze spatiotemporal data (both spatial features and temporal motion) to detect anomalies in video feeds.
- **RTSP & Webcam Support**: Connects seamlessly with IP cameras/CCTVs over the local network via RTSP streams, or uses local webcams for testing.
- **Interactive Dashboard**: A responsive, dynamic frontend dashboard to monitor the video feed, configure AI detection thresholds, and view real-time inference metrics.
- **Motion Gating**: Uses pixel-level motion analysis to aggressively suppress false positives in entirely static scenes.

## Prerequisites
- Python 3.9+
- An IP Camera with an RTSP stream (e.g., CP Plus Illumax) connected to the same network, or a local webcam.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aegis-surveillance.git
   cd aegis-surveillance
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you install `torch` and `torchvision` to enable the full AI model, otherwise the system will fall back to basic motion detection.*

3. **Environment Setup**
   Create a `.env` file in the root directory for your API keys and configuration secrets:
   ```env
   # Add your private keys here
   # SECRET_API_KEY=your_api_key_here
   ```

## Running the System

1. **Start the backend server**
   ```bash
   python run.py
   ```
2. **Access the Dashboard**
   Open your browser and navigate to `http://127.0.0.1:8000`.

## Connecting to CCTV

To connect to your CP Plus Illumax CCTV, enter the RTSP URL in the dashboard settings panel. A typical format looks like:
```text
rtsp://[username]:[password]@[ip_address]:554/cam/realmonitor?channel=1&subtype=0
```
*Tip: If your password contains special characters like `#`, be sure to URL-encode them (e.g., `#` becomes `%23`).*

## Remote Monitoring

If you wish to monitor your home while miles away:
1. **VPN Setup (Recommended)**: Set up a VPN like Tailscale on your home laptop running Aegis, and connect your mobile device to the same VPN.
2. **Secure Tunnels**: Use Cloudflare Tunnels or Ngrok to securely expose your local `8000` port to the internet. 
*(Avoid port-forwarding your CCTV directly to the internet to maintain security).*

## License
MIT License
