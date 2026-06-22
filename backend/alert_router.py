import os
import cv2
import requests
import json
from datetime import datetime
from backend.database import log_alert

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BACKEND_DIR, "static", "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

# Load .env file manually if it exists relative to the backend root directory
def load_env_file():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                os.environ[k] = v

load_env_file()

# Settings (loaded from env)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")

def create_anomaly_clip(raw_frames, filename_prefix="anomaly"):
    """
    Saves the list of raw frames (which covers the anomaly timeline) 
    into a local MP4 file. Returns the relative file path.
    """
    if not raw_frames:
        print("[WARNING] Cannot create clip: Raw frame list is empty.")
        return None
        
    # Generate unique filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp_str}.mp4"
    file_path = os.path.join(CLIPS_DIR, filename)
    relative_path = f"static/clips/{filename}"
    
    # Get frame dimensions
    height, width, layers = raw_frames[0].shape
    
    # Standard web browsers play H264/MP4 best. 
    # Try different codecs in order of compatibility
    codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
    out = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(file_path, fourcc, 6.0, (width, height))
            if out.isOpened():
                break
        except Exception:
            continue
            
    if out is None or not out.isOpened():
        print("[ERROR] OpenCV VideoWriter failed to initialize with all attempted codecs.")
        return None
        
    for frame in raw_frames:
        out.write(frame)
    out.release()
    print(f"Saved 5-second anomaly clip to: {file_path}")
    return relative_path

def get_llm_summary(timestamp, score):
    """
    Queries Gemini API to analyze the context of the anomaly and
    provide a localized emergency script or log entry.
    Falls back to a local rule-based system if API is unavailable.
    """
    prompt = f"""
    An anomaly has been detected at the Backyard Camera.
    Timestamp: {timestamp}
    Anomaly Score: {score:.2f} (Threshold is 0.85)

    Cross-reference default emergency rules:
    If post-midnight (00:00 - 06:00), notify the homeowner immediately and prepare a localized emergency SMS script.
    Otherwise, log the event and outline response actions.

    Based on this data, write a concise alert summary (max 3 sentences) in a direct, professional, urgent tone. Do not use markdown tags, just plain text.
    """
    
    # If API key is available, attempt to call Gemini
    if GEMINI_API_KEY:
        try:
            # Try importing the modern google-genai client first
            try:
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                if response.text:
                    return response.text.strip()
            except (ImportError, Exception):
                # Fall back to google-generativeai
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
        except Exception as e:
            print(f"[WARNING] Gemini API call failed: {e}. Falling back to rule-based parser.")
            
    # Local Rule-based Fallback
    try:
        dt = datetime.fromisoformat(timestamp)
    except Exception:
        dt = datetime.now()
        
    hour = dt.hour
    is_post_midnight = 0 <= hour <= 6
    
    if is_post_midnight:
        summary = (
            f"[CRITICAL ALERT] Anomaly detected at {dt.strftime('%H:%M:%S')} (Post-Midnight). "
            f"Rule Action: Emergency Alert triggered. Prepared SMS: 'WARNING: Unauthorized motion detected "
            f"in Backyard at {dt.strftime('%H:%M')}. Police dispatch prepared. Check live feed!'"
        )
    else:
        summary = (
            f"[WARNING] Anomaly detected at {dt.strftime('%H:%M:%S')} with score {score:.2f}. "
            f"Rule Action: Event logged. Stream monitoring active. No immediate dispatch needed."
        )
        
    return summary

def dispatch_notification(score, summary, clip_url=""):
    """
    Sends Webhook notifications to external chat services (Telegram/Discord)
    if configured.
    """
    if not ALERT_WEBHOOK_URL:
        print("[INFO] Webhook URL not configured. Skipping external notification.")
        return False
        
    payload = {
        "text": f"⚠️ **SMART SURVEILLANCE ALERT** ⚠️\n\nScore: {score:.2f}\nSummary: {summary}\nClip Link: {clip_url}"
    }
    
    try:
        response = requests.post(
            ALERT_WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5
        )
        if response.status_code in (200, 201, 204):
            print("Webhook alert successfully dispatched.")
            return True
        else:
            print(f"[WARNING] Webhook returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to send webhook alert: {e}")
        return False

def trigger_alert(raw_frames, score):
    """
    Main orchestrator for when an anomaly is triggered.
    Saves clip -> Queries LLM summary -> Logs in SQLite -> Dispatches Webhook.
    """
    timestamp = datetime.now().isoformat()
    
    # 1. Save clip
    clip_url = create_anomaly_clip(raw_frames)
    
    # 2. Query Gemini LLM (or rule fallback)
    summary = get_llm_summary(timestamp, score)
    
    # 3. Dispatch alert
    sent = dispatch_notification(score, summary, clip_url)
    status = "sent" if sent else "logged"
    
    # 4. Log to SQLite
    alert_id = log_alert(
        anomaly_score=score,
        clip_path=clip_url,
        llm_summary=summary,
        status=status
    )
    
    print(f"Alert [{alert_id}] logged successfully. Status: {status}")
    return {
        "id": alert_id,
        "timestamp": timestamp,
        "score": score,
        "clip_path": clip_url,
        "summary": summary,
        "status": status
    }

if __name__ == "__main__":
    # Test alert router
    import numpy as np
    print("Testing AlertRouter...")
    mock_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(30)]
    result = trigger_alert(mock_frames, 0.94)
    print("Resulting Alert:", result)
