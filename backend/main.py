import os
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
import cv2
import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.database import init_db, log_alert, get_recent_alerts, get_db_connection
from backend.model_runner import ModelRunner, TORCH_AVAILABLE
from backend.alert_router import trigger_alert

# Global Settings & Thread Locks
class PipelineSettings(BaseModel):
    resolution: int = 112
    stream_fps: int = 30
    inference_interval: int = 500
    threshold: float = 0.85
    source: str = "synthetic"
    rtsp_url: str = ""

settings = PipelineSettings()
settings_lock = threading.Lock()

# Thread state variables
freshest_frame = None
frame_lock = threading.Lock()
pipeline_running = True

# Buffers for spatiotemporal analysis (sliding windows of 32 frames)
raw_buffer = deque(maxlen=32)
preprocessed_buffer = deque(maxlen=32)
buffer_lock = threading.Lock()

# Performance scoring & simulation
current_score = 0.0
intrusion_active = False
intrusion_end_time = 0.0
last_alert_time = 0.0
ALERT_COOLDOWN_SEC = 8.0  # Avoid spamming duplicate alerts

# Instantiations
runner = ModelRunner()

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BACKEND_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_running
    # Startup: Initialize Database
    init_db()
    
    # Start threads
    pipeline_running = True
    ingest_t = threading.Thread(target=ingestion_thread_loop, daemon=True)
    inference_t = threading.Thread(target=inference_thread_loop, daemon=True)
    
    ingest_t.start()
    inference_t.start()
    print("[AEGIS] Ingestion and Inference threads launched.")
    
    yield
    
    # Shutdown
    pipeline_running = False
    print("[AEGIS] Server shutting down. Stopping threads...")

app = FastAPI(lifespan=lifespan, title="Aegis Spatiotemporal Surveillance Backend")

# Serve Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==========================================
# 1. Ingestion Thread (Drops frames to prevent lag)
# ==========================================
def generate_synthetic_frame(tick):
    """Generates a synthetic camera stream using NumPy/OpenCV draw operations."""
    global intrusion_active, intrusion_end_time
    
    # Draw standard 640x480 video frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw static grid pattern (simulates security cameras scanning lines)
    for y in range(0, 480, 40):
        cv2.line(frame, (0, y), (640, y), (12, 16, 24), 1)
    for x in range(0, 640, 40):
        cv2.line(frame, (x, 0), (x, 480), (12, 16, 24), 1)
        
    # Check if intrusion simulation is active
    is_intruding = False
    if intrusion_active:
        if time.time() < intrusion_end_time:
            is_intruding = True
        else:
            intrusion_active = False
            
    # Draw background shapes (e.g. blowing trees or shadows)
    # Slow natural periodic movement
    sway = int(12 * np.sin(tick * 0.05))
    cv2.circle(frame, (120 + sway, 150), 35, (20, 24, 32), -1) # "Bush" shadow
    cv2.line(frame, (120 + sway, 150), (120 + sway, 240), (15, 18, 25), 3)
    
    # Draw moving subject (human/intruder or animal)
    if is_intruding:
        # Rapid, erratic movement across the center screen
        pos_x = int(320 + 150 * np.cos(tick * 0.25))
        pos_y = int(240 + 60 * np.sin(tick * 0.45))
        
        # Intruder body (reddish glow)
        cv2.circle(frame, (pos_x, pos_y), 24, (25, 25, 180), -1)
        cv2.circle(frame, (pos_x, pos_y - 35), 12, (30, 30, 200), -1) # head
        cv2.line(frame, (pos_x, pos_y), (pos_x, pos_y + 40), (25, 25, 180), 4) # spine
        cv2.line(frame, (pos_x, pos_y + 40), (pos_x - 15, pos_y + 80), (25, 25, 180), 3) # leg L
        cv2.line(frame, (pos_x, pos_y + 40), (pos_x + 15, pos_y + 80), (25, 25, 180), 3) # leg R
        
        # Threat indicator text
        cv2.putText(frame, "MOTION THREAT DETECTED: HIGH ACTIVITY", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Normal tiny slow movement (dog / small pet or normal activity)
        pos_x = int(500 + 40 * np.cos(tick * 0.03))
        pos_y = int(380 + 10 * np.sin(tick * 0.02))
        cv2.circle(frame, (pos_x, pos_y), 8, (40, 45, 50), -1) # Small pet
        cv2.putText(frame, "SYSTEM MONITORED - SECURE", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
    # Overlay HUD Telemetry information
    cv2.putText(frame, f"AEGIS SIMULATOR [CAM_01]", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (420, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
    # Scanning laser line sweeping down
    sweep_y = int(240 + 238 * np.sin(tick * 0.08))
    color = (0, 0, 180) if is_intruding else (180, 100, 0)
    cv2.line(frame, (5, sweep_y), (635, sweep_y), color, 1)
    
    return frame

def ingestion_thread_loop():
    """Decoupled thread continuously reading stream source and caching the latest frame."""
    global freshest_frame, pipeline_running
    
    cap = None
    last_source = None
    last_rtsp_url = None
    tick = 0
    
    while pipeline_running:
        with settings_lock:
            curr_source = settings.source
            fps = settings.stream_fps
            rtsp_url = settings.rtsp_url
            
        # Re-initialize capture if stream source/RTSP URL changes, or if active source has no open capture
        need_init = (curr_source != last_source) or (curr_source == "rtsp" and rtsp_url != last_rtsp_url) or (curr_source in ("webcam", "rtsp") and cap is None)
        
        if need_init:
            if cap is not None:
                cap.release()
                cap = None
                
            if curr_source == "webcam":
                print("[AEGIS Ingestion] Connecting to system webcam...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("[WARNING] Webcam not accessible. Defaulting back to Synthetic Cam.")
                    with settings_lock:
                        settings.source = "synthetic"
                    curr_source = "synthetic"
                    cap = None
            elif curr_source == "rtsp":
                if rtsp_url:
                    print(f"[AEGIS Ingestion] Connecting to CCTV RTSP stream: {rtsp_url}")
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        print(f"[WARNING] CCTV stream at {rtsp_url} not accessible. Defaulting to Synthetic Cam.")
                        with settings_lock:
                            settings.source = "synthetic"
                        curr_source = "synthetic"
                        cap = None
                else:
                    print("[WARNING] RTSP selected but no URL provided. Defaulting to Synthetic Cam.")
                    with settings_lock:
                        settings.source = "synthetic"
                    curr_source = "synthetic"
                    cap = None
            else:
                print("[AEGIS Ingestion] Switched to Synthetic Simulator stream.")
                
        last_source = curr_source
        last_rtsp_url = rtsp_url
        
        # Read or generate frame
        frame = None
        if curr_source in ("webcam", "rtsp") and cap is not None:
            ret, raw = cap.read()
            if ret:
                frame = raw
            else:
                # If reading fails, release capture to trigger a reconnect and fall back to synthetic
                print(f"[WARNING] Failed to read frame from {curr_source}. Releasing capture and falling back to synthetic.")
                cap.release()
                cap = None
                time.sleep(0.5)  # Pause to avoid hot looping if connection drops
                frame = generate_synthetic_frame(tick)
                tick += 1
        else:
            frame = generate_synthetic_frame(tick)
            tick += 1
            
        if frame is not None:
            # Overwrite the shared variable with lock
            with frame_lock:
                freshest_frame = frame.copy()
                
        # Sleep to regulate camera frame rate (e.g. 30 FPS = 33ms sleep)
        time.sleep(1.0 / max(fps, 1))
        
    if cap is not None:
        cap.release()

# ==========================================
# 2. Inference Thread (Samples cache periodically)
# ==========================================
def inference_thread_loop():
    """Wakes up periodically, copies freshest_frame, normalizes, and runs SlowFast prediction."""
    global freshest_frame, current_score, pipeline_running, last_alert_time
    
    last_resolution = None

    while pipeline_running:
        with settings_lock:
            interval = settings.inference_interval
            resolution = settings.resolution
            threshold = settings.threshold
            
        # Clear buffers if resolution changes to prevent tensor stacking errors
        if last_resolution is not None and resolution != last_resolution:
            with buffer_lock:
                raw_buffer.clear()
                preprocessed_buffer.clear()
                print(f"[AEGIS] Resolution changed from {last_resolution} to {resolution}. Inference buffers cleared.")
        last_resolution = resolution
            
        start_time = time.time()
        
        # 1. Grab fresh frame from memory safely
        local_frame = None
        with frame_lock:
            if freshest_frame is not None:
                local_frame = freshest_frame.copy()
                
        if local_frame is not None:
            # 2. Preprocess (resize and normalize)
            prep_tensor = runner.preprocess(local_frame, (resolution, resolution))
            
            # 3. Store in queue locks
            with buffer_lock:
                raw_buffer.append(local_frame)
                preprocessed_buffer.append(prep_tensor)
                
                # Check if buffer is filled with 32 frames for spatiotemporal analysis
                # If we don't have 32 frames yet, duplicate the first ones to fill the buffer
                while len(raw_buffer) < 32:
                    raw_buffer.append(local_frame)
                    preprocessed_buffer.append(prep_tensor)
                    
                # Create snapshot copies for prediction to keep buffer lock short
                raw_list = list(raw_buffer)
                pre_list = list(preprocessed_buffer)
                
            # 4. Predict
            score, latency = runner.predict(raw_list, pre_list)
            
            if intrusion_active and time.time() < intrusion_end_time:
                score = 0.95
                
            current_score = score
            
            # 5. Evaluate anomaly trigger threshold
            if score >= threshold:
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN_SEC:
                    last_alert_time = now
                    print(f"\n[ALERT TRIGGERED] Anomaly Score: {score:.3f} >= Threshold: {threshold:.2f}")
                    # Dispatch to alert router asynchronously to avoid blocking inference thread
                    threading.Thread(
                        target=trigger_alert, 
                        args=(raw_list, score), 
                        daemon=True
                    ).start()
                    
        # Sleep for defined interval, adjusting for execution duration to maintain exact intervals
        elapsed = time.time() - start_time
        sleep_dur = max(0.005, (interval / 1000.0) - elapsed)
        time.sleep(sleep_dur)

# ==========================================
# 3. FastAPI REST Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serves index.html dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html

def mjpeg_generator():
    """Generator for MJPEG stream, embedding live anomaly status HUD overlays."""
    global freshest_frame, current_score
    
    while pipeline_running:
        with frame_lock:
            if freshest_frame is None:
                frame_to_stream = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_to_stream, "Awaiting Camera Feed...", (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                frame_to_stream = freshest_frame.copy()
                
        # Draw HUD overlays based on anomaly score
        h, w, _ = frame_to_stream.shape
        
        with settings_lock:
            thresh = settings.threshold
            
        color = (0, 0, 255) if current_score >= thresh else (0, 255, 0)
        status = "ALERT" if current_score >= thresh else "SECURE"
        
        # Border frame indicator
        cv2.rectangle(frame_to_stream, (0, 0), (w - 1, h - 1), color, 4)
        
        # Score HUD pill at bottom-right
        score_text = f"ANOMALY: {current_score:.2f}"
        cv2.rectangle(frame_to_stream, (w - 200, h - 40), (w - 10, h - 10), (10, 14, 20), -1)
        cv2.rectangle(frame_to_stream, (w - 200, h - 40), (w - 10, h - 10), color, 1)
        cv2.putText(frame_to_stream, score_text, (w - 185, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
        # Status text top-right
        cv2.rectangle(frame_to_stream, (w - 110, 10), (w - 10, 35), color, -1)
        cv2.putText(frame_to_stream, status, (w - 95, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if current_score >= thresh else (255, 255, 255), 2)
                    
        # Compress to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_to_stream)
        if not ret:
            time.sleep(0.03)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04) # Stream around ~25 FPS to browser

@app.get("/api/stream")
async def video_stream(source: str = None):
    """MJPEG stream endpoint."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/metrics")
async def get_metrics():
    """Retrieve real-time metrics and state metadata."""
    return {
        "current_score": current_score,
        "pytorch_installed": TORCH_AVAILABLE,
        "model_mocked": runner.is_mock,
        "raw_score": getattr(runner, "last_raw_score", 0.0),
        "motion_score": getattr(runner, "last_motion_score", 0.0),
        "motion_gate": getattr(runner, "last_motion_gate", 0.0)
    }

@app.get("/api/alerts")
async def get_alerts():
    """Retrieve history of security alerts logged in SQLite."""
    return get_recent_alerts(limit=30)

@app.post("/api/settings")
async def post_settings(new_settings: PipelineSettings):
    """Dynamically updates active processing variables."""
    global settings
    with settings_lock:
        resolution_changed = (settings.resolution != new_settings.resolution)
        settings.resolution = new_settings.resolution
        settings.stream_fps = new_settings.stream_fps
        settings.inference_interval = new_settings.inference_interval
        settings.threshold = new_settings.threshold
        settings.source = new_settings.source
        settings.rtsp_url = new_settings.rtsp_url
        
    if resolution_changed:
        with buffer_lock:
            raw_buffer.clear()
            preprocessed_buffer.clear()
            print("[AEGIS] Resolution changed. Cleared temporal buffers to prevent size mismatch.", flush=True)
            
    return {"status": "success", "settings": new_settings}

@app.post("/api/inject_intrusion")
async def inject_intrusion():
    """Forces an intrusion animation block on synthetic stream."""
    global intrusion_active, intrusion_end_time
    intrusion_active = True
    intrusion_end_time = time.time() + 6.0  # Lasts 6 seconds
    return {"status": "anomaly_injected", "duration_sec": 6.0}

@app.post("/api/clear_alerts")
async def clear_alerts():
    """Removes all alert logs from the local database and wipes the saved clips."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alerts")
        conn.commit()
        conn.close()
        
        # Delete clip files on disk
        clips_dir = os.path.join(BACKEND_DIR, "static", "clips")
        if os.path.exists(clips_dir):
            for f in os.listdir(clips_dir):
                f_path = os.path.join(clips_dir, f)
                if os.path.isfile(f_path) and f.endswith(".mp4"):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass
        return {"status": "success", "message": "Alert history cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def mjpeg_clip_generator(clip_path):
    """Generator that reads an MP4 clip and streams it as looping MJPEG."""
    # Resolve absolute path
    abs_path = os.path.join(BACKEND_DIR, clip_path)
    if not os.path.exists(abs_path):
        abs_path = os.path.join(os.path.dirname(BACKEND_DIR), clip_path)
        
    while True:
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open clip file: {abs_path}")
            # Yield dummy error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Clip File Missing/Error", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1.0)
            break
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Loop video
                
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.16)  # ~6 FPS
            
        cap.release()

@app.get("/api/alerts/{alert_id}/clip")
async def get_alert_clip(alert_id: int):
    """Serve a looping MJPEG stream of the saved anomaly video clip."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT clip_path FROM alerts WHERE id = ?", (alert_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not row["clip_path"]:
        raise HTTPException(status_code=404, detail="Alert or clip not found")
        
    clip_path = row["clip_path"]
    return StreamingResponse(
        mjpeg_clip_generator(clip_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

