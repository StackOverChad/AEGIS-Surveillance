import sqlite3
import os
from datetime import datetime

# Paths
DB_NAME = "surveillance.db"
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BACKEND_DIR, DB_NAME)

def get_db_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    os.makedirs(BACKEND_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            clip_path TEXT,
            llm_summary TEXT,
            status TEXT NOT NULL
        )
    """)
    
    # Create system_metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            resolution TEXT NOT NULL,
            stream_fps INTEGER NOT NULL,
            inference_interval INTEGER NOT NULL,
            cpu_usage REAL NOT NULL,
            memory_usage REAL NOT NULL,
            latency_ms REAL NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def log_alert(anomaly_score, clip_path, llm_summary, status="Pending"):
    """Logs a new anomaly alert into the database."""
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO alerts (timestamp, anomaly_score, clip_path, llm_summary, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, anomaly_score, clip_path, llm_summary, status)
    )
    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()
    return alert_id

def get_recent_alerts(limit=50):
    """Retrieves the most recent alerts from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, anomaly_score, clip_path, llm_summary, status FROM alerts ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def log_metrics(resolution, stream_fps, inference_interval, cpu_usage, memory_usage, latency_ms):
    """Logs current performance metrics into the database."""
    timestamp = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO system_metrics (timestamp, resolution, stream_fps, inference_interval, cpu_usage, memory_usage, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, resolution, stream_fps, inference_interval, cpu_usage, memory_usage, latency_ms)
    )
    conn.commit()
    conn.close()

def get_metrics_history(limit=50):
    """Retrieves the history of metrics from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, resolution, stream_fps, inference_interval, cpu_usage, memory_usage, latency_ms 
        FROM system_metrics ORDER BY timestamp DESC LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Automatically initialize database when loaded
if __name__ not in ("__main__",):
    init_db()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
    # Log a dummy alert for testing
    alert_id = log_alert(0.89, "static/clips/test.mp4", "An anomaly was detected. Local emergency scripts generated.", "sent")
    print(f"Logged test alert ID: {alert_id}")
    # Log dummy metrics
    log_metrics("112x112", 30, 500, 12.5, 450.0, 150.2)
    print("Logged test system metrics.")
    print("Recent alerts:", get_recent_alerts(1))
    print("Recent metrics:", get_metrics_history(1))
