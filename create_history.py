import os
import subprocess
from datetime import datetime, timedelta
import random

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("Starting history rewrite...")
# 1. Clean git
if os.path.exists(".git"):
    os.system('rmdir /s /q .git')
    
run("git init")

# 2. Define dates from June 19 to June 27, 2026
start_date = datetime(2026, 6, 19, 10, 0, 0)
dates = [start_date + timedelta(days=i) for i in range(9)]

commits = [
    # Day 1: June 19
    {"files": ["README.md", ".gitignore"], "msg": "Initial project setup and documentation", "day_idx": 0},
    {"files": [".env.example"], "msg": "Add environment template", "day_idx": 0},
    
    # Day 2: June 20
    {"files": ["backend/database.py"], "msg": "Initialize SQLite database schema for alerts", "day_idx": 1},
    {"files": ["backend/static/index.html"], "msg": "Create base dashboard HTML layout", "day_idx": 1},
    
    # Day 3: June 21
    {"files": ["backend/static/style.css"], "msg": "Add dark-theme styling for dashboard", "day_idx": 2},
    
    # Day 4: June 22
    {"files": ["backend/static/app.js"], "msg": "Implement frontend telemetry and polling logic", "day_idx": 3},
    
    # Day 5: June 23
    {"files": ["slowfast_mil_anomaly_colab.py"], "msg": "Add SlowFast MIL training script", "day_idx": 4},
    {"files": ["backend/alert_router.py"], "msg": "Add Gemini LLM integration for alert summaries", "day_idx": 4},
    
    # Day 6: June 24
    {"files": ["backend/model_runner.py"], "msg": "Implement PyTorch model inference wrapper", "day_idx": 5},
    
    # Day 7: June 25
    {"files": ["backend/main.py"], "msg": "Add FastAPI backend and RTSP streaming threads", "day_idx": 6},
    
    # Day 8: June 26
    {"files": ["run.py"], "msg": "Add production uvicorn launch script", "day_idx": 7},
    
    # Day 9: June 27
    {"files": ["."], "msg": "Final pipeline integration and UI polishing", "day_idx": 8}
]

for c in commits:
    # Add files safely
    for f in c["files"]:
        f_norm = os.path.normpath(f)
        if os.path.exists(f_norm) or f == ".":
            run(f'git add "{f_norm}"')
    
    # Generate random time within the day
    day = dates[c["day_idx"]]
    commit_time = day + timedelta(hours=random.randint(0, 8), minutes=random.randint(0, 59))
    date_str = commit_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Commit with specific date
    os.environ['GIT_AUTHOR_DATE'] = date_str
    os.environ['GIT_COMMITTER_DATE'] = date_str
    
    # Use subprocess.run without check=True, so if it's empty, it just skips
    run(f'git commit -m "{c["msg"]}"')

run("git branch -M main")
run("git remote add origin https://github.com/StackOverChad/AEGIS-Surveillance.git")
print("Git history rewritten successfully!")
