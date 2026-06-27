import sys
import os
import subprocess

def check_dependencies():
    """Checks if key dependencies are installed, prints helpful warnings."""
    print("=" * 60)
    print(" AEGIS SURVEILLANCE SYSTEM - PRE-FLIGHT COMPATIBILITY CHECK")
    print("=" * 60)
    
    missing = []
    
    # Check FastAPI
    try:
        import fastapi
        print("  [+] FastAPI: Installed")
    except ImportError:
        missing.append("fastapi")
        print("  [X] FastAPI: Missing")

    # Check Uvicorn
    try:
        import uvicorn
        print("  [+] Uvicorn: Installed")
    except ImportError:
        missing.append("uvicorn")
        print("  [X] Uvicorn: Missing")

    # Check OpenCV
    try:
        import cv2
        print("  [+] OpenCV (cv2): Installed")
    except ImportError:
        missing.append("opencv-python")
        print("  [X] OpenCV (cv2): Missing")

    # Check PyTorch
    try:
        import torch
        import torchvision
        print(f"  [+] PyTorch: Installed (version {torch.__version__})")
    except ImportError:
        print("  [i] PyTorch: Missing (falling back to OpenCV motion inference)")

    print("-" * 60)
    
    if missing:
        print("[!] ERROR: Some critical web dependencies are missing.")
        print(f"Please install them via pip before running the system:\n")
        print(f"  pip install {' '.join(missing)}")
        print("\nNote: To test the full SlowFast network model weights on CPU, run:")
        print("  pip install torch torchvision")
        print("=" * 60)
        sys.exit(1)
    else:
        print("All basic dependencies met. Booting system...")
        print("=" * 60)

def load_env_file():
    """Manually reads and loads environment variables from a .env file if present."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        print("  [+] Loading environment variables from .env...")
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

def main():
    load_env_file()
    check_dependencies()
    
    # Add root folder to python path to avoid import issues
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    try:
        print("Starting FastAPI Uvicorn Server at http://127.0.0.1:8000 ...")
        print("Open this URL in your browser to view the interactive dashboard simulator.")
        print("Press Ctrl+C to terminate the application.")
        print("-" * 60)
        
        # Start uvicorn
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n[AEGIS] System stopped by homeowner (Ctrl+C).")
    except Exception as e:
        print(f"\n[AEGIS] Execution failed: {e}")

if __name__ == "__main__":
    main()
