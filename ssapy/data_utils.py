import os
import subprocess
import threading
from functools import wraps

# Global state to track if data has been checked/downloaded
_data_checked = False
_data_lock = threading.Lock()

def ensure_data_downloaded():
    """
    Lazily clones the ssapy/data directory from GitHub if it doesn't already exist.
    Thread-safe and only runs once per process.
    """
    global _data_checked
    
    # Quick check without lock for performance
    if _data_checked:
        return
        
    # Thread-safe check and download
    with _data_lock:
        if _data_checked:  # Double-check pattern
            return
            
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if os.path.exists(data_dir) and os.listdir(data_dir):
            _data_checked = True
            return
            
        print("[ssapy] Downloading data directory from GitHub...")
        repo_url = "https://github.com/LLNL/SSAPy.git"
        commit_hash = "0ea3174"
        temp_repo_dir = os.path.join(os.path.dirname(__file__), "_temp_ssapy_repo")
        
        try:
            subprocess.run(["git", "clone", repo_url, temp_repo_dir], 
                         check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", temp_repo_dir, "checkout", commit_hash], 
                         check=True, capture_output=True, text=True)
            
            source_data_dir = os.path.join(temp_repo_dir, "ssapy", "data")
            if not os.path.exists(source_data_dir):
                raise FileNotFoundError(f"Expected data directory not found at {source_data_dir}")
                
            os.makedirs(data_dir, exist_ok=True)
            subprocess.run(["cp", "-r", source_data_dir + "/.", data_dir], 
                         check=True, capture_output=True, text=True)
            print(f"[ssapy] Data successfully downloaded to {data_dir}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone and copy data from GitHub: {e.stderr}")
        finally:
            if os.path.exists(temp_repo_dir):
                subprocess.run(["rm", "-rf", temp_repo_dir], check=True)
                
        _data_checked = True

def requires_data(func):
    """
    Decorator that ensures data is downloaded before running a function.
    Use this on functions that need the data directory.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ensure_data_downloaded()
        return func(*args, **kwargs)
    return wrapper

def get_data_dir():
    """
    Get the data directory path, ensuring data is downloaded first.
    """
    ensure_data_downloaded()
    return os.path.join(os.path.dirname(__file__), 'data')