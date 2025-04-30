import os
import subprocess

def ensure_data_downloaded():
    """
    Lazily clones the ssapy/data directory from GitHub if it doesn't already exist.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir) and os.listdir(data_dir):
        # print(f"[ssapy] Data directory already exists at {data_dir}")
        return

    # print("[ssapy] Downloading data directory from GitHub...")

    repo_url = "https://github.com/LLNL/SSAPy.git"
    commit_hash = "0ea3174"
    temp_repo_dir = os.path.join(os.path.dirname(__file__), "_temp_ssapy_repo")

    try:
        subprocess.run(["git", "clone", repo_url, temp_repo_dir], check=True)
        subprocess.run(["git", "-C", temp_repo_dir, "checkout", commit_hash], check=True)

        source_data_dir = os.path.join(temp_repo_dir, "ssapy", "data")

        if not os.path.exists(source_data_dir):
            raise FileNotFoundError(f"Expected data directory not found at {source_data_dir}")

        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["cp", "-r", source_data_dir + "/.", data_dir], check=True)

        print(f"[ssapy] Data successfully downloaded to {data_dir}")

    except subprocess.CalledProcessError as e:
        # print(f"[ssapy] Error downloading data: {e}")
        raise RuntimeError("Failed to clone and copy data from GitHub.")

    finally:
        subprocess.run(["rm", "-rf", temp_repo_dir], check=True)
