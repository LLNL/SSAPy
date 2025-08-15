import os
import shutil
import tempfile
import pytest

from ssapy.data_utils import ensure_data_downloaded, get_data_dir, requires_data


def test_data_already_exists():
    """Test that function works when data directory already exists."""
    # Reset the global state before test
    import ssapy.data_utils
    ssapy.data_utils._data_checked = False
    
    # This will either use existing data or download it
    # Either way, it should complete without error
    ensure_data_downloaded()
    
    # Verify the data directory was created/exists
    data_dir = os.path.join(os.path.dirname(ssapy.data_utils.__file__), 'data')
    assert os.path.exists(data_dir)
    
    # Verify the global flag was set
    assert ssapy.data_utils._data_checked is True


def test_multiple_calls_only_run_once():
    """Test that multiple calls don't re-download."""
    # Reset the global state before test
    import ssapy.data_utils
    ssapy.data_utils._data_checked = False
    
    # Time the first call
    import time
    start_time = time.time()
    ensure_data_downloaded()
    first_call_time = time.time() - start_time
    
    # Time the second call - should be much faster
    start_time = time.time()
    ensure_data_downloaded()
    second_call_time = time.time() - start_time
    
    # Second call should be nearly instantaneous (< 1ms typically)
    assert second_call_time < first_call_time / 10


def test_requires_data_decorator():
    """Test the @requires_data decorator."""
    
    @requires_data
    def dummy_function():
        return "success"
    
    # This should work without error
    result = dummy_function()
    assert result == "success"
    
    # Verify data directory exists after decorated function runs
    data_dir = get_data_dir()
    assert os.path.exists(data_dir)


def test_get_data_dir():
    """Test get_data_dir function."""
    data_path = get_data_dir()
    
    # Should return a valid path ending in 'data'
    assert data_path.endswith("data")
    assert os.path.exists(data_path)


def test_lazy_datadir_access():
    """Test that the lazy datadir in __init__.py works correctly."""
    import ssapy
    
    # Accessing datadir should work
    datadir_path = str(ssapy.datadir)
    assert datadir_path.endswith("data")
    assert os.path.exists(datadir_path)


def test_data_directory_has_files():
    """Test that the downloaded data directory actually contains files."""
    ensure_data_downloaded()
    
    data_dir = get_data_dir()
    files = os.listdir(data_dir)
    
    # Should have some files after download
    assert len(files) > 0
    print(f"Data directory contains: {files}")


# Optional: Test with a clean environment
def test_fresh_download():
    """Test downloading to a temporary location to verify download works."""
    # Create a temporary directory to test fresh download
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a minimal test version that downloads to temp location
        test_data_dir = os.path.join(temp_dir, 'data')
        
        # Manual test of the download logic
        repo_url = "https://github.com/LLNL/SSAPy.git"
        commit_hash = "0ea3174"
        temp_repo_dir = os.path.join(temp_dir, "_temp_ssapy_repo")
        
        try:
            import subprocess
            subprocess.run(["git", "clone", repo_url, temp_repo_dir], 
                         check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", temp_repo_dir, "checkout", commit_hash], 
                         check=True, capture_output=True, text=True)
            
            source_data_dir = os.path.join(temp_repo_dir, "ssapy", "data")
            assert os.path.exists(source_data_dir), f"Data not found at {source_data_dir}"
            
            print("Download test successful - data exists in repo")
            
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Git operations failed (possibly no network): {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
        finally:
            if os.path.exists(temp_repo_dir):
                shutil.rmtree(temp_repo_dir)