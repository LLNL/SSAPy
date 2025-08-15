import os
import shutil
import tempfile
import pytest
from pathlib import Path

from ssapy.data_loader import ensure_data_downloaded, get_data_dir, requires_data, is_data_available


def test_data_chunks_available():
    """Test that chunk files are available in the installed package."""
    # Check if chunks exist in the package
    import ssapy
    package_dir = Path(ssapy.__file__).parent
    chunk_files = list(package_dir.glob("ssapy_data_chunk_*.tar.gz"))
    
    assert len(chunk_files) > 0, f"No chunk files found in {package_dir}"
    print(f"Found {len(chunk_files)} chunk files")
    
    # Verify chunks are under 100 MB
    for chunk_file in chunk_files:
        size_mb = chunk_file.stat().st_size / (1024 * 1024)
        assert size_mb < 100, f"Chunk {chunk_file.name} is {size_mb:.1f} MB (exceeds 100 MB limit)"
        print(f"  {chunk_file.name}: {size_mb:.1f} MB ✓")


def test_data_extraction_works():
    """Test that chunks can be reassembled and extracted."""
    # Reset the global state before test
    import ssapy.data_loader
    ssapy.data_loader._data_loader._extracted = False
    ssapy.data_loader._data_loader._data_dir = None
    
    # This should reassemble chunks and extract data
    ensure_data_downloaded()
    
    # Verify the data directory was created
    data_dir = get_data_dir()
    assert data_dir is not None
    assert os.path.exists(data_dir)
    assert data_dir.endswith("data")


def test_multiple_calls_only_extract_once():
    """Test that multiple calls don't re-extract chunks."""
    # Reset the global state before test
    import ssapy.data_loader
    ssapy.data_loader._data_loader._extracted = False
    ssapy.data_loader._data_loader._data_dir = None
    
    # Time the first call (should do extraction)
    import time
    start_time = time.time()
    ensure_data_downloaded()
    first_call_time = time.time() - start_time
    
    # Time the second call - should be much faster (cached)
    start_time = time.time()
    ensure_data_downloaded()
    second_call_time = time.time() - start_time
    
    # Second call should be nearly instantaneous
    assert second_call_time < first_call_time / 10
    print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")


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
    print(f"Data directory: {data_path}")


def test_is_data_available():
    """Test is_data_available function."""
    # Should return True after chunks are extracted
    available = is_data_available()
    assert available is True, "Data should be available after chunk extraction"


def test_lazy_datadir_access():
    """Test that the lazy datadir in __init__.py works correctly."""
    import ssapy
    
    # Accessing datadir should work and trigger chunk extraction
    datadir_path = str(ssapy.datadir)
    assert datadir_path.endswith("data")
    assert os.path.exists(datadir_path)
    print(f"Lazy datadir access works: {datadir_path}")


def test_data_directory_has_expected_files():
    """Test that the extracted data directory contains expected files."""
    ensure_data_downloaded()
    
    data_dir = get_data_dir()
    data_path = Path(data_dir)
    files = list(data_path.iterdir())
    
    # Should have the expected 22 files after extraction
    file_count = len([f for f in files if f.is_file()])
    assert file_count > 0, "Data directory should contain files after extraction"
    print(f"Data directory contains {file_count} files")
    
    # Check for some expected file types
    egm_files = list(data_path.glob("*.egm"))
    bsp_files = list(data_path.glob("*.bsp")) 
    cof_files = list(data_path.glob("*.cof"))
    
    print(f"Found: {len(egm_files)} .egm files, {len(bsp_files)} .bsp files, {len(cof_files)} .cof files")
    
    # Should have at least some of the expected file types
    assert len(egm_files) + len(bsp_files) + len(cof_files) > 0, "Should have some orbital data files"


def test_get_data_file():
    """Test get_data_file function for accessing specific files."""
    from ssapy.data_loader import get_data_file
    
    ensure_data_downloaded()
    
    # Look for a file we know should exist
    data_dir = Path(get_data_dir())
    actual_files = [f.name for f in data_dir.iterdir() if f.is_file()]
    
    if actual_files:
        # Test getting a file that exists
        test_file = actual_files[0]
        file_path = get_data_file(test_file)
        assert file_path is not None, f"Should find existing file {test_file}"
        assert file_path.exists(), f"Returned path should exist: {file_path}"
        print(f"Successfully found file: {test_file}")
        
        # Test getting a file that doesn't exist
        nonexistent_file = get_data_file("nonexistent_file.txt")
        assert nonexistent_file is None, "Should return None for nonexistent file"


def test_list_data_files():
    """Test list_data_files function."""
    from ssapy.data_loader import list_data_files
    
    ensure_data_downloaded()
    
    # List all files
    all_files = list_data_files()
    assert len(all_files) > 0, "Should find some data files"
    print(f"Found {len(all_files)} total data files")
    
    # List specific file types
    egm_files = list_data_files("*.egm")
    print(f"Found {len(egm_files)} .egm files")
    
    # All returned items should be Path objects that exist
    for file_path in all_files[:5]:  # Check first 5
        assert isinstance(file_path, Path), "Should return Path objects"
        assert file_path.exists(), f"File should exist: {file_path}"


def test_thread_safety():
    """Test that the data loader is thread-safe."""
    import threading
    import time
    
    results = []
    
    def worker():
        try:
            ensure_data_downloaded()
            data_dir = get_data_dir()
            results.append(data_dir)
        except Exception as e:
            results.append(f"Error: {e}")
    
    # Reset state
    import ssapy.data_loader
    ssapy.data_loader._data_loader._extracted = False
    ssapy.data_loader._data_loader._data_dir = None
    
    # Start multiple threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # All threads should get the same data directory
    assert len(results) == 3, "All threads should complete"
    assert all(isinstance(r, str) and r.endswith("data") for r in results), f"All should succeed: {results}"
    assert len(set(results)) == 1, f"All threads should get same directory: {results}"
    print(f"Thread safety test passed: {results[0]}")


@pytest.mark.skipif(not os.environ.get("SSAPY_TEST_PERFORMANCE"), 
                   reason="Performance test only runs when SSAPY_TEST_PERFORMANCE is set")
def test_extraction_performance():
    """Test extraction performance (optional)."""
    import time
    
    # Reset state to force fresh extraction
    import ssapy.data_loader
    ssapy.data_loader._data_loader._extracted = False
    ssapy.data_loader._data_loader._data_dir = None
    if ssapy.data_loader._data_loader._temp_dir:
        shutil.rmtree(ssapy.data_loader._data_loader._temp_dir)
        ssapy.data_loader._data_loader._temp_dir = None
    
    # Time the extraction
    start_time = time.time()
    ensure_data_downloaded()
    extraction_time = time.time() - start_time
    
    print(f"Chunk reassembly and extraction took: {extraction_time:.2f} seconds")
    
    # Should complete in reasonable time (adjust threshold as needed)
    assert extraction_time < 60, f"Extraction took too long: {extraction_time:.2f}s"