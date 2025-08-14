"""
SSAPy Runtime Data Loader (Chunked Version)

This module handles loading data files from chunked tar archives at runtime.
"""

import os
import sys
import tarfile
import atexit
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Optional
from functools import wraps


class SSAPyDataLoader:
    """
    Handles loading and managing SSAPy data files from chunked tar archives
    """
    
    _instance = None
    _data_dir = None
    _temp_dir = None
    _extracted = False
    _lock = threading.Lock()
    
    CHUNK_PREFIX = "ssapy_data_chunk_"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Register cleanup function to run on exit"""
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Clean up temporary directories on exit"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def _find_chunk_files(self) -> list:
        """
        Find chunk files in the package installation
        
        Returns:
            List of chunk file paths, empty if not found
        """
        # Try to find chunks relative to this module
        try:
            import ssapy
            package_dir = Path(ssapy.__file__).parent
            chunk_pattern = f"{self.CHUNK_PREFIX}*.tar.gz"
            chunk_files = sorted(list(package_dir.glob(chunk_pattern)))
            
            if chunk_files:
                return chunk_files
                
        except ImportError:
            pass
        
        # Fallback: look in common locations
        search_dirs = [
            Path.cwd() / "ssapy",
            Path(__file__).parent,
        ]
        
        for search_dir in search_dirs:
            chunk_pattern = f"{self.CHUNK_PREFIX}*.tar.gz"
            chunk_files = sorted(list(search_dir.glob(chunk_pattern)))
            if chunk_files:
                return chunk_files
                
        return []
    
    def _reassemble_chunks(self, chunk_files: list, output_path: Path) -> bool:
        """
        Reassemble chunk files into a single archive
        
        Args:
            chunk_files: List of chunk file paths
            output_path: Path for reassembled archive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"[ssapy] Reassembling {len(chunk_files)} chunks...")
            
            with open(output_path, 'wb') as output:
                for chunk_file in chunk_files:
                    with open(chunk_file, 'rb') as chunk:
                        while True:
                            data = chunk.read(8192)
                            if not data:
                                break
                            output.write(data)
                            
            print(f"[ssapy] Chunks reassembled to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error reassembling chunks: {e}")
            return False
    
    def _extract_archive(self, archive_path: Path, target_dir: Path) -> bool:
        """
        Extract the reassembled archive to target directory
        
        Args:
            archive_path: Path to the reassembled tar archive
            target_dir: Directory to extract to
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            print(f"[ssapy] Extracting data archive...")
            
            with tarfile.open(archive_path, "r:gz") as tar:
                # Safety check for malicious archives
                def is_safe_member(member):
                    member_path = Path(member.name)
                    return not (
                        member_path.is_absolute() or 
                        ".." in member_path.parts or
                        member.name.startswith("/")
                    )
                
                # Filter safe members
                safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                
                if not safe_members:
                    print("Warning: No safe members found in archive")
                    return False
                
                # Extract safe members
                tar.extractall(path=target_dir, members=safe_members)
                
            print(f"[ssapy] Data successfully extracted to {target_dir / 'data'}")
            return True
            
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return False
    
    def get_data_dir(self) -> Optional[str]:
        """
        Get the path to the extracted data directory
        Thread-safe and only extracts once per process.
        
        Returns:
            String path to the data directory, or None if extraction failed
        """
        # Quick check without lock for performance
        if self._extracted and self._data_dir:
            return str(self._data_dir)
        
        # Thread-safe check and extraction
        with self._lock:
            if self._extracted and self._data_dir:  # Double-check pattern
                return str(self._data_dir)
            
            # Find the chunk files
            chunk_files = self._find_chunk_files()
            if not chunk_files:
                print("Error: SSAPy data chunks not found")
                print("This may indicate an incomplete installation.")
                return None
            
            # Create temporary directory for reassembly and extraction
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="ssapy_data_")
            
            temp_path = Path(self._temp_dir)
            temp_archive = temp_path / "reassembled_data.tar.gz"
            
            # Reassemble chunks
            if not self._reassemble_chunks(chunk_files, temp_archive):
                return None
            
            # Extract reassembled archive
            if self._extract_archive(temp_archive, temp_path):
                self._data_dir = temp_path / "data"
                if self._data_dir.exists():
                    self._extracted = True
                    # Clean up the temporary archive to save space
                    temp_archive.unlink()
                    return str(self._data_dir)
                else:
                    print("Error: Expected 'data' directory not found after extraction")
                    return None
            else:
                return None


# Global instance for easy access
_data_loader = SSAPyDataLoader()


def ensure_data_downloaded():
    """
    Ensures data is extracted from chunked archives if needed.
    Thread-safe and only runs once per process.
    
    This function replaces the git clone functionality from data_utils.
    """
    data_dir = _data_loader.get_data_dir()
    if data_dir is None:
        raise RuntimeError(
            "SSAPy data files could not be loaded. "
            "This may indicate an incomplete installation. "
            "Please reinstall the package."
        )


def requires_data(func):
    """
    Decorator that ensures data is extracted before running a function.
    Use this on functions that need the data directory.
    
    Direct replacement for the data_utils decorator.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ensure_data_downloaded()
        return func(*args, **kwargs)
    return wrapper


def get_data_dir():
    """
    Get the data directory path, ensuring data is extracted first.
    
    Direct replacement for data_utils.get_data_dir().
    
    Returns:
        str: Path to the data directory
    """
    ensure_data_downloaded()
    return _data_loader.get_data_dir()


def is_data_available() -> bool:
    """
    Check if SSAPy data files are available
    
    Returns:
        True if data is available, False otherwise
    """
    return _data_loader.get_data_dir() is not None


def get_data_file(relative_path: str) -> Optional[Path]:
    """
    Get the full path to a specific data file
    
    Args:
        relative_path: Path relative to the data directory
        
    Returns:
        Full path to the file, or None if not found
    """
    data_dir = _data_loader.get_data_dir()
    if not data_dir:
        return None
        
    file_path = Path(data_dir) / relative_path
    if file_path.exists():
        return file_path
    else:
        return None


def list_data_files(pattern: str = "*") -> list:
    """
    List all data files matching a pattern
    
    Args:
        pattern: Glob pattern to match files
        
    Returns:
        List of Path objects for matching files
    """
    data_dir = _data_loader.get_data_dir()
    if not data_dir:
        return []
        
    return list(Path(data_dir).rglob(pattern))