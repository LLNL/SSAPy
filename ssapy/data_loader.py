"""
SSAPy Runtime Data Loader

This module handles loading data files from the tar archive at runtime.
It replaces the git clone approach with efficient tar extraction.
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
    Handles loading and managing SSAPy data files from tar archive
    """
    
    _instance = None
    _data_dir = None
    _temp_dir = None
    _extracted = False
    _lock = threading.Lock()
    
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
    
    def _find_archive(self) -> Optional[Path]:
        """
        Find the data archive in the package installation
        
        Returns:
            Path to the archive file, or None if not found
        """
        # Try to find the archive relative to this module
        try:
            import ssapy
            package_dir = Path(ssapy.__file__).parent
            archive_path = package_dir / "ssapy_data.tar.gz"
            
            if archive_path.exists():
                return archive_path
                
        except ImportError:
            pass
        
        # Fallback: look in common locations
        search_paths = [
            Path.cwd() / "ssapy" / "ssapy_data.tar.gz",
            Path(__file__).parent / "ssapy_data.tar.gz",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        return None
    
    def _extract_archive(self, archive_path: Path, target_dir: Path) -> bool:
        """
        Extract the data archive to target directory
        
        Args:
            archive_path: Path to the tar archive
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
            
            # Find the archive
            archive_path = self._find_archive()
            if not archive_path:
                print("Error: SSAPy data archive not found")
                print("This may indicate an incomplete installation.")
                return None
            
            # Create temporary directory for extraction
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="ssapy_data_")
            
            temp_path = Path(self._temp_dir)
            
            # Extract archive
            if self._extract_archive(archive_path, temp_path):
                self._data_dir = temp_path / "data"
                if self._data_dir.exists():
                    self._extracted = True
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
    Ensures data is extracted from archive if needed.
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


if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="SSAPy Data Loader")
    parser.add_argument("--test", action="store_true", help="Test data loading")
    parser.add_argument("--info", action="store_true", help="Show data directory info")
    
    args = parser.parse_args()
    
    if args.test:
        try:
            ensure_data_downloaded()
            print("✓ Data loading test successful")
        except Exception as e:
            print(f"✗ Data loading test failed: {e}")
            sys.exit(1)
    
    elif args.info:
        data_dir = get_data_dir()
        if data_dir:
            data_path = Path(data_dir)
            files = list(data_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
            print(f"Data directory: {data_dir}")
            print(f"Files: {file_count}")
            print(f"Total size: {total_size:.1f} MB")
        else:
            print("Data directory not available")
    
    else:
        # Default: just test that data is available
        try:
            data_dir = get_data_dir()
            print(f"✓ SSAPy data available at: {data_dir}")
        except Exception as e:
            print(f"✗ SSAPy data not available: {e}")
            sys.exit(1)