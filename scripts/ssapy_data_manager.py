#!/usr/bin/env python
"""
SSAPy Chunked Data Manager

Splits large tar archives into chunks that fit within PyPI's 100MB per-file limit.
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path


class ChunkedDataManager:
    """
    Manages SSAPy data files using chunked tar archives for PyPI compatibility
    """
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data_dir = self.base_dir / "ssapy" / "data"
        self.chunk_size = 80 * 1024 * 1024  # 80 MB chunks (under 100 MB limit)
        self.chunk_prefix = "ssapy_data_chunk_"
        
    def create_chunked_archive(self, compression_level=6):
        """
        Create tar archive and split into chunks under 100MB
        
        Args:
            compression_level (int): Compression level (1-9, default 6)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.data_dir.exists():
            print(f"Error: Data directory {self.data_dir} does not exist")
            return False
            
        print(f"Creating chunked data archive from: {self.data_dir}")
        
        # Step 1: Create temporary full tar archive
        temp_tar = self.base_dir / "ssapy" / "temp_ssapy_data.tar.gz"
        
        try:
            # Create full archive first
            print("Creating temporary full archive...")
            with tarfile.open(temp_tar, f"w:gz", compresslevel=compression_level) as tar:
                tar.add(self.data_dir, arcname="data")
                
            archive_size = temp_tar.stat().st_size / (1024 * 1024)
            print(f"Full archive size: {archive_size:.1f} MB")
            
            # Step 2: Split into chunks
            chunk_files = self._split_into_chunks(temp_tar)
            
            # Step 3: Clean up temporary file
            temp_tar.unlink()
            
            if chunk_files:
                print(f"Successfully created {len(chunk_files)} chunks:")
                for i, chunk_file in enumerate(chunk_files):
                    size_mb = chunk_file.stat().st_size / (1024 * 1024)
                    print(f"  {chunk_file.name}: {size_mb:.1f} MB")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error creating chunked archive: {e}")
            if temp_tar.exists():
                temp_tar.unlink()
            return False
    
    def _split_into_chunks(self, archive_path):
        """Split archive into chunks under the size limit"""
        chunk_files = []
        chunk_num = 0
        
        print(f"Splitting archive into {self.chunk_size / (1024*1024):.0f}MB chunks...")
        
        try:
            with open(archive_path, 'rb') as source:
                while True:
                    chunk_filename = f"{self.chunk_prefix}{chunk_num:03d}.tar.gz"
                    chunk_path = self.base_dir / "ssapy" / chunk_filename
                    
                    with open(chunk_path, 'wb') as chunk_file:
                        bytes_written = 0
                        while bytes_written < self.chunk_size:
                            remaining = self.chunk_size - bytes_written
                            data = source.read(min(8192, remaining))
                            if not data:
                                break
                            chunk_file.write(data)
                            bytes_written += len(data)
                    
                    if bytes_written == 0:
                        # No data written, remove empty file
                        chunk_path.unlink()
                        break
                    
                    chunk_files.append(chunk_path)
                    chunk_num += 1
                    
                    if bytes_written < self.chunk_size:
                        # Last chunk
                        break
                        
        except Exception as e:
            print(f"Error splitting archive: {e}")
            # Clean up any partial chunks
            for chunk_file in chunk_files:
                if chunk_file.exists():
                    chunk_file.unlink()
            return []
            
        return chunk_files
    
    def reassemble_chunks(self, target_dir=None):
        """
        Reassemble chunks back into full archive and extract
        
        Args:
            target_dir: Directory to extract to (defaults to ssapy/)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if target_dir is None:
            target_dir = self.base_dir / "ssapy"
        else:
            target_dir = Path(target_dir)
        
        # Find all chunk files
        chunk_pattern = f"{self.chunk_prefix}*.tar.gz"
        chunk_files = sorted(list((self.base_dir / "ssapy").glob(chunk_pattern)))
        
        if not chunk_files:
            print(f"Error: No chunk files found matching {chunk_pattern}")
            return False
            
        print(f"Found {len(chunk_files)} chunks to reassemble")
        
        # Reassemble into temporary full archive
        temp_archive = target_dir / "temp_reassembled.tar.gz"
        
        try:
            with open(temp_archive, 'wb') as output:
                for chunk_file in chunk_files:
                    print(f"Reading chunk: {chunk_file.name}")
                    with open(chunk_file, 'rb') as chunk:
                        while True:
                            data = chunk.read(8192)
                            if not data:
                                break
                            output.write(data)
            
            print("Chunks reassembled, extracting data...")
            
            # Extract the reassembled archive
            with tarfile.open(temp_archive, "r:gz") as tar:
                # Safety check for malicious archives
                def is_safe_member(member):
                    member_path = Path(member.name)
                    return not (
                        member_path.is_absolute() or 
                        ".." in member_path.parts or
                        member.name.startswith("/")
                    )
                
                safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                tar.extractall(path=target_dir, members=safe_members)
                
            print(f"Successfully extracted {len(safe_members)} items")
            
            # Clean up temporary archive
            temp_archive.unlink()
            return True
            
        except Exception as e:
            print(f"Error reassembling chunks: {e}")
            if temp_archive.exists():
                temp_archive.unlink()
            return False
    
    def verify_chunks(self):
        """Verify that all chunks exist and can be reassembled"""
        chunk_pattern = f"{self.chunk_prefix}*.tar.gz"
        chunk_files = sorted(list((self.base_dir / "ssapy").glob(chunk_pattern)))
        
        if not chunk_files:
            print(f"No chunk files found matching {chunk_pattern}")
            return False
            
        print(f"Found {len(chunk_files)} chunks:")
        total_size = 0
        
        for chunk_file in chunk_files:
            size_mb = chunk_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            status = "✓" if size_mb < 100 else "✗ TOO LARGE"
            print(f"  {chunk_file.name}: {size_mb:.1f} MB {status}")
            
        print(f"Total reassembled size: {total_size:.1f} MB")
        
        # Verify all chunks are under 100 MB
        oversized = [f for f in chunk_files if f.stat().st_size > 100 * 1024 * 1024]
        if oversized:
            print(f"ERROR: {len(oversized)} chunks exceed 100 MB limit")
            return False
            
        return True
    
    def clean_chunks(self):
        """Remove all chunk files"""
        chunk_pattern = f"{self.chunk_prefix}*.tar.gz"
        chunk_files = list((self.base_dir / "ssapy").glob(chunk_pattern))
        
        for chunk_file in chunk_files:
            chunk_file.unlink()
            print(f"Removed chunk: {chunk_file.name}")
            
        print(f"Cleaned up {len(chunk_files)} chunk files")
    
    def get_chunk_files(self):
        """Get list of chunk files for package_data"""
        chunk_pattern = f"{self.chunk_prefix}*.tar.gz"
        chunk_files = sorted(list((self.base_dir / "ssapy").glob(chunk_pattern)))
        return [f.name for f in chunk_files]


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="SSAPy Chunked Data Manager")
    parser.add_argument("--base-dir", help="Base directory (default: current directory)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create chunks
    create_parser = subparsers.add_parser("create", help="Create chunked archive")
    create_parser.add_argument("--compression", type=int, default=6,
                              help="Compression level (1-9, default: 6)")
    
    # Verify chunks
    subparsers.add_parser("verify", help="Verify chunks are under 100MB")
    
    # Reassemble chunks
    reassemble_parser = subparsers.add_parser("reassemble", help="Reassemble chunks and extract")
    reassemble_parser.add_argument("--target", help="Target directory")
    
    # Clean chunks
    subparsers.add_parser("clean", help="Remove chunk files")
    
    # List chunks
    subparsers.add_parser("list", help="List chunk files for package_data")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    manager = ChunkedDataManager(args.base_dir)
    
    if args.command == "create":
        success = manager.create_chunked_archive(args.compression)
        return 0 if success else 1
        
    elif args.command == "verify":
        success = manager.verify_chunks()
        return 0 if success else 1
        
    elif args.command == "reassemble":
        success = manager.reassemble_chunks(args.target)
        return 0 if success else 1
        
    elif args.command == "clean":
        manager.clean_chunks()
        return 0
        
    elif args.command == "list":
        chunks = manager.get_chunk_files()
        print("Chunk files for package_data:")
        for chunk in chunks:
            print(f'    "{chunk}",')
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())