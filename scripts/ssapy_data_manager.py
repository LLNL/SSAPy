#!/usr/bin/env python
"""
SSAPy Build and Data Management Scripts

This module provides utilities for managing large data files in the SSAPy package
using tar archives instead of git clones.
"""

import os
import sys
import shutil
import tarfile
import argparse
from pathlib import Path


class SSAPyDataManager:
    """
    Manages SSAPy data files using tar archives
    """
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.data_dir = self.base_dir / "ssapy" / "data"
        self.archive_path = self.base_dir / "ssapy" / "ssapy_data.tar.gz"
        
    def create_archive(self, compression_level=6):
        """
        Create a compressed tar archive of the data directory
        
        Args:
            compression_level (int): Compression level (1-9, default 6)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.data_dir.exists():
            print(f"Error: Data directory {self.data_dir} does not exist")
            return False
            
        print(f"Creating data archive: {self.archive_path}")
        print(f"Source directory: {self.data_dir}")
        
        try:
            # Count files for progress indication
            file_count = sum(1 for _ in self.data_dir.rglob("*") if _.is_file())
            print(f"Archiving {file_count} files...")
            
            # Create the archive with specified compression
            with tarfile.open(self.archive_path, f"w:gz", compresslevel=compression_level) as tar:
                # Add files individually for better control
                for file_path in self.data_dir.rglob("*"):
                    if file_path.is_file():
                        # Create relative archive path (remove ssapy/data prefix)
                        arcname = file_path.relative_to(self.data_dir)
                        tar.add(file_path, arcname=f"data/{arcname}")
                        
            archive_size = self.archive_path.stat().st_size / (1024 * 1024)
            print(f"Archive created successfully: {archive_size:.1f} MB")
            return True
            
        except Exception as e:
            print(f"Error creating archive: {e}")
            return False
    
    def extract_archive(self, target_dir=None, force=False):
        """
        Extract the data archive
        
        Args:
            target_dir (str): Target directory (defaults to ssapy directory)
            force (bool): Force extraction even if data directory exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.archive_path.exists():
            print(f"Error: Archive {self.archive_path} does not exist")
            return False
            
        if target_dir is None:
            target_dir = self.base_dir / "ssapy"
        else:
            target_dir = Path(target_dir)
            
        data_target = target_dir / "data"
        
        if data_target.exists() and not force:
            print(f"Data directory {data_target} already exists (use --force to overwrite)")
            return True
            
        print(f"Extracting data archive to: {target_dir}")
        
        try:
            with tarfile.open(self.archive_path, "r:gz") as tar:
                # Extract with safety checks
                def is_safe_path(member_path):
                    return not (member_path.is_absolute() or ".." in member_path.parts)
                
                members = tar.getmembers()
                safe_members = []
                
                for member in members:
                    member_path = Path(member.name)
                    if is_safe_path(member_path):
                        safe_members.append(member)
                    else:
                        print(f"Skipping unsafe path: {member.name}")
                
                tar.extractall(path=target_dir, members=safe_members)
                
            print(f"Successfully extracted {len(safe_members)} files")
            return True
            
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return False
    
    def verify_archive(self):
        """
        Verify the integrity of the data archive
        
        Returns:
            bool: True if archive is valid, False otherwise
        """
        if not self.archive_path.exists():
            print(f"Archive {self.archive_path} does not exist")
            return False
            
        try:
            with tarfile.open(self.archive_path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"Archive contains {len(members)} entries")
                
                # Check for basic structure
                has_data_dir = any(m.name.startswith("data/") for m in members)
                if not has_data_dir:
                    print("Warning: Archive does not contain expected 'data/' directory")
                    return False
                    
                # Calculate total uncompressed size
                total_size = sum(m.size for m in members if m.isfile()) / (1024 * 1024)
                print(f"Total uncompressed size: {total_size:.1f} MB")
                
                # Show file type breakdown
                file_types = {}
                for member in members:
                    if member.isfile():
                        ext = Path(member.name).suffix.lower()
                        if not ext:
                            ext = "no_extension"
                        file_types[ext] = file_types.get(ext, 0) + 1
                
                print("File type breakdown:")
                for ext, count in sorted(file_types.items()):
                    print(f"  {ext}: {count} files")
                    
                # List some large files
                large_files = [m for m in members if m.isfile() and m.size > 10*1024*1024]
                if large_files:
                    print("Large files (>10MB):")
                    for member in sorted(large_files, key=lambda x: x.size, reverse=True)[:5]:
                        size_mb = member.size / (1024 * 1024)
                        print(f"  {member.name} ({size_mb:.1f} MB)")
                
                return True
                
        except Exception as e:
            print(f"Error verifying archive: {e}")
            return False
    
    def clean(self):
        """Remove generated archive and temporary files"""
        if self.archive_path.exists():
            self.archive_path.unlink()
            print(f"Removed archive: {self.archive_path}")
        
        # Clean up any build artifacts
        for pattern in ["build", "dist", "*.egg-info", "_skbuild"]:
            for path in self.base_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                elif path.is_file():
                    path.unlink()
                    print(f"Removed file: {path}")

    def cleanup_data_directory(self):
        """Remove the original data directory after archiving"""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            print(f"Cleaned up original {self.data_dir} directory")
    
    def get_statistics(self):
        """Get statistics about data directory and archive"""
        stats = {}
        
        # Original data directory stats
        if self.data_dir.exists():
            files = list(self.data_dir.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            stats['original'] = {
                'file_count': file_count,
                'total_size_mb': total_size / (1024 * 1024),
                'exists': True
            }
        else:
            stats['original'] = {'exists': False}
        
        # Archive stats
        if self.archive_path.exists():
            archive_size = self.archive_path.stat().st_size
            stats['archive'] = {
                'size_mb': archive_size / (1024 * 1024),
                'exists': True
            }
            
            # Calculate compression ratio if both exist
            if stats['original']['exists']:
                compression_ratio = (1 - archive_size / (stats['original']['total_size_mb'] * 1024 * 1024)) * 100
                stats['compression_ratio'] = compression_ratio
        else:
            stats['archive'] = {'exists': False}
            
        return stats


def main():
    """Command-line interface for data management"""
    parser = argparse.ArgumentParser(description="SSAPy Data Management Utility")
    parser.add_argument("--base-dir", help="Base directory (default: current directory)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Create data archive")
    archive_parser.add_argument("--compression", type=int, default=6, 
                               help="Compression level (1-9, default: 6)")
    archive_parser.add_argument("--cleanup", action="store_true",
                               help="Remove original data directory after archiving")
    
    # Extract command  
    extract_parser = subparsers.add_parser("extract", help="Extract data archive")
    extract_parser.add_argument("--target", help="Target directory")
    extract_parser.add_argument("--force", action="store_true",
                               help="Force extraction even if data exists")
    
    # Verify command
    subparsers.add_parser("verify", help="Verify archive integrity")
    
    # Clean command
    subparsers.add_parser("clean", help="Clean up generated files")
    
    # Stats command
    subparsers.add_parser("stats", help="Show statistics about data and archive")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    manager = SSAPyDataManager(args.base_dir)
    
    if args.command == "archive":
        success = manager.create_archive(args.compression)
        if success and args.cleanup:
            manager.cleanup_data_directory()
        return 0 if success else 1
        
    elif args.command == "extract":
        success = manager.extract_archive(args.target, args.force)
        return 0 if success else 1
        
    elif args.command == "verify":
        success = manager.verify_archive()
        return 0 if success else 1
        
    elif args.command == "clean":
        manager.clean()
        return 0
        
    elif args.command == "stats":
        stats = manager.get_statistics()
        
        print("=== SSAPy Data Statistics ===")
        
        if stats['original']['exists']:
            orig = stats['original']
            print(f"Original data directory:")
            print(f"  Files: {orig['file_count']}")
            print(f"  Size: {orig['total_size_mb']:.1f} MB")
        else:
            print("Original data directory: Not found")
            
        if stats['archive']['exists']:
            arch = stats['archive']
            print(f"Data archive:")
            print(f"  Size: {arch['size_mb']:.1f} MB")
            
            if 'compression_ratio' in stats:
                print(f"  Compression: {stats['compression_ratio']:.1f}% reduction")
        else:
            print("Data archive: Not found")
            
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())