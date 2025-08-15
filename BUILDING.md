# SSAPy Build Instructions for Developers

This document explains how to build and maintain the SSAPy package, particularly the custom chunked data handling system.

## Overview

SSAPy uses a **chunked tar archive approach** for handling large data files (307+ MB) to comply with PyPI's 100 MB per-file limit. This provides faster, more reliable installations for end users while staying within distribution limits.

## Build System Architecture

### Data File Handling
- **Source**: Large data files in `ssapy/data/` (307.4 MB, 22 files)
- **Build time**: Compressed and split into chunks (4 files: 3×83.8 MB + 1×2.8 MB)
- **Install time**: Chunks reassembled and extracted to temporary directory
- **Runtime**: Data accessed from extracted location

### Key Files
```
ssapy/
├── data/                         # Original large data files (excluded from git)
├── ssapy_data_chunk_000.tar.gz   # Chunk 1 (~80 MB, created during build)
├── ssapy_data_chunk_001.tar.gz   # Chunk 2 (~80 MB, created during build)
├── ssapy_data_chunk_002.tar.gz   # Chunk 3 (~80 MB, created during build)
├── data_loader.py                # Runtime data reassembly and access
└── __init__.py                   # Uses data_loader instead of data_utils

scripts/
├── ssapy_data_manager.py         # Chunked data archive management utility
└── build_ssapy.sh                # Automated build workflow

setup.py                          # Custom setuptools commands for chunked data handling
pyproject.toml                    # Modern packaging configuration  
MANIFEST.in                       # Controls what goes in source distributions
```

## Building SSAPy

### Prerequisites
- Python 3.8+
- CMake (for C++ extensions)
- Standard build tools

### Quick Build
```bash
# Create chunked data archives and build package
python scripts/ssapy_data_manager.py create
python -m build
```

### Automated Build (Recommended)
```bash
# Complete build and test workflow
./scripts/build_ssapy.sh full
```

### Individual Steps
```bash
# 1. Create chunked data archives
python scripts/ssapy_data_manager.py create

# 2. Verify chunks are under 100 MB  
python scripts/ssapy_data_manager.py verify

# 3. Build package
python -m build

# 4. Test installation
./scripts/build_ssapy.sh test
```

## Data Management Commands

### Chunked Archive Operations
```bash
# Create chunked archives from ssapy/data/
python scripts/ssapy_data_manager.py create

# Verify chunks are under PyPI 100MB limit
python scripts/ssapy_data_manager.py verify

# List chunk files (for setup.py package_data)
python scripts/ssapy_data_manager.py list

# Reassemble chunks and extract (for testing)
python scripts/ssapy_data_manager.py reassemble --target ./test_dir

# Clean up chunk files and build artifacts
python scripts/ssapy_data_manager.py clean
```

### Build Workflow Commands
```bash
# Individual operations
./scripts/build_ssapy.sh archive   # Create chunked archives
./scripts/build_ssapy.sh build     # Clean and build package
./scripts/build_ssapy.sh test      # Test package installation
./scripts/build_ssapy.sh clean     # Remove build artifacts
./scripts/build_ssapy.sh stats     # Show file size statistics

# Complete workflow
./scripts/build_ssapy.sh full      # Create chunks → Build → Test
```

## Data File Access (Runtime)

### For End Users
```python
import ssapy

# Lazy-loaded data directory (works like before)
data_path = ssapy.datadir
print(data_path)  # /tmp/ssapy_data_xyz/data

# Data is automatically reassembled and extracted on first access
```

### For Developers
```python
from ssapy.data_loader import get_data_dir, ensure_data_downloaded

# Direct data directory access (handles chunk reassembly)
data_dir = get_data_dir()

# Ensure data is available before use
ensure_data_downloaded()

# Decorator for functions requiring data
from ssapy.data_loader import requires_data

@requires_data
def my_function():
    # Data guaranteed to be available here
    pass
```

## PyPI Size Limit Compliance

### Chunking Strategy
- **Original data**: 307.4 MB (22 files)
- **Compressed**: 242.7 MB (21% reduction)
- **Chunked**: 4 files (3×83.8 MB + 1×2.8 MB)
- **PyPI limit**: 100 MB per file ✓

### Chunk File Naming
```
ssapy_data_chunk_000.tar.gz  # First 83.8 MB
ssapy_data_chunk_001.tar.gz  # Second 83.8 MB  
ssapy_data_chunk_002.tar.gz  # Third 83.8 MB
ssapy_data_chunk_003.tar.gz  # Final 2.8 MB
```

## Troubleshooting

### Common Issues

**"Chunks not found"**
```bash
# Solution: Create the chunked archives
python scripts/ssapy_data_manager.py create
```

**"Data directory does not exist"**
- Ensure `ssapy/data/` contains the 22 data files
- Check that files aren't excluded by `.gitignore`

**"Chunk exceeds 100 MB"**
- Reduce chunk size in `chunked_data_manager.py`
- Current default: 80 MB (safe margin under 100 MB limit)

**"CMake build fails"**
- Ensure CMake is installed and in PATH
- Check that C++ build tools are available

**"Package data not included in wheel"**
- Verify `MANIFEST.in` includes `ssapy/ssapy_data_chunk_*.tar.gz`
- Check that chunks were created before building

### Debug Commands

```bash
# Check what files are in the data directory
ls -la ssapy/data/

# Verify chunk files and sizes
python scripts/ssapy_data_manager.py verify

# List chunk files for setup.py
python scripts/ssapy_data_manager.py list

# Test data loading (reassembly + extraction)
python -c "from ssapy.data_loader import get_data_dir; print(get_data_dir())"

# Check what chunks are in your package
ls -lh ssapy/ssapy_data_chunk_*.tar.gz
```

## Release Process

### For New Releases

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Ensure data is current**:
   ```bash
   # If data files changed, recreate chunks
   python scripts/ssapy_data_manager.py clean
   python scripts/ssapy_data_manager.py create
   ```
3. **Verify PyPI compliance**:
   ```bash
   python scripts/ssapy_data_manager.py verify
   ```
4. **Build and test**:
   ```bash
   ./scripts/build_ssapy.sh full
   ```
5. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

### Data File Updates

If the data files in `ssapy/data/` are updated:

```bash
# 1. Remove old chunks
python scripts/ssapy_data_manager.py clean

# 2. Create new chunks with updated data
python scripts/ssapy_data_manager.py create

# 3. Verify sizes are still compliant
python scripts/ssapy_data_manager.py verify

# 4. Rebuild package
python -m build
```

### Adding New Large Data Files

If you add files that increase the total size significantly:

1. **Check if chunking is still adequate**:
   ```bash
   python scripts/ssapy_data_manager.py create
   python scripts/ssapy_data_manager.py verify
   ```

2. **If chunks exceed 100 MB**, reduce chunk size in `scripts/ssapy_data_manager.py`:
   ```python
   self.chunk_size = 70 * 1024 * 1024  # Reduce to 70 MB
   ```

## Migration Notes

This system replaced the previous git clone approach:

### Before (v1.1.0 and earlier)
- Data downloaded via git clone during first import
- Required git and network access
- Slower first-time usage
- Potential network/git failures

### After (v1.2.0+)
- Data included in package as chunked compressed archives
- No git or network dependency for users
- Fast, reliable installations
- Works offline
- **PyPI compliant**: All files under 100 MB limit

### Code Changes Made
- `data_utils.py` → `data_loader.py`
- Git clone → Chunked tar reassembly and extraction
- Same API preserved for backward compatibility
- Added chunking system for PyPI compliance

## File Size Reference

Current data statistics:
- **22 files** in `ssapy/data/` (includes .egm, .bsp, .cof, .png files)
- **307.4 MB** uncompressed
- **242.7 MB** compressed (21.1% reduction)
- **4 chunks** (3×83.8 MB + 1×2.8 MB, all under PyPI 100 MB limit)
- **Largest files**: de430.bsp (119.7 MB), gggrx_1200a_sha.tab (88.1 MB), egm2008.egm.cof (75.8 MB)

## Architecture Benefits

### For Developers
- **Automated chunking** handles PyPI size limits
- **Same build process** as before
- **Better compression** than git clone approach

### For End Users  
- **No git dependency** required
- **Faster installation** (no network downloads)
- **Offline capable** installations
- **Automatic data management** (transparent chunking)

## Contact

For questions about the build system, see the main repository issues or contact the SSAPy development team.