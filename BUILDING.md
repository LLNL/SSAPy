# SSAPy Build Instructions for Developers

This document explains how to build and maintain the SSAPy package, particularly the custom data handling system.

## Overview

SSAPy uses a **tar archive approach** for handling large data files (307+ MB) instead of git clones during installation. This provides faster, more reliable installations for end users.

## Build System Architecture

### Data File Handling
- **Source**: Large data files in `ssapy/data/` (307.4 MB, 22 files)
- **Build time**: Compressed into `ssapy/ssapy_data.tar.gz` (242.7 MB, 21% reduction)
- **Install time**: Archive extracted to temporary directory
- **Runtime**: Data accessed from extracted location

### Key Files
```
ssapy/
├── data/                    # Original large data files (excluded from git)
├── ssapy_data.tar.gz       # Compressed archive (created during build)
├── data_loader.py          # Runtime data extraction and access
└── __init__.py             # Uses data_loader instead of data_utils

scripts/
├── ssapy_data_manager.py   # Data archive management utility
└── build_ssapy.sh          # Automated build workflow

setup.py                    # Custom setuptools commands for data handling
pyproject.toml              # Modern packaging configuration  
MANIFEST.in                 # Controls what goes in source distributions
```

## Building SSAPy

### Prerequisites
- Python 3.8+
- CMake (for C++ extensions)
- Standard build tools

### Quick Build
```bash
# Create data archive and build package
python scripts/ssapy_data_manager.py archive
python -m build
```

### Automated Build (Recommended)
```bash
# Complete build and test workflow
./scripts/build_ssapy.sh full
```

### Individual Steps
```bash
# 1. Create data archive
python scripts/ssapy_data_manager.py archive

# 2. Verify archive integrity  
python scripts/ssapy_data_manager.py verify

# 3. Build package
python -m build

# 4. Test installation
./scripts/build_ssapy.sh test
```

## Data Management Commands

### Archive Operations
```bash
# Create compressed archive from ssapy/data/
python scripts/ssapy_data_manager.py archive

# Verify archive integrity
python scripts/ssapy_data_manager.py verify

# Show compression statistics
python scripts/ssapy_data_manager.py stats

# Extract archive (for testing)
python scripts/ssapy_data_manager.py extract --target ./test_dir

# Clean up build artifacts
python scripts/ssapy_data_manager.py clean
```

### Build Workflow Commands
```bash
# Individual operations
./scripts/build_ssapy.sh archive   # Create data archive
./scripts/build_ssapy.sh build     # Clean and build package
./scripts/build_ssapy.sh test      # Test package installation
./scripts/build_ssapy.sh clean     # Remove build artifacts
./scripts/build_ssapy.sh stats     # Show file size statistics

# Complete workflow
./scripts/build_ssapy.sh full      # Archive → Build → Test
```

## Data File Access (Runtime)

### For End Users
```python
import ssapy

# Lazy-loaded data directory (works like before)
data_path = ssapy.datadir
print(data_path)  # /tmp/ssapy_data_xyz/data

# Data is extracted automatically on first access
```

### For Developers
```python
from ssapy.data_loader import get_data_dir, ensure_data_downloaded

# Direct data directory access
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

## Troubleshooting

### Common Issues

**"Data archive not found"**
```bash
# Solution: Create the archive
python scripts/ssapy_data_manager.py archive
```

**"Data directory does not exist"**
- Ensure `ssapy/data/` contains the 22 data files
- Check that files aren't excluded by `.gitignore`

**"CMake build fails"**
- Ensure CMake is installed and in PATH
- Check that C++ build tools are available

**"Package data not included in wheel"**
- Verify `MANIFEST.in` includes `ssapy/ssapy_data.tar.gz`
- Check that archive was created before building

### Debug Commands

```bash
# Check what files are in the data directory
ls -la ssapy/data/

# Verify archive contents
python scripts/ssapy_data_manager.py verify

# Test data loading
python -c "from ssapy.data_loader import get_data_dir; print(get_data_dir())"

# Check package contents
python -c "import tarfile; tar=tarfile.open('dist/*.tar.gz'); tar.list()"
```

## Release Process

### For New Releases

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Ensure data is current**:
   ```bash
   # If data files changed, recreate archive
   python scripts/ssapy_data_manager.py clean
   python scripts/ssapy_data_manager.py archive
   ```
3. **Build and test**:
   ```bash
   ./scripts/build_ssapy.sh full
   ```
4. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

### Data File Updates

If the data files in `ssapy/data/` are updated:

```bash
# 1. Remove old archive
python scripts/ssapy_data_manager.py clean

# 2. Create new archive with updated data
python scripts/ssapy_data_manager.py archive

# 3. Rebuild package
python -m build
```

## Migration Notes

This system replaced the previous git clone approach:

### Before (v1.1.0 and earlier)
- Data downloaded via git clone during first import
- Required git and network access
- Slower first-time usage
- Potential network/git failures

### After (v1.2+)
- Data included in package as compressed archive
- No git or network dependency for users
- Fast, reliable installations
- Works offline

### Code Changes Made
- `data_utils.py` → `data_loader.py`
- Git clone → Tar extraction
- Same API preserved for backward compatibility

## File Size Reference

Current data statistics:
- **22 files** in `ssapy/data/`
- **307.4 MB** uncompressed
- **242.7 MB** compressed archive (21.1% reduction)
- **File types**: `.egm`, `.bsp`, `.cof`, `.png`, `.zip`, etc.

## Contact

For questions about the build system, see the main repository issues or contact the SSAPy development team.