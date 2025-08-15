#!/bin/bash
# build_ssapy.sh - Comprehensive build script for SSAPy with tar-based data handling

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/ssapy/data"
ARCHIVE_PATH="$PROJECT_ROOT/ssapy/ssapy_data.tar.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if data directory exists
check_data_dir() {
    if [ ! -d "$DATA_DIR" ]; then
        log_error "Data directory $DATA_DIR does not exist"
        log_info "Please ensure your data files are in the ssapy/data directory"
        return 1
    fi
    
    local file_count=$(find "$DATA_DIR" -type f | wc -l)
    local dir_size=$(du -sh "$DATA_DIR" | cut -f1)
    log_info "Found $file_count files in data directory ($dir_size)"
    
    # Show some large files for verification
    log_info "Large files (>10MB):"
    find "$DATA_DIR" -type f -size +10M -exec ls -lh {} \; | head -5 | awk '{print "  " $9 " (" $5 ")"}'
    
    return 0
}

# Function to create data archive
create_archive() {
    log_info "Creating data archive..."
    
    if [ ! -d "$DATA_DIR" ]; then
        log_error "Data directory $DATA_DIR not found"
        return 1
    fi
    
    # Remove existing archive
    if [ -f "$ARCHIVE_PATH" ]; then
        rm "$ARCHIVE_PATH"
        log_info "Removed existing archive"
    fi
    
    # Create archive using the Python data manager
    cd "$PROJECT_ROOT"
    python3 scripts/ssapy_data_manager.py archive
    
    if [ $? -eq 0 ]; then
        log_success "Data archive created successfully"
        return 0
    else
        log_error "Failed to create data archive"
        return 1
    fi
}

# Function to verify archive
verify_archive() {
    log_info "Verifying data archive..."
    
    if [ ! -f "$ARCHIVE_PATH" ]; then
        log_error "Archive $ARCHIVE_PATH not found"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    python3 scripts/ssapy_data_manager.py verify
    
    if [ $? -eq 0 ]; then
        log_success "Archive verification passed"
        return 0
    else
        log_error "Archive verification failed"
        return 1
    fi
}

# Function to extract archive (for testing)
extract_archive() {
    local target_dir="${1:-./test_extract}"
    
    log_info "Extracting archive to $target_dir"
    
    if [ ! -f "$ARCHIVE_PATH" ]; then
        log_error "Archive $ARCHIVE_PATH not found"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    python3 scripts/ssapy_data_manager.py extract --target "$target_dir" --force
    
    if [ $? -eq 0 ]; then
        log_success "Archive extracted successfully"
        return 0
    else
        log_error "Archive extraction failed"
        return 1
    fi
}

# Function to clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Use the Python data manager for cleaning
    python3 scripts/ssapy_data_manager.py clean
    
    # Additional cleanup for CMake artifacts
    for dir in _skbuild; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            log_info "Removed $dir"
        fi
    done
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    log_success "Build artifacts cleaned"
}

# Function to build package
build_package() {
    log_info "Building SSAPy package with CMake and data archive..."
    
    cd "$PROJECT_ROOT"
    
    # Ensure we have the latest build tools
    python3 -m pip install --upgrade build setuptools wheel
    
    # Create archive first
    if ! create_archive; then
        log_error "Failed to create data archive"
        return 1
    fi
    
    # Build the package (this will trigger CMake build and data archive inclusion)
    log_info "Building source distribution and wheel..."
    python3 -m build
    
    if [ $? -eq 0 ]; then
        log_success "Package built successfully"
        
        # Show what was built
        if [ -d "dist" ]; then
            log_info "Built packages:"
            ls -lh dist/
        fi
        
        return 0
    else
        log_error "Package build failed"
        return 1
    fi
}

# Function to test installation
test_install() {
    log_info "Testing package installation..."
    
    # Create a temporary virtual environment
    local test_env="test_ssapy_env"
    
    if [ -d "$test_env" ]; then
        rm -rf "$test_env"
    fi
    
    python3 -m venv "$test_env"
    source "$test_env/bin/activate"
    
    # Install build dependencies (needed for CMake)
    pip install cmake scikit-build
    
    # Install from local build
    pip install dist/*.whl
    
    # Test that the package can be imported and data is available
    python3 -c "
import ssapy
import os
from pathlib import Path

print('Testing SSAPy installation...')

# Check if the package imports correctly
try:
    import ssapy
    print('✓ SSAPy package imports successfully')
except ImportError as e:
    print(f'✗ Failed to import SSAPy: {e}')
    exit(1)

# Check if data loader works
try:
    from ssapy.data_loader import is_data_available, get_data_dir, list_data_files
    
    if is_data_available():
        data_dir = get_data_dir()
        files = list_data_files('*.egm')  # Look for gravity model files
        print(f'✓ Data directory found: {data_dir}')
        print(f'✓ Found {len(files)} .egm files')
        
        # Check for specific files we know should exist
        egm_files = list_data_files('egm*.egm')
        bsp_files = list_data_files('*.bsp')
        cof_files = list_data_files('*.cof')
        
        print(f'✓ EGM files: {len(egm_files)}')
        print(f'✓ BSP files: {len(bsp_files)}')
        print(f'✓ COF files: {len(cof_files)}')
        
    else:
        print('✗ SSAPy data files not available')
        exit(1)
        
except Exception as e:
    print(f'✗ Data loader test failed: {e}')
    exit(1)

print('✓ All installation tests passed')
"
    
    local test_result=$?
    deactivate
    rm -rf "$test_env"
    
    if [ $test_result -eq 0 ]; then
        log_success "Installation test passed"
        return 0
    else
        log_error "Installation test failed"
        return 1
    fi
}

# Function to show archive statistics
show_stats() {
    log_info "SSAPy Data Archive Statistics"
    
    if [ -f "$ARCHIVE_PATH" ]; then
        local archive_size=$(ls -lh "$ARCHIVE_PATH" | awk '{print $5}')
        log_info "Archive size: $archive_size"
    fi
    
    if [ -d "$DATA_DIR" ]; then
        local original_size=$(du -sh "$DATA_DIR" | cut -f1)
        local file_count=$(find "$DATA_DIR" -type f | wc -l)
        log_info "Original data size: $original_size ($file_count files)"
        
        # Show largest files
        log_info "Largest files:"
        find "$DATA_DIR" -type f -exec ls -lh {} \; | sort -k5 -hr | head -5 | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    if [ -f "$ARCHIVE_PATH" ] && [ -d "$DATA_DIR" ]; then
        # Calculate compression ratio
        python3 -c "
import os
archive_size = os.path.getsize('$ARCHIVE_PATH')
original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk('$DATA_DIR')
                   for filename in filenames)
ratio = (1 - archive_size / original_size) * 100
print(f'Compression ratio: {ratio:.1f}% reduction')
"
    fi
}

# Main script logic
case "${1:-}" in
    "archive")
        check_data_dir && create_archive
        ;;
    "verify")
        verify_archive
        ;;
    "extract")
        extract_archive "$2"
        ;;
    "clean")
        clean_build
        ;;
    "build")
        clean_build && build_package
        ;;
    "test")
        test_install
        ;;
    "stats")
        show_stats
        ;;
    "full")
        log_info "Running full build and test cycle..."
        check_data_dir && \
        clean_build && \
        create_archive && \
        verify_archive && \
        show_stats && \
        build_package && \
        test_install
        ;;
    *)
        echo "Usage: $0 {archive|verify|extract|clean|build|test|stats|full}"
        echo ""
        echo "Commands:"
        echo "  archive  - Create compressed tar archive of data directory"
        echo "  verify   - Verify integrity of existing archive"
        echo "  extract  - Extract archive (for testing)"
        echo "  clean    - Remove build artifacts"
        echo "  build    - Clean and build package with data archive"
        echo "  test     - Test package installation"
        echo "  stats    - Show archive and data statistics"
        echo "  full     - Run complete build and test cycle"
        echo ""
        echo "Examples:"
        echo "  $0 archive           # Create data archive"
        echo "  $0 full              # Complete build and test"
        echo "  $0 stats             # Show compression statistics"
        exit 1
        ;;
esac