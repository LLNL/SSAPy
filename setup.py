from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build import build
from setuptools.command.sdist import sdist
import subprocess
import sys
import os
import tarfile
import shutil
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read dependencies from pyproject.toml
def get_dependencies():
    try:
        with open('pyproject.toml', 'rb') as f:
            pyproject = tomllib.load(f)
        return pyproject['project']['dependencies']
    except (FileNotFoundError, KeyError):
        return []


class DataManager:
    """Utility class for managing SSAPy chunked data files via tar archives"""
    
    DATA_DIR = "ssapy/data"
    CHUNK_PREFIX = "ssapy_data_chunk_"
    CHUNK_SIZE = 75 * 1024 * 1024  # 75 MB chunks (safer margin)
    
    @classmethod
    def create_chunked_archive(cls):
        """Create chunked tar archives of data directory"""
        if not os.path.exists(cls.DATA_DIR):
            print(f"Warning: {cls.DATA_DIR} directory not found, skipping archive creation")
            return False
            
        print(f"Creating chunked data archives from: {cls.DATA_DIR}")
        
        # Use the external chunked data manager script
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "scripts/ssapy_data_manager.py", "create"
            ], check=True, capture_output=True, text=True)
            print("Chunked archives created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunked archives: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    @classmethod
    def reassemble_and_extract(cls, target_dir=None):
        """Reassemble chunks and extract data"""
        if target_dir is None:
            target_dir = os.path.dirname(cls.DATA_DIR)
            
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "scripts/ssapy_data_manager.py", 
                "reassemble", "--target", target_dir
            ], check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error reassembling chunks: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    @classmethod
    def get_chunk_filenames(cls):
        """Get list of chunk filenames for package_data"""
        chunk_pattern = f"{cls.CHUNK_PREFIX}*.tar.gz"
        ssapy_dir = Path("ssapy")
        if ssapy_dir.exists():
            chunk_files = sorted([f.name for f in ssapy_dir.glob(chunk_pattern)])
            return chunk_files
        return []


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if 'CMAKE_VERBOSE_MAKEFILE' in env:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=1']
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


class CustomBuild(build):
    """Custom build command that creates chunked data archive before building"""
    
    def run(self):
        # Create the chunked data archive before building
        print("=== SSAPy Custom Build Process (Chunked) ===")
        DataManager.create_chunked_archive()
        
        # Run the standard build process
        build.run(self)


class CustomSdist(sdist):
    """Custom sdist command that ensures chunked data archive is included"""
    
    def run(self):
        # Ensure chunked data archive exists before creating source distribution
        print("=== Creating SSAPy Source Distribution (Chunked) ===")
        DataManager.create_chunked_archive()
        
        # Run the standard sdist process
        sdist.run(self)


class CustomInstall(install):
    """Custom install command that reassembles chunks and extracts data files"""
    
    def run(self):
        # Run the standard installation first (including CMake build)
        print("=== SSAPy Custom Install Process (Chunked) ===")
        install.run(self)
        
        # Then reassemble chunks and extract data files
        self._reassemble_and_extract_data()
    
    def _reassemble_and_extract_data(self):
        """Reassemble chunks and extract data files to the installed package location"""
        # Find where the package was installed
        install_dir = None
        for path in sys.path:
            potential_path = os.path.join(path, "ssapy")
            if os.path.exists(potential_path):
                install_dir = potential_path
                break
        
        if install_dir is None:
            print("Warning: Could not locate installed ssapy package for data extraction")
            return
            
        print(f"Reassembling data chunks in: {install_dir}")
        
        # Find chunk files in the installed package
        chunk_pattern = f"{DataManager.CHUNK_PREFIX}*.tar.gz"
        chunk_files = sorted(list(Path(install_dir).glob(chunk_pattern)))
        
        if not chunk_files:
            print("Warning: No chunk files found in installed package")
            return
            
        print(f"Found {len(chunk_files)} chunks to reassemble")
        
        # Reassemble chunks into temporary archive
        temp_archive = os.path.join(install_dir, "temp_reassembled.tar.gz")
        
        try:
            with open(temp_archive, 'wb') as output:
                for chunk_file in chunk_files:
                    with open(chunk_file, 'rb') as chunk:
                        while True:
                            data = chunk.read(8192)
                            if not data:
                                break
                            output.write(data)
            
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
                tar.extractall(path=install_dir, members=safe_members)
                
            print(f"Successfully reassembled and extracted {len(safe_members)} data files")
            
            # Clean up temporary archive
            os.remove(temp_archive)
            
            # Optionally remove chunk files to save space after extraction
            for chunk_file in chunk_files:
                os.remove(chunk_file)
                
        except Exception as e:
            print(f"Warning: Failed to reassemble and extract data files: {e}")


class CustomDevelop(develop):
    """Custom develop command for editable installs"""
    
    def run(self):
        print("=== SSAPy Development Install (Chunked) ===")
        
        # For development installs, try to use existing data or reassemble from chunks
        if os.path.exists(DataManager.DATA_DIR):
            print("Using existing data directory for development install")
        else:
            # Try to reassemble from chunks
            chunk_files = DataManager.get_chunk_filenames()
            if chunk_files:
                print("Reassembling chunks for development install")
                DataManager.reassemble_and_extract()
            else:
                print(f"Warning: Neither {DataManager.DATA_DIR} nor chunk files found")
                print("You may need to create the chunked archives or obtain the data files")
        
        # Run the standard develop process
        develop.run(self)


# Get package data including dynamic chunk files
def get_package_data():
    """Get package data including chunk files"""
    # Always include the CMake shared library
    package_files = ['_ssapy*.so']
    
    # Look for existing chunk files
    chunk_pattern = f"ssapy_data_chunk_*.tar.gz"
    ssapy_dir = Path("ssapy")
    if ssapy_dir.exists():
        chunk_files = sorted([f.name for f in ssapy_dir.glob(chunk_pattern)])
        package_files.extend(chunk_files)
        print(f"Including {len(chunk_files)} chunk files in package")
    else:
        print("Warning: No chunk files found - they may be created during build")
        # Include expected chunk files even if they don't exist yet
        # The build process will create them
        expected_chunks = [
            "ssapy_data_chunk_000.tar.gz",
            "ssapy_data_chunk_001.tar.gz", 
            "ssapy_data_chunk_002.tar.gz",
            "ssapy_data_chunk_003.tar.gz",  # Added 4th chunk
        ]
        package_files.extend(expected_chunks)
        print(f"Including {len(expected_chunks)} expected chunk files")
    
    return {
        'ssapy': package_files
    }


setup(
    name='ssapy',
    version='1.2.0',  # Bump version for chunked approach
    
    # CMake extension (preserved from original)
    ext_modules=[CMakeExtension("ssapy._ssapy")],
    
    # Combined command classes (CMake + Chunked Data handling)
    cmdclass={
        "build_ext": CMakeBuild,        # Original CMake build
        "build": CustomBuild,           # New: Creates chunked archives
        "sdist": CustomSdist,           # New: Includes chunks in source dist
        "install": CustomInstall,       # New: Reassembles chunks on install
        "develop": CustomDevelop,       # New: Handles editable installs
    },
    
    # Package configuration
    packages=find_packages(),
    
    # Package data - include both CMake artifacts and chunk files
    package_data=get_package_data(),    # Dynamic chunk inclusion
    
    # Metadata (preserved from original)
    license='MIT',
    zip_safe=False,
    include_package_data=True,
    install_requires=get_dependencies(),
)