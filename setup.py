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
    """Utility class for managing SSAPy data files via tar archives"""
    
    DATA_DIR = "ssapy/data"
    DATA_TAR = "ssapy/ssapy_data.tar.gz"
    
    @classmethod
    def create_data_archive(cls):
        """Create compressed tar archive of data directory"""
        if not os.path.exists(cls.DATA_DIR):
            print(f"Warning: {cls.DATA_DIR} directory not found, skipping archive creation")
            return False
            
        print(f"Creating data archive: {cls.DATA_TAR}")
        try:
            with tarfile.open(cls.DATA_TAR, "w:gz", compresslevel=6) as tar:
                # Add the entire data directory, but strip the path prefix
                tar.add(cls.DATA_DIR, arcname="data")
            
            archive_size = os.path.getsize(cls.DATA_TAR) / (1024 * 1024)
            print(f"Successfully created {cls.DATA_TAR} ({archive_size:.1f} MB)")
            return True
        except Exception as e:
            print(f"Error creating data archive: {e}")
            return False
    
    @classmethod
    def extract_data_archive(cls, target_dir=None):
        """Extract data archive to the appropriate location"""
        if target_dir is None:
            target_dir = os.path.dirname(cls.DATA_DIR)
            
        if not os.path.exists(cls.DATA_TAR):
            print(f"Warning: {cls.DATA_TAR} not found, skipping data extraction")
            return False
            
        print(f"Extracting data archive to: {target_dir}")
        
        try:
            with tarfile.open(cls.DATA_TAR, "r:gz") as tar:
                # Safety check for malicious archives
                def is_safe_member(member):
                    member_path = Path(member.name)
                    return not (
                        member_path.is_absolute() or 
                        ".." in member_path.parts or
                        member.name.startswith("/")
                    )
                
                # Filter and extract safe members
                safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                tar.extractall(path=target_dir, members=safe_members)
                
            print(f"Successfully extracted {len(safe_members)} items")
            return True
        except Exception as e:
            print(f"Error extracting data archive: {e}")
            return False


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

class CustomSdist(sdist):
    """Custom sdist command that ensures data archive is included"""
    
    def run(self):
        # Ensure data archive exists before creating source distribution
        print("=== Creating SSAPy Source Distribution ===")
        DataManager.create_data_archive()
        
        # Run the standard sdist process
        sdist.run(self)

class CustomDevelop(develop):
    """Custom develop command for editable installs"""
    
    def run(self):
        print("=== SSAPy Development Install ===")
        
        # For development installs, extract to the source directory if archive exists
        if os.path.exists(DataManager.DATA_TAR):
            DataManager.extract_data_archive()
        elif not os.path.exists(DataManager.DATA_DIR):
            print(f"Warning: Neither {DataManager.DATA_TAR} nor {DataManager.DATA_DIR} found")
            print("You may need to create the data archive or obtain the data files")
        
        # Run the standard develop process
        develop.run(self)

# Updated DataManager class for setup.py

class DataManager:
    """Utility class for managing SSAPy chunked data files"""
    
    DATA_DIR = "ssapy/data"
    CHUNK_PREFIX = "ssapy_data_chunk_"
    CHUNK_SIZE = 80 * 1024 * 1024  # 80 MB chunks
    
    @classmethod
    def create_chunked_archive(cls):
        """Create chunked tar archives of data directory"""
        if not os.path.exists(cls.DATA_DIR):
            print(f"Warning: {cls.DATA_DIR} directory not found, skipping archive creation")
            return False
            
        # Use the chunked data manager
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/chunked_data_manager.py", "create"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Chunked archive created successfully")
            return True
        else:
            print(f"Error creating chunked archive: {result.stderr}")
            return False
    
    @classmethod
    def reassemble_and_extract(cls, target_dir=None):
        """Reassemble chunks and extract data"""
        if target_dir is None:
            target_dir = os.path.dirname(cls.DATA_DIR)
            
        # Use the chunked data manager to reassemble
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/chunked_data_manager.py", 
            "reassemble", "--target", target_dir
        ], capture_output=True, text=True)
        
        return result.returncode == 0
    
    @classmethod
    def get_chunk_filenames(cls):
        """Get list of chunk filenames for package_data"""
        chunk_pattern = f"{cls.CHUNK_PREFIX}*.tar.gz"
        ssapy_dir = Path("ssapy")
        chunk_files = sorted(list(ssapy_dir.glob(chunk_pattern)))
        return [f.name for f in chunk_files]


# Updated CustomBuild class
class CustomBuild(build):
    """Custom build command that creates chunked data archive"""
    
    def run(self):
        print("=== SSAPy Custom Build Process (Chunked) ===")
        DataManager.create_chunked_archive()
        build.run(self)


# Updated CustomInstall class  
class CustomInstall(install):
    """Custom install command that reassembles chunks and extracts data"""
    
    def run(self):
        print("=== SSAPy Custom Install Process (Chunked) ===")
        install.run(self)
        self._reassemble_and_extract_data()
    
    def _reassemble_and_extract_data(self):
        """Reassemble chunks and extract data files"""
        # Find where the package was installed
        install_dir = None
        for path in sys.path:
            potential_path = os.path.join(path, "ssapy")
            if os.path.exists(potential_path):
                install_dir = potential_path
                break
        
        if install_dir is None:
            print("Warning: Could not locate installed ssapy package")
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
                    print(f"Reading chunk: {chunk_file.name}")
                    with open(chunk_file, 'rb') as chunk:
                        while True:
                            data = chunk.read(8192)
                            if not data:
                                break
                            output.write(data)
            
            # Extract the reassembled archive
            with tarfile.open(temp_archive, "r:gz") as tar:
                tar.extractall(path=install_dir)
                
            print("Data successfully reassembled and extracted")
            
            # Clean up temporary archive
            os.remove(temp_archive)
            
            # Optionally remove chunk files to save space
            for chunk_file in chunk_files:
                os.remove(chunk_file)
                
        except Exception as e:
            print(f"Error reassembling chunks: {e}")


def get_package_data():
    """Get package data including dynamic chunk files"""
    chunk_files = DataManager.get_chunk_filenames()
    return {
        'ssapy': ['_ssapy*.so'] + chunk_files
    }

setup(
    name='ssapy',
    version='1.2.0',
    ext_modules=[CMakeExtension("ssapy._ssapy")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build": CustomBuild,           # Uses chunked archive creation
        "sdist": CustomSdist, 
        "install": CustomInstall,       # Uses chunk reassembly
        "develop": CustomDevelop,
    },
    packages=find_packages(),
    package_data=get_package_data(),    # Dynamic chunk inclusion
    license='MIT',
    zip_safe=False,
    include_package_data=True,
    install_requires=get_dependencies(),
)