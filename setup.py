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


class CustomBuild(build):
    """Custom build command that creates data archive before building"""
    
    def run(self):
        # Create the data archive before building
        print("=== SSAPy Custom Build Process ===")
        DataManager.create_data_archive()
        
        # Run the standard build process
        build.run(self)


class CustomSdist(sdist):
    """Custom sdist command that ensures data archive is included"""
    
    def run(self):
        # Ensure data archive exists before creating source distribution
        print("=== Creating SSAPy Source Distribution ===")
        DataManager.create_data_archive()
        
        # Run the standard sdist process
        sdist.run(self)


class CustomInstall(install):
    """Custom install command that extracts data files after installation"""
    
    def run(self):
        # Run the standard installation first (including CMake build)
        print("=== SSAPy Custom Install Process ===")
        install.run(self)
        
        # Then extract data files to the installed package location
        self._extract_data_files()
    
    def _extract_data_files(self):
        """Extract data files to the installed package location"""
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
            
        print(f"Installing data files to: {install_dir}")
        
        # Look for the data archive in the package directory
        package_tar = os.path.join(install_dir, "ssapy_data.tar.gz")
        
        if os.path.exists(package_tar):
            try:
                # Extract the archive directly to the package directory
                with tarfile.open(package_tar, "r:gz") as tar:
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
                    
                print(f"Successfully extracted {len(safe_members)} data files")
                
                # Optionally remove the archive after extraction to save space
                # Uncomment the next line if you want to clean up the archive
                # os.remove(package_tar)
                
            except Exception as e:
                print(f"Warning: Failed to extract data files: {e}")
        else:
            print(f"Warning: Data archive not found at {package_tar}")


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


setup(
    name='ssapy',
    version='1.1.1',
    
    # CMake extension (preserved from original)
    ext_modules=[CMakeExtension("ssapy._ssapy")],
    
    # Combined command classes (CMake + Data handling)
    cmdclass={
        "build_ext": CMakeBuild,        # Original CMake build
        "build": CustomBuild,           # New: Creates data archive
        "sdist": CustomSdist,           # New: Includes archive in source dist
        "install": CustomInstall,       # New: Extracts data on install
        "develop": CustomDevelop,       # New: Handles editable installs
    },
    
    # Package configuration
    packages=find_packages(),
    
    # Package data - include both CMake artifacts and data archive
    package_data={
        'ssapy': [
            '_ssapy*.so',           # Original CMake shared library
            'ssapy_data.tar.gz',    # New: Data archive
        ]
    },
    
    # Metadata (preserved from original)
    license='MIT',
    zip_safe=False,
    include_package_data=True,
    install_requires=get_dependencies(),
)