from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import subprocess
import sys
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--', '-j4']
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if 'CMAKE_VERBOSE_MAKEFILE' in env:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=1']

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        print("Downloading ssapy/data directory from GitHub...")
        repo_url = "https://github.com/LLNL/SSAPy.git"
        commit_hash = "0ea3174"
        temp_repo_dir = "temp_repo"
        target_dir = os.path.join(os.path.dirname(__file__), "ssapy", "data")

        try:
            subprocess.run(["git", "clone", repo_url, temp_repo_dir], check=True)
            subprocess.run(["git", "-C", temp_repo_dir, "checkout", commit_hash], check=True)
            subprocess.run(["cp", "-r", os.path.join(temp_repo_dir, "ssapy/data"), target_dir], check=True)
            print(f"Data directory downloaded to {target_dir}")
        finally:
            subprocess.run(["rm", "-rf", temp_repo_dir], check=True)

setup(
    packages=find_packages(exclude=["ssapy.data", "ssapy.data.*"]),
    ext_modules=[CMakeExtension('ssapy._ssapy')],
    cmdclass={
        'build_ext': CMakeBuild,
        'install': PostInstallCommand
    },
    zip_safe=False,
    include_package_data=True,
)