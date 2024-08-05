from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os


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


setup(
    name='SSAPy',
    version='0.7',
<<<<<<< HEAD
    description='Space Situational Awareness',
    author='Michael D. Schneider, William A. Dawson, Julia T. Ebert, Josh Meyers, Eddie Schlafly',
    author_email='meyers18@llnl.gov',
    url='https://github.com/LLNL/SSAPy',
=======
    description='Space Situational Awareness for Python',
    author='Michael Schneider, Joshua Meyers, Edward Schlafly, Julia Ebert, Travis Yeager',
    author_email='yeager7@llnl.gov',
    url='https://lc.llnl.gov/bitbucket/scm/mad/ssa.git',
>>>>>>> 7677c7e (Updated author email and fixed author list inconsistencies)
    packages=['ssapy'],
    package_dir={'ssapy': 'ssapy'},
    package_data={'ssapy': ['ssapy/**/*']},
    ext_modules=[CMakeExtension('ssapy._ssapy')],
    cmdclass=dict(build_ext=CMakeBuild),
    license='MIT',
    tests_require=[
        'pytest',
        'pytest-xdist'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'pyerfa',
        'emcee',
        'lmfit',
        'sgp4',
        'matplotlib',
        'pandas',
        'h5py',
        'pypdf2',
        'imageio',
        'ipython',
        'ipyvolume',
        'ipython_genutils',
        'jplephem',
        'tqdm',
        'myst-parser',
        'graphviz',
    ],
    zip_safe=False,
    include_package_data=True,
)
