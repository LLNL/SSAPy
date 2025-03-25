# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import importlib
# import inspect
from functools import reduce
import subprocess

import os
import sys

def install_cmake():
    cmake_url = "https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.tar.gz"
    cmake_tar = "cmake.tar.gz"
    cmake_dir = "cmake-3.27.0-linux-x86_64"
    local_bin = os.path.expanduser("~/bin")

    # Download cmake
    subprocess.run(["curl", "-L", cmake_url, "-o", cmake_tar], check=True)

    # Extract cmake
    subprocess.run(["tar", "-xzf", cmake_tar], check=True)

    # Create local bin directory if it doesn't exist
    os.makedirs(local_bin, exist_ok=True)

    # Move cmake binary to local bin
    subprocess.run(["mv", f"{cmake_dir}/bin/cmake", local_bin], check=True)

    # Add local bin to PATH
    os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"

    # Verify cmake installation
    cmake_version = subprocess.run(["cmake", "--version"], check=True, capture_output=True, text=True)
    print("CMake installed successfully:")
    print(cmake_version.stdout)

# on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# if on_rtd:
#     install_cmake()
#     sys.path.insert(0, os.path.abspath('../../ssapy'))
#     subprocess.run(["python3", "setup.py", "build"], cwd=os.path.abspath("../.."))
#     subprocess.run(["python3", "setup.py", "install"], cwd=os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SSAPy'
copyright = '2018, Lawrence Livermore National Security, LLC'
author = 'Michael Schneider, Josh Meyers, Edward Schlafly, Julia Ebert, Travis Yeager, et al.'

githash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# def linkcode_resolve(domain, info):
#     if domain != 'py':
#         return None
#     if not info['module']:
#         return None
#     filename = info['module'].replace('.', '/')
#     return "https://github.com/LLNL/SSAPy/tree/main/ssapy/%s.py" % filename

autosummary_generate = True
numpydoc_show_class_members = False
sphinx_tabs_valid_builders = ['linkcheck']
source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
}
tls_verify = False
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/images/logo/ssapy_logo.svg"
html_theme_options = {
    'logo_only': True,
}
html_favicon = '_static/images/logo/ssapy_logo.ico'

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"
bibtex_debug = True
