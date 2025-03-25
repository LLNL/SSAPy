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

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    # Initialize Git submodules
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=os.path.abspath("../.."), check=True)

    # Install Git LFS
    subprocess.run(["git", "lfs", "install"], cwd=os.path.abspath("../.."), check=True)

    # Pull files managed by Git LFS
    subprocess.run(["git", "lfs", "pull"], cwd=os.path.abspath("../.."), check=True)

    # Add the ssapy directory to the Python path
    sys.path.insert(0, os.path.abspath('../../ssapy'))

    # Build and install the package
    subprocess.run(["python3", "setup.py", "build"], cwd=os.path.abspath("../.."), check=True)
    subprocess.run(["python3", "setup.py", "install"], cwd=os.path.abspath("../.."), check=True)


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

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/LLNL/SSAPy/tree/main/ssapy/%s.py" % filename

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
