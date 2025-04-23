# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SSAPy'
copyright = '2018, Lawrence Livermore National Security, LLC'
author = 'Michael Schneider, Josh Meyers, Edward Schlafly, Julia Ebert, Travis Yeager, et al.'

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
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosectionlabel',
]

source_suffix = {
	'.rst': 'restructuredtext',
	'.txt': 'markdown',
	'.md': 'markdown',
}

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/LLNL/SSAPy/tree/main/%s.py" % filename

autosummary_generate = True
numpydoc_show_class_members = False
autodoc_member_order = 'bysource'
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

bibtex_cache = 'none' 
bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"
bibtex_debug = True
