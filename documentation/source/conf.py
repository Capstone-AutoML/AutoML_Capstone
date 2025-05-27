# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the root directory of the project to sys.path (for autodoc to work)
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'AutoML CI/CD/CT: Continuous Training and Deployment Pipeline'
copyright = '2025, Elshaday Yoseph, Nhan Tien Nguyen, Rongze Liu and Sepehr Heydarian'
author = 'Elshaday Yoseph, Nhan Tien Nguyen, Rongze Liu and Sepehr Heydarian'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',             # For Markdown support
    'sphinx.ext.autodoc',      # Auto-generate API docs from docstrings
    'sphinx.ext.napoleon',     # Support for Google/NumPy docstring styles
    'sphinx.ext.viewcode',     # Add links to source code
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for Markdown ----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Allows ::: blocks
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

