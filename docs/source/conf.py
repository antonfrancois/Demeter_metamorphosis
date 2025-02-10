# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

# Ajoutez le chemin du module 'examples' à sys.path
sys.path.insert(0, os.path.abspath('../../examples'))
sys.path.insert(0, os.path.abspath('../../src/demeter'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Demeter_metamorphosis'
copyright = '2025, Anton François'
author = 'Anton François'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# !! Important !!
# Pour générer les fichiers .rst nécessaires dans le répertoire docs/source/.
# sphinx-apidoc -o docs/source/ src/demeter

ignore_files = [
    '__init__.py',
    '*.ipynb',
    'brats_utils.py',
    'visualize_geodesicOptim.py',
    'weightedMetamorphosis_3D.py',
    'toyExample_grayScott_JoinedMetamorphosis.py',
    'toyExample_grayScott_constrainedMetamorphosis.py',
    'brain_radioaide_metamorphosis.py',
]

# Générer le motif ignore_pattern
ignore_pattern = r'(' + '|'.join([f.replace('.', r'\.').replace('*', r'.*') for f in ignore_files]) + r')'

# Configuration de Sphinx Gallery
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # Répertoire contenant les exemples
    'gallery_dirs': 'auto_examples',          # Répertoire où les exemples générés seront stockés
    'within_subsection_order': FileNameSortKey,
     'ignore_pattern': ignore_pattern,
    "filename_pattern": r".*",  # Ensure it includes all files
}

extensions = [
     'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Pour supporter les docstrings Google et NumPy
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = []

# For maths

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']]
    }
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# # html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_static_path = ['_static']
