
import sys, os
sys.path.insert(0, os.path.abspath('/mnt/data/Professional/projects/finnpy/latest/finnpy'))
sys.path.insert(0, os.path.abspath('/mnt/data/Professional/projects/finnpy/latest/'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FiNN'
copyright = '2022, Maximilian Scherer'
author = 'Maximilian Scherer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_toolbox.collapse']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

#Ignore import mayavi as sphinx doesn't like it
autodoc_mock_imports = ["mayavi"]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
        '**': [
            "search-field.html", 'globaltoc.html', 'versions.html'
        ],
    }


























