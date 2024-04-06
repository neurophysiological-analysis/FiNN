
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

html_context = {
  'current_version' : "1.4.0",
  'versions' : [["1.4.0", "link to 1.4.0"], ],
}

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
        '**': [
            "search-field.html", 'globaltoc.html', 'versions.html'
        ],
    }


sys.path.insert(0, os.path.abspath('/mnt/data/Professional/projects/finnpy/latest/docs/'))
import build_docs

# get the environment variable build_all_docs and pages_root
build_all_docs = os.environ.get("build_all_docs")
pages_root = os.environ.get("pages_root", "")

# if not there, we dont call this
if build_all_docs is not None:
  # we get the current version
  current_version = os.environ.get("current_version")

  # we set the html_context to the current version 
  # and empty versions for now
  html_context = {
    'current_version' : current_version,
    'versions' : [],
  }

  # and we append all versions accordingly 
  # we treat the main branch as latest 
  if (current_version == 'develop'):
    html_context['versions'].append(['latest', pages_root])

  # and loop over all other versions from our yaml file
  # to set versions
  import pathlib
  import yaml
  with open(str(pathlib.Path().resolve()) + "/../versions.yaml", "r") as yaml_file:
    docs = yaml.safe_load(yaml_file)

  for version, details in docs.items():
    html_context['versions'].append([version, pages_root+'/'+version])























