# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to Python path
project_root = os.path.abspath('..')
sys.path.insert(0, project_root)

# Debug: Print sys.path and try importing
if 'READTHEDOCS' not in os.environ:
    print(f"Python path: {sys.path}")
    try:
        import pyppur
        print(f"Successfully imported pyppur from {pyppur.__file__}")
    except ImportError as e:
        print(f"Failed to import pyppur: {e}")
        # Try to import the main class directly
        try:
            from pyppur.projection_pursuit import ProjectionPursuit
            print("Successfully imported ProjectionPursuit")
        except ImportError as e2:
            print(f"Failed to import ProjectionPursuit: {e2}")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Read metadata from installed package to avoid duplication with pyproject.toml
import datetime

try:
    from importlib.metadata import metadata
    pkg_metadata = metadata('pyppur')
    project = pkg_metadata['Name']
    # Author might be in 'Author' or parsed from 'Author-email'
    author = pkg_metadata.get('Author') or pkg_metadata.get('Author-email', '').split(' <')[0]
    release = pkg_metadata['Version']
    version = '.'.join(release.split('.')[:2])  # Major.minor version
except Exception:
    # Fallback: parse pyproject.toml directly if package not installed
    import tomllib
    import pathlib
    
    pyproject_path = pathlib.Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject_path, 'rb') as f:
        pyproject_data = tomllib.load(f)
    
    project_info = pyproject_data['project']
    project = project_info['name']
    author = project_info['authors'][0]['name']
    release = project_info['version']
    version = '.'.join(release.split('.')[:2])

copyright = f'{datetime.date.today().year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Handle import errors gracefully
autodoc_mock_imports = []
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# autosummary configuration
autosummary_generate = True
autosummary_imported_members = True

# napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# mathjax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# HTML theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}