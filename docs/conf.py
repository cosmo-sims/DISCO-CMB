"""Sphinx configuration for the DISCO-CMB documentation.

Pages are MyST Markdown and the API reference is autodoc. autodoc imports the
package, so the docs environment installs DISCO-CMB with its dependencies.
Importing needs no GPU.
"""

project = "DISCO-CMB"
author = "Lawrence Berry and Oliver Hahn"
copyright = "Lawrence Berry and Oliver Hahn"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

exclude_patterns = ["_build"]
# layout.html there removes the theme's top navbar.
templates_path = ["_templates"]

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",  # ::: fences, for sphinx-design's grids and dropdowns
    "deflist",
    "dollarmath",
    "html_image",
]
# `[text](page.md#a-heading)` links resolve to headings up to this depth.
myst_heading_anchors = 3

# -- autodoc ------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
autoclass_content = "both"
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# -- HTML ---------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "DISCO-CMB"
html_logo = "assets/disco-cmb.png"
html_static_path: list[str] = []
html_theme_options = {
    "repository_url": "https://github.com/cosmo-sims/DISCO-CMB",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
