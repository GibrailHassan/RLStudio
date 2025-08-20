from __future__ import annotations
import os, sys
from datetime import datetime

project = "RLStudio"
author = "RLStudio Contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.0.0"

root_dir = os.path.abspath(os.path.join(__file__, "..", "..", "src"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = ["colon_fence"]
autosummary_generate = True
html_theme = "alabaster"
html_theme_options = {"description": "Early-stage reinforcement learning framework"}
exclude_patterns: list[str] = ["_build"]
html_static_path: list[str] = []
python_use_unqualified_type_names = True
rst_epilog = ".. |project| replace:: RLStudio"
