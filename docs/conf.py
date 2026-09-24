"""Sphinx documentation configuration for so101-camera-multiview-bench.

Documentation for a camera-viewpoint experiment framework for imitation
learning with the SO-ARM101 manipulator in Isaac Lab.

Adapted from third_party/imitation/docs/conf.py (HumanCompatibleAI).
"""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make the package importable for autodoc (the src layout requires this).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "so101-camera-multiview-bench"
copyright = "2026, Ilja Schirschow"
author = "Ilja Schirschow"

try:
    from importlib import metadata
    version = metadata.version("so101-camera-multiview-bench")
except Exception:
    version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    # myst_nb supersedes myst_parser — do NOT load both, they conflict.
    "myst_nb",
]

# Markdown / MyST setup ------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Notebook execution: OFF for RTD (Isaac Sim / GPU unavailable in sandbox).
# Notebooks must be executed locally and committed with outputs.
nb_execution_mode = os.getenv("NB_EXECUTION_MODE", "off")

# Napoleon (Google-style docstrings) ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

# Autodoc --------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    # NOT undoc-members: dataclass fields that already appear in a napoleon
    # "Attributes:" docstring section would be documented twice, which is
    # exactly the "duplicate object description" warning class.
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"

# RTD sandbox cannot import Isaac Sim / heavy ML deps — mock them so that
# autodoc can still read docstrings from package modules.
autodoc_mock_imports = [
    # Isaac Sim / Lab
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaacsim",
    "omni",
    "carb",
    "pxr",
    "usdrt",
    # LeRobot / ML
    "lerobot",
    "lerobot_so101_teleop",
    "torch",
    "torchvision",
    "torchcodec",
    "torchaudio",
    "datasets",
    "huggingface_hub",
    # CV / numeric
    "cv2",
    "PIL",
    "matplotlib",
    "pandas",
    "scipy",
    "skimage",
    "imageio",
    # Hardware / IO
    "pygame",
    "scservo_sdk",
    "feetech_servo_sdk",
    "evdev",
    "rerun",
    "rerun_sdk",
    "numpy",
    "pyarrow",
    "gymnasium",
]

# Intersphinx ----------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "SO-101 Multi-View Imitation-Learning Framework"
html_short_title = "SO-101 Docs"
# Every repository-backed feature of this theme is off, because the repository is
# not published yet and each of them renders a link. A dead button is worse than
# an absent one, and a placeholder URL is not an option.
#
# To switch them on after publishing, add back:
#   "repository_url": "https://github.com/<owner>/so101-camera-multiview-bench",
#   "repository_branch": "main",
#   "path_to_docs": "docs",
#   "use_repository_button" / "use_source_button" / "use_edit_page_button" /
#   "use_issues_button": True,
#   "icon_links": [{"name": "GitHub", "url": <same url>, "icon": "fa-brands fa-github"}],
# and fill in [project.urls] in pyproject.toml at the same time.
html_theme_options = {
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

templates_path = ["_templates"]
# tools/ holds generator scripts and the tables they emit. The tables are
# pulled into pages via {include}; they are not standalone documents.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
    "tools/*.md",
]

html_static_path = ["_static"]

# Copybutton ----------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True

# Suppress noisy warnings during early iteration -----------------------------
suppress_warnings = ["myst.header"]


def setup(app):
    """Hook for future extension wiring (custom roles, event listeners)."""
    return {"parallel_read_safe": True, "parallel_write_safe": True}
