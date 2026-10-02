"""
Shared pytest configuration for the pipeline suite.

Adapters whose dependency is heavy enough to stay commented out of
`requirements.txt` until the sprint that needs it are imported at module scope
by their test module. One missing package therefore aborts collection for the
WHOLE run: `main` currently reports "1 error during collection" and executes
zero tests, which reads as a broken suite rather than as an absent optional
dependency.

Naming the module and the package it needs here turns that into an honest skip
and keeps the mapping in one place. `fasttext` is deliberately absent from this
table: the adapter imports it lazily inside `__init__` and falls back to a
heuristic, so its tests run with no optional dependency installed.
"""

from __future__ import annotations

import importlib.util

_OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "test_pdf_text_extractor.py": "pymupdf",
    "test_sql_repositories.py": "sqlalchemy",
}

collect_ignore = [
    module
    for module, package in _OPTIONAL_DEPENDENCIES.items()
    if importlib.util.find_spec(package) is None
]
