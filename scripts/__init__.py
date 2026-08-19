"""Runnable entry points for the pipeline.

Modules here are executables, not a library: each one is meant to be run as
``python -m scripts.<name>`` from the repository root, which is also where
their relative data paths (``data/``, ``results/``) resolve. Nothing is
exported, so importing this package has no side effects.

Available entry points
----------------------
run_baseline
    Retrieve, generate, and score the 300-question HotpotQA dev subset,
    writing ``results/baseline_week1.json``.
"""

from __future__ import annotations

__all__: list[str] = []
