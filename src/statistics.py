"""Compatibility statistics API.

Historically this project exposed statistical helpers at ``src.statistics``.
The primary implementation now lives in :mod:`src.experiments.statistics`.
"""
from __future__ import annotations

from .experiments.statistics import *  # noqa: F401,F403
