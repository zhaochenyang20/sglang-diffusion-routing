"""Pytest configuration: force local src import precedence."""
from __future__ import annotations

import sys
from pathlib import Path

src_str = str(Path(__file__).resolve().parent.parent / "src")
while src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)
