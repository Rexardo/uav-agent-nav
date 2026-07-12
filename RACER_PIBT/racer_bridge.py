"""Load the unchanged RACER modules from the sibling directory."""

from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
RACER_DIR = MODULE_DIR.parent / "RACER"

# Prefer copied/shared interfaces in RACER_PIBT, then load the remaining
# unchanged map, HGrid, planner, and simulator modules from RACER.
for path in (str(MODULE_DIR), str(RACER_DIR)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(1, str(RACER_DIR))
