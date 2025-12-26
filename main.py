"""Compatibility runner to execute the calibration CLI without installing the package."""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from calibration.app import main as _main

    _main()


if __name__ == "__main__":
    _bootstrap()
