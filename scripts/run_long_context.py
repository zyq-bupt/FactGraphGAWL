#!/usr/bin/env python3
"""滑动窗口 FK-Graph 长文本实验 CLI。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from long_context.run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
