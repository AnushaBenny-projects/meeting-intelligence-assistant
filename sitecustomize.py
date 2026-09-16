from __future__ import annotations

import os
import sys


if any("pytest" in part.lower() for part in sys.argv):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
