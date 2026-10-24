"""Market Risk Model Validation Workbench (RVW)."""

import os
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

__version__ = "0.1.0"
