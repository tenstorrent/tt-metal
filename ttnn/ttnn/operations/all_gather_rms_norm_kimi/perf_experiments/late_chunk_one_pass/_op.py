"""Import the experiment's op copy (agrn_copy) by path; knobs via env: AGRN_NORM_IMPL, AGRN_LATE_BLOCK, AGRN_XG_LATE."""
import sys
from pathlib import Path

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
from agrn_copy import all_gather_rms_norm_kimi  # noqa: E402,F401
