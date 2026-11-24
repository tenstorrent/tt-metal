# Golden correctness of a sender variant on this dir's kernel copy: every SUPPORTED golden cell (arch-max packet) plus
# every LOOSE case (pinned packet).  VAR=sspart (A2AMM_ABLATE tokens), STEP=k runs every k-th golden cell.
import importlib.util
import os
import traceback
from pathlib import Path

from eval.feature_matrix import cartesian, invalid_reason, unsupported_reason
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from eval.golden_tests.all_to_all_matmul.feature_spec import INPUTS, INVALID, TARGET
from eval.golden_tests.all_to_all_matmul.helpers import run_all_to_all_matmul
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases
from ttnn.operations.all_to_all_matmul import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED

_spec = importlib.util.spec_from_file_location("_ss_common", Path(__file__).parent / "_common.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

os.environ["A2AMM_ABLATE"] = ",".join(t for t in os.environ.get("VAR", "").split(",") if t and t != "base")
step = int(os.environ.get("STEP", "1"))
cells = [
    (i, a)
    for i in INPUTS
    for a in cartesian(TARGET, INPUT_TAGGERS, i)
    if invalid_reason(a, INVALID) is None and unsupported_reason(a, SUPPORTED, EXCLUSIONS) is None
][::step]
cache = _FabricMeshCache()
ok = bad = 0
mesh = cache.get((1, 4))
for inputs, axes in cells:
    try:
        run_all_to_all_matmul(inputs, mesh_device=mesh, **axes)
        ok += 1
    except Exception:
        bad += 1
        print("FAIL", inputs, axes)
        traceback.print_exc()
for inputs, axes, extras in (c.values for c in _loose_cases()):
    mesh = cache.get((1, 4), payload_bytes=extras.get("fabric_payload_bytes"))
    try:
        run_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, **axes)
        ok += 1
    except Exception:
        bad += 1
        print("FAIL loose", inputs, axes)
        traceback.print_exc()
cache.close()
print(f"GOLDEN {os.environ['A2AMM_ABLATE'] or 'base'}: {ok} pass, {bad} fail")
assert bad == 0
