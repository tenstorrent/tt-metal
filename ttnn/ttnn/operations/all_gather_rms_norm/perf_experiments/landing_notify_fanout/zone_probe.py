# Zone probe on agrms_copy: 3 untraced calls of LOOSE_CASES[AGRMS_ZP_CASE] (default 0), then ReadDeviceProfiler.
# Run: AGRMS_PROFILE_ZONES=1 TT_METAL_DEVICE_PROFILER=1 [knobs] scripts/tt-probe.sh all_gather_rms_norm <<< \
#      'exec(open("<this file>").read())'   ; then python /tmp/p2/agg.py generated/profiler/.logs/profile_log_device.csv
import os, sys

sys.path.insert(0, "ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/landing_notify_fanout")
import ttnn
from agrms_copy import all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases

c = _loose_cases()[int(os.environ.get("AGRMS_ZP_CASE", "0"))]
inputs, axes, extras = c.values
cache = _FabricMeshCache()
mesh = cache.get((1, 4))
call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes)
for _ in range(3):
    out = call()
    if not os.environ.get("AGRMS_ABLATE"):
        check(out)
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
cache.close()
print("PROBE_DONE")
