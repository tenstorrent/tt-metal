# port_stream_rate: 1-call zone timeline of agrms_copy on the focus case (or AGRMS_PERF_* overrides).
# Run: AGRMS_PROFILE_ZONES=1 TT_METAL_DEVICE_PROFILER=1 PSR_ZONE_OUT=<csv> scripts/tt-probe.sh all_gather_rms_norm < zone_probe.py
import os, shutil, sys

sys.path.insert(0, "ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/port_stream_rate")
import ttnn
from agrms_copy import all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases

c = _loose_cases()[int(os.environ.get("PSR_CASE", "0"))]
inputs, axes, extras = c.values
if os.environ.get("AGRMS_PERF_NUM_LINKS"):
    axes = dict(axes, num_links=int(os.environ["AGRMS_PERF_NUM_LINKS"]))
cache = _FabricMeshCache()
mesh = cache.get((1, 4))
call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes)
for _ in range(3):
    out = call()
if not os.environ.get("AGRMS_ABLATE"):
    check(out)
    print("GOLDEN_OK")
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
cache.close()
dst = os.environ.get("PSR_ZONE_OUT")
if dst:
    shutil.copy("generated/profiler/.logs/profile_log_device.csv", dst)
print("PROBE_DONE")
