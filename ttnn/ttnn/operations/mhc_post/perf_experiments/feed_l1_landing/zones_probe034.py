# Perf 1: zone timeline (MHC_POST_PERF_ZONES=1), TL_FEED selects the feed call
import os, ttnn
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import LOOSE_CASES
from eval.golden_tests.mhc_post.helpers import *
import sys, importlib

sys.path.insert(
    0,
    "/home/malimpic/localdev/2026_11_13/2257_malimpic_mhc-post-fused/clones/mhc_post_run1/tt-metal/ttnn/ttnn/operations/mhc_post/perf_experiments/feed_l1_landing",
)
mhc_post = importlib.import_module(
    "mhc_post_l1" if os.environ.get("MHC_FL_VARIANT", "l1") == "l1" else "mhc_post_base"
).mhc_post

FEED = os.environ.get("TL_FEED", "1") == "1"
cache = _FabricMeshCache()
mesh = cache.get((1, 4))
try:
    case = LOOSE_CASES[0]
    inputs = case["inputs"]
    cases = make_mesh_case(
        inputs, 4, mode="collapse", coef_source="packed", norm="on", dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16
    )
    kw = dict(
        mode="collapse",
        coef_source="packed",
        mesh_device=mesh,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        op=mhc_post,
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
    )
    fo = feed_buffer(mesh, inputs[1], cluster_axis=1)
    fkw = (
        dict(feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=2, feed_cluster_axis=1)
        if FEED
        else {}
    )
    for rep in range(3):
        call_mhc_post_mesh(cases, **kw, **fkw)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    print("done")
finally:
    cache.close()
