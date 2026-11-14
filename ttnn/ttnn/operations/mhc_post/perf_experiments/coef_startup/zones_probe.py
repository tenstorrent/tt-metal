# Zone profile of the focus no-feed call (TL_FEED=1 -> feed) on my copies; MHC_CS_VARIANT=cs|base.
# Run: TT_METAL_DEVICE_PROFILER=1 MHC_POST_PERF_ZONES=reader scripts/run_safe_pytest.sh <this> -s --import-mode=prepend
import importlib, os, sys
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import LOOSE_CASES
from eval.golden_tests.mhc_post.helpers import call_mhc_post_mesh, feed_buffer, feed_semaphores, make_mesh_case

mhc_post = importlib.import_module(
    "mhc_post_cs" if os.environ.get("MHC_CS_VARIANT", "cs") == "cs" else "mhc_post_base"
).mhc_post


def test_zones():
    feed = os.environ.get("TL_FEED", "0") == "1"
    cache = _FabricMeshCache()
    mesh = cache.get((1, 4))
    try:
        inputs = LOOSE_CASES[0]["inputs"]
        cases = make_mesh_case(
            inputs,
            4,
            mode="collapse",
            coef_source="packed",
            norm="on",
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
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
        fkw = {}
        if feed:
            fo = feed_buffer(mesh, inputs[1], cluster_axis=1)
            fkw = dict(feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=2, feed_cluster_axis=1)
        for _ in range(int(os.environ.get("ZP_CALLS", "3"))):
            call_mhc_post_mesh(cases, **kw, **fkw)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
    finally:
        cache.close()
