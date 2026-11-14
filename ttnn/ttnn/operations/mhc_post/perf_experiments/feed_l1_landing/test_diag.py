# feed_l1_landing diagnosis: which D tiles are wrong (links=FL_LINKS, T=FL_T, C=FL_C)
import os, sys, importlib, ttnn

sys.path.insert(
    0,
    "/home/malimpic/localdev/2026_11_13/2257_malimpic_mhc-post-fused/clones/mhc_post_run1/tt-metal/ttnn/ttnn/operations/mhc_post/perf_experiments/feed_l1_landing",
)
mhc_post = importlib.import_module(
    "mhc_post_l1" if os.environ.get("MHC_FL_VARIANT", "l1") == "l1" else "mhc_post_base"
).mhc_post
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import *
from eval.golden_tests.mhc_post.helpers import pytorch_feed, chip_results

T, C, L = int(os.environ.get("FL_T", "160")), int(os.environ.get("FL_C", "5120")), int(os.environ.get("FL_LINKS", "1"))
from eval.golden_tests.mhc_post.conftest import _fabric_mesh_cache, mesh_device  # noqa


def test_diag(mesh_device):
    mesh = mesh_device
    inputs = _case((1, 1, T, 4 * C))
    cases = make_mesh_case(
        inputs,
        4,
        mode="collapse",
        coef_source="packed",
        seed=7,
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
    if os.environ.get("FL_PLAIN", "1") == "1":
        plain = call_mhc_post_mesh([without_norm(c) for c in cases], **kw)
    want = call_mhc_post_mesh(cases, **kw)
    if os.environ.get("FL_CHECK", "1") == "1":
        check_mesh_case(
            want, cases, mode="collapse", dtype=ttnn.bfloat16, math_fidelity=ttnn.MathFidelity.HiFi2, plain_ref=plain
        )
    fo = feed_buffer(mesh, inputs[1], cluster_axis=1)
    for rep in range(2):
        got = call_mhc_post_mesh(
            cases, **kw, feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=L, feed_cluster_axis=1
        )
        chips = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(fo)]
        ys = [ttnn.to_torch(r[3]) for r in chip_results(want)]
        exp = pytorch_feed(ys, (1, 4), 1)
        for i, (g, w) in enumerate(zip(chips, exp)):
            g = g.bfloat16().reshape(-1, g.shape[-1])
            w = w.bfloat16().reshape(-1, w.shape[-1])
            bad = g != w
            if bad.any():
                tiles = sorted({(int(r) // 32, int(c) // 32) for r, c in bad.nonzero()})
                print(f"rep {rep} chip {i}: {int(bad.sum())} bad elems, {len(tiles)} tiles: {tiles[:40]}")
                r0 = tiles[0]
                blk = bad[r0[0] * 32 : (r0[0] + 1) * 32, r0[1] * 32 : (r0[1] + 1) * 32]
                print(
                    "   first bad tile rows:",
                    sorted({int(r) for r, c in blk.nonzero()}),
                    "cols:",
                    sorted({int(c) for r, c in blk.nonzero()})[:40],
                )
            else:
                print(f"rep {rep} chip {i}: ok")
