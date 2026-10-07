"""Host-only (no device): the auto depth rule (depth_patch._choose_depth) over every golden INPUTS shape x dtype x
fp32_dest x scatter_dim, at G (env G, default 4) on the 11x9 LoudBox compute rectangle. Prints every cell whose auto
depth is < G (fallback) and a summary."""
import collections
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ttnn
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as pd

pd._tile_bytes = lambda dt: {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dt]
pd._seg_tiles = lambda blk_n: max(1, min(7, blk_n))
os.environ["MMRS_HANDOFF_DEPTH"] = "auto"
import depth_patch  # noqa: E402
from eval.golden_tests.matmul_reduce_scatter.feature_spec import INPUTS  # noqa: E402

G = int(os.environ.get("G", "4"))
l1_free = 1461248 - pd.L1_RESERVE
hist = collections.Counter()
for (a, w), ad, wd, fp32, sd in itertools.product(
    INPUTS, [ttnn.bfloat16, ttnn.bfloat8_b], [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], [False, True], [-1, -2]
):
    M, K, N = a[-2], a[-1], w[-1]
    if (M if sd == -2 else N) % (32 * G):
        continue
    kw = dict(
        comp_rows=9,
        comp_cols=11,
        Mt=M // 32,
        Kt=K // 32,
        Nt=N // 32,
        G=G,
        scatter_dim=sd,
        a_dtype=ad,
        w_dtype=wd,
        fp32_acc=fp32,
    )
    try:
        d, b = depth_patch._choose_depth(kw, l1_free, G)
    except ValueError as e:
        print("plan error", M, K, N, sd, e)
        continue
    hist[d] += 1
    if d < G:
        print(
            f"fallback depth {d}: M{M} K{K} N{N} sd{sd} a={ad} w={wd} fp32={fp32} {b.regime} kbt{b.k_block_tiles} "
            f"core {b.core_m_tiles}x{b.core_n_tiles}"
        )
print("auto depth histogram:", dict(hist))
