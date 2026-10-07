"""Host-only (no device): per case, the plan (regime, K-block, L1 footprint) at hand-off depth 2..G.

Replicates matmul_reduce_scatter._get_plan's blocking convergence loop with HANDOFF_DEPTH patched.
Usage: python3 plan_fit.py [comp_rows comp_cols l1_bank_bytes]"""
import sys

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as pd
import ttnn

# host-only: no cluster/fabric queries (tile sizes and the 14 KiB payload segment hard-coded)
pd._tile_bytes = lambda dt: {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dt]
SEG = 7

CASES = {
    "focus": ((640, 2048), (2048, 7168), -1, ttnn.bfloat8_b, False),
    "glm": ((640, 4096), (4096, 6144), -1, ttnn.bfloat8_b, False),
    "mimo": ((2048, 2048), (2048, 4096), -2, ttnn.bfloat8_b, True),
    "smallk": ((640, 512), (512, 7168), -1, ttnn.bfloat8_b, False),
    "r2": ((2048, 4096), (4096, 4096), -1, ttnn.bfloat8_b, True),
}
rows, cols, bank = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (9, 11, 1461248)
G = int(__import__("os").environ.get("G", "4"))
l1_free = bank - pd.L1_RESERVE


def plan(depth, **kw):
    pd.HANDOFF_DEPTH = depth
    hb = lambda b: b.handoff_slots * b.core_m_tiles * b.core_n_tiles * pd.BF16_TILE_BYTES
    blk = pd._plan_blocking(**kw, seg_tiles=SEG, l1_cb_budget=l1_free)
    for _ in range(3):
        nxt = pd._plan_blocking(**kw, seg_tiles=SEG, l1_cb_budget=l1_free - hb(blk))
        if hb(nxt) <= hb(blk):
            return nxt, hb(nxt)
        blk = nxt
    raise ValueError("no converge")


for name, ((M, K), (_, N), sd, wdt, fp32) in CASES.items():
    kw = dict(
        comp_rows=rows,
        comp_cols=cols,
        Mt=M // 32,
        Kt=K // 32,
        Nt=N // 32,
        G=G,
        scatter_dim=sd,
        a_dtype=ttnn.bfloat16,
        w_dtype=wdt,
        fp32_acc=fp32,
    )
    for d in range(2, G + 1):
        try:
            b, h = plan(d, **kw)
        except ValueError as e:
            print(f"{name} depth {d}: DOES NOT FIT ({e})")
            continue
        a_pages = b.core_m_tiles * b.Kt if b.a_resident else pd.OPERAND_DEPTH * b.core_m_tiles * b.k_block_tiles
        w_pages = b.Kt * b.core_n_tiles if b.w_resident else pd.OPERAND_DEPTH * b.k_block_tiles * b.core_n_tiles
        cbs = a_pages * b.a_tile_bytes + w_pages * b.w_tile_bytes + b.core_m_tiles * b.core_n_tiles * b.acc_tile_bytes
        print(
            f"{name} depth {d}: {b.regime} core {b.core_m_tiles}x{b.core_n_tiles} kbt {b.k_block_tiles} "
            f"handoff {h} B, CBs {cbs} B, total {cbs + h} B of {l1_free} (free {l1_free - cbs - h})"
        )
