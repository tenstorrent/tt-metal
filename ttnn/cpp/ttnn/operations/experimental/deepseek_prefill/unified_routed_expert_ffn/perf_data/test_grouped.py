# Correctness + quick perf driver for the GROUPED unified_routed_expert_moe path (scratch).
# usage: python test_grouped.py [--G 1,2,4] [--rows 8] [--dists kimi_u,m3_u4] [--dtype bf4] [--layout rm|tile|both] [--perf]
import argparse
import os
import sys
import time

from loguru import logger

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import MultiExpertCase, OP_KERNEL_DIR, append_jsonl, env_info, open_dev, timed  # noqa: E402

DISTS = {
    "kimi_u": ("kimi", [107] * 12),
    "kimi_e4": ("kimi", [107] * 4),
    "kimi_e4_32": ("kimi", [32] * 4),
    "kimi_e10": ("kimi", [107] * 10),
    "kimi_e20": ("kimi", [107] * 20),
    "kimi_e2": ("kimi", [107] * 2),
    "kimi_e4_512": ("kimi", [512] * 4),
    "kimi7_e4": ("kimi7", [107] * 4),
    "kimi7_u": ("kimi7", [107] * 12),
    "m3_7_u4": ("m3_7", [160] * 4),
    "kimi_e1": ("kimi", [128]),
    "kimi_zipf": ("kimi", [640, 320, 224, 160, 128, 96, 64, 64, 32, 32, 0, 0]),
    "kimi_zeros": ("kimi", [0, 320, 0, 160, 96, 0, 0, 64, 0, 0, 0, 32]),
    "kimi_giant": ("kimi", [2048] + [64] * 11),
    "kimi_prod5120": ("kimi", [5120] + [107] * 11),
    "kimi_e3": ("kimi", [100, 200, 300]),
    "kimi_single_nonempty": ("kimi", [0, 0, 160, 0, 0, 0, 0, 0]),
    "kimi_empty": ("kimi", [0] * 12),
    "kimi_e24": ("kimi", [107] * 24),
    "m3_u4": ("m3", [160] * 4),
    "m3_u8": ("m3", [160] * 8),
    "m3_u16": ("m3", [160] * 16),
    "m3_skew8": ("m3", [800, 400, 200, 100, 50, 25, 0, 5]),
    "m3_giant4": ("m3", [2048, 64, 32, 32]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", default="1,2,4")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--mmax", type=int, default=0)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--strided", type=int, default=0)
    ap.add_argument("--down_split", type=int, default=1)
    ap.add_argument("--dists", default="kimi_e1,kimi_e4,m3_u4")
    ap.add_argument("--dtype", default="bf4")
    ap.add_argument("--layout", default="rm")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="also run num_row_groups=0 for A/B")
    ap.add_argument("--repeat", type=int, default=1, help="dispatch the op N times (cache-hit path)")
    a = ap.parse_args()
    dev = open_dev()
    try:
        info = env_info(dev)
        logger.info(f"env: {info}")
        layouts = ["rm", "tile"] if a.layout == "both" else [a.layout]
        gs = [int(g) for g in a.G.split(",")]
        if a.legacy:
            gs = [0] + gs
        fails = 0
        for dist in a.dists.split(","):
            model, counts = DISTS[dist]
            for layout in layouts:
                for G in gs:
                    ffn = (
                        dict(
                            ffn_num_row_groups=G,
                            ffn_grid_rows=a.rows,
                            ffn_grid_cols=a.cols,
                            ffn_per_core_m_max=a.mmax,
                            ffn_weight_cb_depth=a.depth,
                            ffn_col_strided=a.strided,
                            ffn_down_split=a.down_split,
                        )
                        if G > 0
                        else {}
                    )
                    tag = f"{dist} {a.dtype} {layout} G={G} rows={a.rows} mmax={a.mmax} D={a.depth} strided={a.strided} ds={a.down_split}"
                    t0 = time.time()
                    try:
                        case = MultiExpertCase(
                            dev, counts, model, dtype_key=a.dtype, x_row_major=(layout == "rm"), **ffn
                        )
                        for _ in range(a.repeat):
                            case.run()
                        ttnn.synchronize_device(dev)
                        res = case.check() if any(c > 0 for c in counts) else []
                        ok = all(r["ok"] for r in res)
                        pmin = min([r["pcc"] for r in res], default=None)
                        perf = None
                        if a.perf:
                            perf = timed(dev, case.run, OP_KERNEL_DIR, iters=3, warmup=1)
                            chunk_tok = 32 * (a.mmax or 4) * ((a.rows or 8) // max(G, 1)) if G > 0 else 2048
                            wb, fl = case.bytes_and_flops(chunk_tok)
                            perf.update(GBps=wb / perf["ns"], TFLOPs=fl / perf["ns"] / 1e3)
                        logger.info(
                            f"{'PASS' if ok else 'FAIL'} {tag} pcc_min={pmin} "
                            + (f"ns={perf['ns']:.0f} GB/s={perf['GBps']:.1f} TF={perf['TFLOPs']:.1f}" if perf else "")
                            + f" ({time.time()-t0:.1f}s)"
                        )
                        append_jsonl(
                            "grouped.jsonl",
                            dict(
                                section="grouped",
                                dist=dist,
                                model=model,
                                counts=counts,
                                dtype=a.dtype,
                                layout=layout,
                                G=G,
                                rows=a.rows,
                                cols=a.cols,
                                mmax=a.mmax,
                                depth=a.depth,
                                strided=a.strided,
                                down_split=a.down_split,
                                ok=ok,
                                pcc_min=pmin,
                                pcc=res,
                                perf=perf,
                            ),
                        )
                        if not ok:
                            fails += 1
                            for r in res:
                                if not r["ok"]:
                                    logger.error(
                                        f"   expert {r['expert']} count={r['count']} pcc={r['pcc']:.4f} nan={r['nan']}"
                                    )
                        del case
                    except Exception as e:  # noqa: BLE001
                        fails += 1
                        logger.error(f"ERROR {tag}: {type(e).__name__}: {str(e)[:400]}")
                        raise
        logger.info(f"done, failures={fails}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
