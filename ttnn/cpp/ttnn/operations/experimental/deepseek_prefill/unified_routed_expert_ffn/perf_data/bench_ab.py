# A/B sweep: legacy (G=0) vs grouped configs, interleaved in one process, realtime profiler timing.
import argparse
import itertools
import os
import sys
import time

from loguru import logger

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import (
    MODELS,
    MultiExpertCase,
    OP_KERNEL_DIR,
    append_jsonl,
    env_info,
    open_dev,
    timed,
)  # noqa: E402
from test_grouped import DISTS  # noqa: E402


def cfg_tag(c):
    if c["G"] == 0:
        return "legacy"
    return f"G{c['G']}r{c['rows']}c{c['cols']}m{c['mmax']}d{c['depth']}s{c['strided']}ds{c['down_split']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", default="kimi_u,kimi_zipf,kimi_zeros,kimi_giant,m3_u4,m3_u8,m3_u16,m3_skew8,m3_giant4")
    ap.add_argument("--dtypes", default="bf4,bf8")
    ap.add_argument("--layout", default="rm")
    ap.add_argument(
        "--configs",
        default="legacy;G10r10;G5r10;G8r8;G4r8;G2r8;G1r8",
        help="';'-separated: legacy | G<g>r<rows>[c<cols>][m<mmax>][d<depth>][s<strided>][ds<down_split>]",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    def parse(cs):
        if cs == "legacy":
            return dict(G=0, rows=0, cols=0, mmax=0, depth=0, strided=0, down_split=1)
        import re

        d = dict(G=0, rows=0, cols=0, mmax=0, depth=0, strided=0, down_split=1)
        for key, val in re.findall(r"(ds|G|r|c|m|d|s)(\d+)", cs):
            d[
                {"G": "G", "r": "rows", "c": "cols", "m": "mmax", "d": "depth", "s": "strided", "ds": "down_split"}[key]
            ] = int(val)
        return d

    configs = [parse(c) for c in a.configs.split(";")]
    dev = open_dev()
    try:
        info = env_info(dev)
        logger.info(f"env: {info}")
        for dist in a.dists.split(","):
            model, counts = DISTS[dist]
            for dtype in a.dtypes.split(","):
                results = {}
                # Build the case (weights, buffer, TtRoutedExpert) ONCE per (dist, dtype); the grouped knobs
                # are read at call time from expert.ffn_kwargs, so configs only swap that dict.
                case = MultiExpertCase(dev, counts, model, dtype_key=dtype, x_row_major=(a.layout == "rm"))
                for c in configs:
                    ffn = dict(
                        num_row_groups=0,
                        grid_rows=0,
                        grid_cols=0,
                        per_core_m_max=0,
                        weight_cb_depth=0,
                        col_strided=0,
                        down_split=1,
                        lpt_fixed_cost_tiles=0,
                    )
                    if c["G"] > 0:
                        ffn.update(
                            num_row_groups=c["G"],
                            grid_rows=c["rows"],
                            grid_cols=c["cols"],
                            per_core_m_max=c["mmax"],
                            weight_cb_depth=c["depth"],
                            col_strided=c["strided"],
                            down_split=c["down_split"],
                        )
                    case.expert.ffn_kwargs = ffn
                    tag = cfg_tag(c)
                    try:
                        perf = timed(dev, case.run, OP_KERNEL_DIR, iters=a.iters, warmup=2)
                        pcc = case.check() if a.check else []
                        rows = c["rows"] or 8
                        chunk_tok = 2048 if c["G"] == 0 else 32 * (c["mmax"] or 4) * (rows // c["G"])
                        wb, fl = case.bytes_and_flops(chunk_tok)
                        rec = dict(
                            section="ab",
                            dist=dist,
                            model=model,
                            counts=counts,
                            dtype=dtype,
                            layout=a.layout,
                            cfg=tag,
                            **c,
                            ns=perf["ns"],
                            ns_all=perf["ns_all"],
                            ghz=perf["ghz"],
                            weight_bytes=wb,
                            GBps=wb / perf["ns"],
                            TFLOPs=fl / perf["ns"] / 1e3,
                            pcc_min=min([r["pcc"] for r in pcc], default=None),
                            pcc_ok=all(r["ok"] for r in pcc) if pcc else None,
                        )
                        results[tag] = rec
                        base = results.get("legacy")
                        sp = f"{base['ns']/rec['ns']:.2f}x" if base else "-"
                        logger.info(
                            f"{dist:14s} {dtype} {tag:22s} {rec['ns']/1e3:9.1f} us {rec['GBps']:6.1f} GB/s {rec['TFLOPs']:6.1f} TF "
                            f"speedup={sp} pcc_min={rec['pcc_min']}"
                        )
                        rec["speedup_vs_legacy"] = base["ns"] / rec["ns"] if base else None
                        append_jsonl("ab.jsonl", rec)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"{dist} {dtype} {tag}: {type(e).__name__}: {str(e)[:300]}")
                        append_jsonl(
                            "ab.jsonl",
                            dict(section="ab", dist=dist, model=model, dtype=dtype, cfg=tag, **c, error=str(e)[:300]),
                        )
                del case
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
