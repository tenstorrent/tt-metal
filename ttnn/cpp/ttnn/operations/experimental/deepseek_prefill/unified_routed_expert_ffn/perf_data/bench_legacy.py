# Legacy (production) op baseline on the local P100: single-expert token sweep + multi-expert distributions.
import json
import os
import sys
import time

from loguru import logger

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_bench_common import (  # noqa: E402
    MODELS,
    MultiExpertCase,
    OP_KERNEL_DIR,
    RESULTS,
    append_jsonl,
    env_info,
    open_dev,
    timed,
)

DISTS = {
    "kimi": {
        "u107_e12": [107] * 12,
        "zipf_e12": [640, 320, 224, 160, 128, 96, 64, 64, 32, 32, 0, 0],
        "zeros_e12": [0, 320, 0, 160, 96, 0, 0, 64, 0, 0, 0, 32],
        "giant_e12": [2048] + [64] * 11,
        "prod5120_e12": [5120] + [107] * 11,
        "empty_e12": [0] * 12,
        "u107_e24": [107] * 24,
    },
    "m3": {
        "u160_e4": [160] * 4,
        "skew_e4": [384, 160, 64, 0],
        "u160_e8": [160] * 8,
        "skew_e8": [800, 400, 200, 100, 50, 25, 0, 5],
        "u160_e16": [160] * 16,
        "skew_e16": [800, 400, 200, 100, 50, 25, 0, 5, 160, 96, 64, 32, 0, 0, 0, 0],
        "giant_e4": [2048, 64, 32, 32],
    },
}
SINGLE_TOKENS = [0, 32, 64, 128, 256, 512, 1024, 2048, 5120]


def measure(dev, tag, model, counts, dtype_key, x_row_major, chunk_tokens, check=True, iters=3, **ffn):
    t0 = time.time()
    case = MultiExpertCase(dev, counts, model, dtype_key=dtype_key, x_row_major=x_row_major, **ffn)
    build_s = time.time() - t0
    res = timed(dev, case.run, OP_KERNEL_DIR, iters=iters, warmup=1)
    pcc = case.check() if check and any(c > 0 for c in counts) else []
    wb, fl = case.bytes_and_flops(chunk_tokens)
    rec = dict(
        section="legacy",
        tag=tag,
        model=model,
        emb=case.emb,
        hidden=case.hidden,
        act=case.act,
        dtype=dtype_key,
        layout="x_rm" if x_row_major else "x_tile",
        E=len(counts),
        counts=counts,
        tokens=sum(counts),
        ns=res["ns"],
        ns_all=res["ns_all"],
        ghz=res["ghz"],
        weight_bytes=wb,
        GBps=wb / res["ns"] if wb else 0.0,
        TFLOPs=fl / res["ns"] / 1e3 if fl else 0.0,
        chunk_tokens=chunk_tokens,
        pcc_min=min([r["pcc"] for r in pcc], default=None),
        pcc_ok=all(r["ok"] for r in pcc) if pcc else None,
        build_s=build_s,
        ffn=ffn,
    )
    append_jsonl("legacy.jsonl", rec)
    logger.info(
        f"{tag:28s} {model} {dtype_key} {rec['layout']:6s} E={len(counts):2d} tok={sum(counts):5d} "
        f"{res['ns']/1e3:8.1f} us  {rec['GBps']:6.1f} GB/s  {rec['TFLOPs']:6.1f} TF  pcc_min={rec['pcc_min']}"
    )
    del case
    return rec


def main():
    quick = "--quick" in sys.argv
    dev = open_dev()
    try:
        info = env_info(dev)
        logger.info(f"env: {info}")
        chunk = 2048  # legacy: per_core_M_max 8 x 8 rows x 32 tokens (bf8 may shrink; factory log_info tells)
        dtypes = ["bf4"] if quick else ["bf4", "bf8"]
        for model in ["kimi", "m3"]:
            for dtype_key in dtypes:
                for t in [128, 5120] if quick else SINGLE_TOKENS:
                    measure(dev, f"single_t{t}", model, [t], dtype_key, True, chunk, check=(t > 0))
                if dtype_key == "bf4" and not quick:
                    for t in [128, 512, 5120]:
                        measure(dev, f"single_t{t}", model, [t], dtype_key, False, chunk, check=True)
                for tag, counts in DISTS[model].items():
                    if quick and tag not in ("u107_e12", "u160_e4"):
                        continue
                    measure(dev, tag, model, counts, dtype_key, True, chunk, check=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
