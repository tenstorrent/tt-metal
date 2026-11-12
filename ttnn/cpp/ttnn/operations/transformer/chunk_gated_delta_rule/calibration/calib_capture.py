# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One capture's worth of chunk_gated_delta_rule launches at a chosen geometry, for the device profiler.

  python -m tracy -r -p -o <dir> --op-support-count 4000 calib_capture.py --hv 16 --T 2048 --iters 5 \
      [--nv 2 --np 6 --rl 1 --nbuf 3]                       # per-head geometry, pinned
      [--pool --nv 2 --np 78 --share 0.346 --nbuf 2]        # producer pool, pinned (np = the pool size, share = num / P)
      [--phased]                                            # the two-phase program (prep + scan)
      (nothing)                                             # the op's own pick

Flat inputs (batch 1, T tokens, hk key heads, hv value heads, head size d), chunk size 32, the WY inverse pinned by
--method so the item time does not depend on the dispatch. Every launch is one ChunkGdnDeviceOperation (or a
ChunkGdnPrepOperation + ChunkGdnScanOperation pair with --phased); the calibration takes the median of launches 2..n.
"""

import argparse

import torch
import torch.nn.functional as F

import ttnn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="FORWARD_SUBSTITUTION", help="ChunkGdnWyInverse member")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--nv", type=int, default=0, help="num_receivers (0 = free)")
    ap.add_argument("--np", type=int, default=0, help="num_producers per head, or the pool size with --pool (0 = free)")
    ap.add_argument("--rl", type=int, default=-1, help="row_local 1/0; -1 = the model's choice")
    ap.add_argument("--nbuf", type=int, default=0, help="handoff_depth (0 = free)")
    ap.add_argument("--pool", action="store_true", help="producer_pool=True")
    ap.add_argument("--share", type=float, default=None, help="pool_extra_share (num / P)")
    ap.add_argument("--phased", action="store_true", help="the phased program instead of the fused one")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--hk", type=int, default=4)
    ap.add_argument("--hv", type=int, default=12)
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    try:
        torch.manual_seed(0)
        T, hk, hv, d = a.T, a.hk, a.hv, a.d
        q = torch.randn(1, T, hk * d).to(torch.bfloat16)
        k = torch.randn(1, T, hk * d).to(torch.bfloat16)
        v = (0.5 * torch.randn(1, T, hv * d)).to(torch.bfloat16)
        g = -F.softplus(torch.randn(1, T, hv)) * 0.5
        beta = torch.sigmoid(torch.randn(1, T, hv))
        s0 = torch.zeros(1, hv, d, d)

        def dt(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

        qt, kt, vt = dt(q, ttnn.bfloat16), dt(k, ttnn.bfloat16), dt(v, ttnn.bfloat16)
        gt, bt, st = dt(g, ttnn.float32), dt(beta, ttnn.float32), dt(s0, ttnn.float32)
        method = getattr(ttnn.ChunkGdnWyInverse, a.method)
        if a.phased:
            pc = ttnn.ChunkGdnPhasedProgramConfig()
        elif a.nv or a.np or a.pool or a.nbuf or a.rl >= 0:
            kw = {}
            if a.rl >= 0:
                kw["row_local"] = bool(a.rl)
            if a.nbuf:
                kw["handoff_depth"] = a.nbuf
            if a.pool:
                kw["producer_pool"] = True
            if a.share is not None:
                kw["pool_extra_share"] = a.share
            pc = ttnn.ChunkGdnFusedProgramConfig(num_receivers=a.nv or None, num_producers=a.np or None, **kw)
        else:
            pc = None
        for _ in range(a.iters):
            o, fs = ttnn.transformer.chunk_gated_delta_rule(
                qt,
                kt,
                vt,
                gt,
                bt,
                initial_state=st,
                output_final_state=True,
                chunk_size=32,
                wy_inverse=method,
                program_config=pc,
            )
            ttnn.deallocate(o)
            ttnn.deallocate(fs)
        ttnn.synchronize_device(dev)
        print("done", a.method, "phased" if a.phased else "fused")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
