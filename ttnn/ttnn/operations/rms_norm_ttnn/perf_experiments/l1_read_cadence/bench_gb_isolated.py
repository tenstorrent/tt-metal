# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# RMSN_GB L1->L1 in ISOLATION -- one op, one dispatch, fresh process.
#
# This is the instrument that decided NOT to cap the per-channel read's barrier
# cadence.  bench_family.py cannot settle it: its multi-case loop leaves the L1
# allocator in a different state for each case, and the ~2.9 us effect here is
# smaller than that confound.  Run it several times and compare medians.
#
#   BENCH_OP=native BENCH_TAG=<label> scripts/tt-probe.sh rms_norm_ttnn < bench_gb_isolated.py
import hashlib
import os

import ttnn

N, C, H, W = 1, 9, 384, 1024
EPS = 1e-2


def main():
    import torch
    from ttnn.operations import normalization as nrm

    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)
    L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    x = torch.rand((N, C, H, W)) * 2 - 0.95
    gamma = torch.rand(1, 1, 1, W) * 2 - 1
    beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1

    op = nrm._native_rms_norm if os.environ.get("BENCH_OP") == "native" else ttnn.rms_norm
    tag = os.environ.get("BENCH_TAG", "run")

    # gamma and bias BEFORE x, as test_rmsnorm.py allocates them.
    tg = ttnn.from_torch(
        gamma.expand(1, 1, 32, W).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=L1,
    )
    tb = ttnn.from_torch(
        beta.expand(1, 1, 32, W).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=L1,
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    ttnn.ReadDeviceProfiler(dev)
    out = op(tx, epsilon=EPS, weight=tg, bias=tb, memory_config=L1)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    per = ttnn.get_latest_programs_perf_data()
    ns = sum(
        float(pr.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration)
        for v in (per or {}).values()
        for pr in v
        if "DEVICE KERNEL DURATION [ns]" in (pr.program_analyses_results or {})
    )

    t = ttnn.to_torch(out).to(torch.float32)
    ref = x.to(torch.float32)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + EPS) * gamma.to(torch.float32) + beta.to(torch.float32)
    pcc = float(torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0, 1])
    sha = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
    print(f"GBRESULT {tag:14s} {ns:9.0f} ns  PCC {pcc:.6f}  sha {sha}", flush=True)
    ttnn.close_device(dev)
    print("PROBE_OK", flush=True)


main()
