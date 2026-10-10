# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Flat routed expert bench: one chip, fake weights, device time per call.

For quick perf iteration (e.g. on a reserved Galaxy). Weights are allocated in the flat expert's bank layout without
data (FlatRoutedExpert(weights="fake")): the bfp matmuls take the same time whatever the bytes are, so there is no host
packing or upload, and the numbers equal real-weight runs. Outputs are garbage; use the accuracy tests for numerics.

Timing: the program real-time profiler (device start / end of each call, median of ITERS calls) when it is active;
otherwise traced replay (wall time per back-to-back replay; the device's throughput). The real-time profiler is off
with the fabric Tensix mux (row dispatch), on hosts without IOMMU that need it, and with the streaming profiler.

Per case: total ms, us / expert, DRAM % of 512 GB/s (weights + x reads + y writes), math % of the LoFi peak over the
plan's compute cores (gate/up + down) and over the whole grid.

Knobs (env):
  FLAT_BENCH_MODEL   glm53flash (default: 4096 x 2048, 36 experts / chip, clamped_silu, indexed x as in the model),
                     kimi27 (7168 x 2048, 12, silu), glm53 (6144 x 2048, 8, silu), k3 (3584 x 3072, 12, situ)
  FLAT_BENCH_H / _I / _E / _NG / _ACT   override the preset's hidden / intermediate / local experts / global experts /
                     activation (flat_expert.ACTS)
  FLAT_BENCH_M       comma list of per-expert token counts for balanced routing (default 32 .. 5120)
  FLAT_BENCH_HOT     "M:factor[:hot_count]": hot-expert routing, every expert M tokens except hot_count (default 1)
                     experts with M * factor (replaces FLAT_BENCH_M)
  FLAT_BENCH_COUNTS / FLAT_BENCH_COUNTS_KEY   json {key: {name: [count per local expert]}} (replaces both)
  FLAT_BENCH_CAP     capacity (max tokens per expert the program is built for, default 8192)
  FLAT_BENCH_PIN     schedule pinning (default 1; the runtime pin rule SE_PIN_RATIO decides per call)
  FLAT_BENCH_YRM     1 (default): y row-major bf16; 0: bfp8 tiles
  FLAT_BENCH_INDEXED 1: x is the T gathered tokens, flat rows read x[token_index] (GLM-Flash's all-gather MoE path);
                     0: x is a dispatch buffer (the experts' rows back to back, as with dispatch / combine).
                     Default: the preset's.
  FLAT_BENCH_REAL    1: real (random) weights instead of fake ones
  FLAT_BENCH_ITERS   timed calls per case (default 10)
  flat planner probes (MIMO_FL_ROWS, MIMO_FL_XDOWN, MIMO_FL_RD_SAMECOL, ...) apply as usual.

Run:  scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/test_flat_bench.py -s
      (on a Galaxy, one chip: TT_VISIBLE_DEVICES=0 ...; a 4-case sweep takes about a minute)
"""

import json
import os
import statistics
import time

import pytest
import torch

import ttnn

PRESETS = {
    # name: (H, I, local experts, global experts, activation, indexed x)
    "glm53flash": (4096, 2048, 36, 288, "clamped_silu", True),
    "kimi27": (7168, 2048, 12, 384, "silu", False),
    "glm53": (6144, 2048, 8, 256, "silu", False),
    "k3": (3584, 3072, 12, 384, "situ", False),
}
MODEL = os.environ.get("FLAT_BENCH_MODEL", "glm53flash")
_H, _I, _E, _NG, _ACT, _IDX = PRESETS[MODEL]
H = int(os.environ.get("FLAT_BENCH_H", _H))
I = int(os.environ.get("FLAT_BENCH_I", _I))
E = int(os.environ.get("FLAT_BENCH_E", _E))
NG = max(int(os.environ.get("FLAT_BENCH_NG", _NG)), E)
ACT = os.environ.get("FLAT_BENCH_ACT", _ACT)
INDEXED = os.environ.get("FLAT_BENCH_INDEXED", "1" if _IDX else "0") == "1"
CAP = int(os.environ.get("FLAT_BENCH_CAP", "8192"))
PIN = int(os.environ.get("FLAT_BENCH_PIN", "1"))
YRM = os.environ.get("FLAT_BENCH_YRM", "1") == "1"
REAL = os.environ.get("FLAT_BENCH_REAL") == "1"
ITERS = int(os.environ.get("FLAT_BENCH_ITERS", "10"))
T = 5120  # gathered tokens the indexed mode reads from (GLM-Flash's chunk)
DRAM_GBS, FLOP_CYC, BFP4_TILE = 512.0, 4096, 576


def _cases():
    if os.environ.get("FLAT_BENCH_COUNTS"):
        pats = json.load(open(os.environ["FLAT_BENCH_COUNTS"]))[os.environ["FLAT_BENCH_COUNTS_KEY"]]
        return [(name, list(cl)) for name, cl in pats.items()]
    if os.environ.get("FLAT_BENCH_HOT"):
        parts = [int(v) for v in os.environ["FLAT_BENCH_HOT"].split(":")]
        m, factor, n_hot = parts[0], parts[1], (parts[2] if len(parts) > 2 else 1)
        return [(f"hot {m}x{factor}", [m * factor] * n_hot + [m] * (E - n_hot))]
    ms = os.environ.get("FLAT_BENCH_M")
    ms = [int(v) for v in ms.split(",")] if ms else [32, 128, 256, 512, 1024, 2048, 5120]
    return [(m, [m] * E) for m in ms]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
def test_flat_bench(device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    torch.manual_seed(0)
    if REAL:
        weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    else:
        weights = "fake"
    t_build = time.time()
    op = FlatRoutedExpert(
        device, weights, m=CAP, H=H, I=I, gids=[list(range(E))], n_global=NG, wdtype="bf4", act=ACT, pin=PIN
    )
    del weights
    lay = dict(op.plan)
    grid = device.compute_with_storage_grid_size()
    n_compute = len(lay["gu"]) + lay["nd"]
    n_grid = grid.x * grid.y
    print(
        f"[bench] {MODEL} H {H} I {I} E {E} act {ACT} {'indexed' if INDEXED else 'dispatch-buffer'} x, "
        f"{'real' if REAL else 'fake'} weights (built in {time.time() - t_build:.1f} s); plan np {lay['np']} "
        f"gu {len(lay['gu'])} down {lay['nd']} readers {lay['n_rd']} rdown {lay['rdown']}; grid {grid.x} x {grid.y}",
        flush=True,
    )
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w_bytes = E * 3 * (H // 32) * (I // 32) * BFP4_TILE

    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    recs = {}

    def cb(batch):
        for r in batch.records:
            recs[r.runtime_id] = (r.end_timestamp - r.start_timestamp) / (r.frequency * 1e3), r.frequency  # us, GHz

    hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb) if rt else None
    print(f"[bench] timing: {'program real-time profiler' if rt else 'traced replay'}", flush=True)
    rows_out = []
    try:
        for name, cl in _cases():
            assert len(cl) == E and max(cl) <= CAP, f"case {name}: {len(cl)} counts (E {E}), max {max(cl)} (CAP {CAP})"
            pads = [(c + 31) // 32 * 32 for c in cl]
            counts = torch.zeros(1, NG, dtype=torch.int32)
            regions = torch.zeros(1, NG, dtype=torch.int32)
            counts[0, :E] = torch.tensor(cl, dtype=torch.int32)
            regions[0, :E] = torch.tensor([sum(pads[:e]) for e in range(E)], dtype=torch.int32)
            n_rows = max(sum(pads), 32)
            cd, rd = rm(counts, ttnn.uint32), rm(regions, ttnn.uint32)
            if INDEXED:
                x = rm(torch.randn(T, H).to(torch.bfloat16) * 0.3, ttnn.bfloat16)
                tidx = rm(torch.randint(0, T, (1, n_rows), dtype=torch.int32), ttnn.uint32)
                call = lambda: op(x, cd, rd, token_index=tidx, y_row_major=YRM)  # noqa: E731
            else:
                x = rm(torch.randn(n_rows, H).to(torch.bfloat16) * 0.3, ttnn.bfloat16)
                tidx = None
                call = lambda: op(x, cd, rd, y_row_major=YRM)  # noqa: E731
            ttnn.deallocate(call())  # compile + warm-up
            ttnn.synchronize_device(device)
            ghz = 1.35
            if rt:
                ids = []
                for _ in range(ITERS):
                    i0 = ttnn._ttnn.get_device_operation_id()
                    ttnn.deallocate(call())
                    ids.append((i0, ttnn._ttnn.get_device_operation_id()))
                ttnn.synchronize_device(device)
                time.sleep(0.5)  # let the receiver thread deliver the last records
                per = [sum(recs[i][0] for i in range(a, b) if i in recs) for a, b in ids]
                hit = [i for a, b in ids for i in range(a, b) if i in recs]
                assert hit, "real-time profiler active but no records arrived"
                ms, ghz = statistics.median(per) / 1e3, recs[hit[0]][1]
            else:
                tr = ttnn.begin_trace_capture(device, cq_id=0)
                y = call()
                ttnn.end_trace_capture(device, tr, cq_id=0)
                ttnn.execute_trace(device, tr, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    ttnn.execute_trace(device, tr, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                ms = (time.perf_counter() - t0) / ITERS * 1e3
                ttnn.release_trace(device, tr)
                ttnn.deallocate(y)
            rows = sum(cl)
            y_b = 2 if YRM else 1088 / 1024
            dram = (w_bytes + rows * H * (2 + y_b)) / (ms * 1e-3) / 1e9 / DRAM_GBS
            flops = rows * 6 * H * I
            peak_core = FLOP_CYC * ghz * 1e9
            mc = flops / (ms * 1e-3) / (peak_core * n_compute)
            mg = flops / (ms * 1e-3) / (peak_core * n_grid)
            rows_out.append((name, rows, ms, ms / E * 1e3, dram, mc, mg))
            print(
                f"[bench] {name!s:>10} rows {rows:6d}: {ms:8.3f} ms {ms / E * 1e3:7.1f} us/expert DRAM {dram:5.1%} "
                f"math {mc:5.1%} ({n_compute} compute) {mg:5.1%} ({n_grid} grid) @ {ghz:.3f} GHz",
                flush=True,
            )
            for t_ in (cd, rd, x) + ((tidx,) if tidx is not None else ()):
                ttnn.deallocate(t_)
    finally:
        if hdl is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
    print(f"[bench] | case | M_total | total ms | us/expert | DRAM % | math % ({n_compute} compute) | math % ({n_grid} grid) |")
    for name, rows, ms, pe, dram, mc, mg in rows_out:
        print(f"[bench] | {name} | {rows} | {ms:.3f} | {pe:.1f} | {dram:.0%} | {mc:.0%} | {mg:.0%} |")
