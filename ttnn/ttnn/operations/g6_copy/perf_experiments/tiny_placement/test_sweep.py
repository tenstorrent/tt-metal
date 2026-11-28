"""tiny_placement sweeps: many plans in one profiled process, each launched TP_LAUNCHES times on the
same tensors (device sync between launches, like the case bench), bit-exact check once per plan.
Prints 'TP_CFG <idx> <launches> <label>' in dispatch order; analyze.py maps the ops CSV rows onto it.

TP_SWEEP picks the sweep:
  place1  1-page bf16 TILE (1,1): every core with reader NoC0 / writer NoC1, plus NoC flips on a subset.
  count   N pages x C cores (bank-major, placed vs row-wise) for page sizes of the small cells.
"""
import os
import sys
from pathlib import Path

import pytest

torch = __import__("torch")  # experiment script; ttnn bans a global torch import
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import tp_plan  # noqa: E402

N0, N1 = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1
LAUNCHES = int(os.environ.get("TP_LAUNCHES", "5"))


def _tensors(device, shape, dtype, layout):
    torch.manual_seed(0)
    if dtype in (ttnn.uint32, ttnn.int32):
        src = torch.randint(0, 2**30, shape, dtype=torch.int32)
    else:
        src = torch.randn(shape, dtype=torch.float32)
    inp = ttnn.from_torch(src, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.from_torch(
        torch.zeros_like(src), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref = ttnn.to_torch(inp)
    return inp, out, ref


def _cc(x, y):
    return ttnn.CoreCoord(x, y)


def _row_wise_cores(n, grid):
    return [_cc(i % grid.x, i // grid.x) for i in range(n)]


def _split(n, c):
    q, r = divmod(n, c)
    out, f = [], 0
    for i in range(c):
        k = q + (1 if i < r else 0)
        out.append((f, k))
        f += k
    return out


def _configs_place1(device, grid):
    cfgs = []
    for y in range(grid.y):
        for x in range(grid.x):
            cfgs.append((f"core{x},{y} r0w1", [(_cc(x, y), 0, 1, 0)], False, N0, N1))
    for x in (0, 6, 7, 10):
        for y in range(grid.y):
            for rn, wn in ((N1, N0), (N0, N0), (N1, N1)):
                tag = f"r{int(rn == N1)}w{int(wn == N1)}"
                cfgs.append((f"core{x},{y} {tag}", [(_cc(x, y), 0, 1, 0)], False, rn, wn))
    return cfgs


def _placed_bank_major(n, c, grid, opt0, opt1, which):
    """c cores over the bank-major enumeration; core i takes a contiguous j range. Core placement:
    'row' = row-wise from (0,0); 'opt0' = the NoC0-optimal core of the range's first bank (next free
    core below it when taken); 'opt1' likewise for NoC1."""
    ranges = _split(n, c)
    banks = tp_plan.bank_major_ranges(n)
    bank_of_j = []
    for b, j0, k in banks:
        bank_of_j += [b] * k
    if which == "row":
        cores = _row_wise_cores(c, grid)
    else:
        opt = opt0 if which == "opt0" else opt1
        used, cores = set(), []
        for f, k in ranges:
            x, y = opt[bank_of_j[f]]
            # walk the column, then neighbouring columns, to the first unused core
            cand = [(x, (y + d) % grid.y) for d in range(grid.y)]
            cand += [((x + dx) % grid.x, (y + d) % grid.y) for dx in (1, -1, 2, -2, 3, -3) for d in range(grid.y)]
            for xy in cand:
                if xy not in used:
                    used.add(xy)
                    cores.append(_cc(*xy))
                    break
    return [(cores[i], f, k, i) for i, (f, k) in enumerate(ranges)]


def _configs_count(device, grid, n, label):
    opt0, opt1 = tp_plan.optimal_cores(device)
    rows0 = [c[1] for c in opt0]
    cfgs = []
    c_op = min(n, grid.x * grid.y)
    nat = [(core, f, k, i) for i, ((f, k), core) in enumerate(zip(_split(n, c_op), _row_wise_cores(c_op, grid)))]
    cfgs.append(("op r0w1", nat, False, N0, N1))
    cfgs.append(("op r0w0", nat, False, N0, N0))
    nb = -(-n // 8)
    ks = sorted({k for k in (1, 2, 3, 4, 5, 6, 8, 10, 13) if k <= nb} | ({min(nb, 13)}))
    if n > 1000:
        ks = [13]
    for k in ks:
        w = tp_plan.placed_work(n, grid, rows0, k)
        cfgs.append((f"placed k{k} c{len(w)} r0w0", w, True, N0, N0))
        if k in (ks[0], ks[-1]):
            cfgs.append((f"placed k{k} c{len(w)} r0w1", w, True, N0, N1))
    if n <= 64:
        cfgs.append(("single c1 row-bank0 r0w0", [(_cc(0, rows0[0]), 0, n, 0)], True, N0, N0))
    return cfgs


def _configs_cross(device, grid, n):
    opt0, _ = tp_plan.optimal_cores(device)
    rows0 = [c[1] for c in opt0]
    c_op = min(n, grid.x * grid.y)
    nat = [(core, f, k, i) for i, ((f, k), core) in enumerate(zip(_split(n, c_op), _row_wise_cores(c_op, grid)))]
    cfgs = [("op r0w1", nat, False, N0, N1)]
    for k in (5, 8, 11, 13) if n < 1000 else (11, 13):
        w = tp_plan.placed_work(n, grid, rows0, min(k, -(-n // 8)))
        cfgs.append((f"placed k{k} c{len(w)} r0w0", w, True, N0, N0))
    w = tp_plan.placed_work(n, grid, rows0, 13)
    cfgs.append((f"placed k13 c{len(w)} r0w1", w, True, N0, N1))
    return cfgs


def _configs_modes(device, grid, n):
    opt0, _ = tp_plan.optimal_cores(device)
    rows0 = [c[1] for c in opt0]
    c_op = min(n, grid.x * grid.y)
    nat = [(core, f, k, i) for i, ((f, k), core) in enumerate(zip(_split(n, c_op), _row_wise_cores(c_op, grid)))]
    cfgs = [("op r0w1", nat, False, N0, N1)]
    k = min(grid.x, -(-n // 8))
    for mode in ("fill", "balanced", "aligned"):
        w = tp_plan.placed_work_v(n, grid, rows0, k, mode)
        cfgs.append((f"placed {mode} k{k} c{len(w)} r0w0", w, True, N0, N0))
    return cfgs


CASES = {
    # sweep -> list of (case label, shape, dtype, layout)
    "place1": [("bf16_1x1", (1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)],
    "cross": [
        ("p6_bf16_1x2048x128", (1, 2048, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 256 x 2048 B
        ("p8_bf16_1x1x2304x128", (1, 1, 2304, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 288 x 2048 B
        ("p3_f32_1x12x128x128", (1, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),  # 192 x 4096 B
        ("p12_bf16_1x256x3072", (1, 256, 3072), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 768 x 2048 B
        ("p2_f32_2x12x128x128", (2, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),  # 384 x 4096 B
    ],
    "cross2": [
        ("p9_bf16_1x8192x128", (1, 8192, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 1024 x 2048 B (2 MiB)
        ("p5_bf16_1x1x32x62080", (1, 1, 32, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 1940 x 2048 B (3.9 MiB)
        ("p1_f32_8x12x128x128", (8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),  # 1536 x 4096 B (6 MiB)
        ("p0_f32_16x12x128x128", (16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),  # 3072 x 4096 B (12 MiB)
    ],
    "modes": [
        ("p15_bf8b_5x1x32x256", (5, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        ("p19_bf16rm_16x3x2560", (16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("p6_bf16_1x2048x128", (1, 2048, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ("p2_f32_2x12x128x128", (2, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
        ("p0_f32_16x12x128x128", (16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
    ],
    "count": [
        ("p16_bf16rm_1x3x2560", (1, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),  # 3 x 5120 B
        ("p13_bf16_1x1x1x256", (1, 1, 1, 256), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 8 x 2048 B
        ("p18_bf16rm_8x3x2560", (8, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),  # 24 x 5120 B
        ("p15_bf8b_5x1x32x256", (5, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT),  # 40 x 1088 B
        ("p19_bf16rm_16x3x2560", (16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),  # 48 x 5120 B
        ("p6_bf16_1x2048x128", (1, 2048, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),  # 256 x 2048 B
        ("p0_f32_16x12x128x128", (16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),  # 3072 x 4096 B
    ],
}


def test_sweep(device):
    sweep = os.environ.get("TP_SWEEP", "place1")
    only = os.environ.get("TP_ONLY")
    grid = device.compute_with_storage_grid_size()
    idx = 0
    for label, shape, dtype, layout in CASES[sweep]:
        if only and label not in only.split(","):
            continue
        inp, out, ref = _tensors(device, shape, dtype, layout)
        n, pb = inp.buffer_num_pages(), inp.buffer_aligned_page_size()
        if sweep == "place1":
            cfgs = _configs_place1(device, grid)
        elif sweep == "modes":
            cfgs = _configs_modes(device, grid, n)
        elif sweep in ("cross", "cross2"):
            cfgs = _configs_cross(device, grid, n)
        else:
            cfgs = _configs_count(device, grid, n, label)
        for name, work, bank_major, rn, wn in cfgs:
            plan = tp_plan.make_plan(work, n, pb, grid)
            desc = tp_plan.build(inp, out, plan, bank_major, rn, wn)
            for _ in range(LAUNCHES):
                ttnn.generic_op([inp, out], desc)
                ttnn.synchronize_device(device)
            got = ttnn.to_torch(out)
            ok = torch.equal(
                got.view(torch.int32) if got.dtype == torch.float32 else got,
                ref.view(torch.int32) if ref.dtype == torch.float32 else ref,
            )
            print(f"TP_CFG {idx} {LAUNCHES} {label} | {name} | pages={n} pb={pb} plan={plan.plan} ok={ok}", flush=True)
            assert ok, f"{label} {name} mismatch"
            # fresh output for the next config so a skipped page cannot pass on stale data
            ttnn.deallocate(out)
            out = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.float32 if dtype not in (ttnn.uint32, ttnn.int32) else torch.int32),
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.ReadDeviceProfiler(device)
            idx += 1
