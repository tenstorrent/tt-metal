# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_pre — ProgramDescriptor (regime R1 `group_ksplit_resident`, see op_design.md Blocking Model).

Token tile-rows are split over *groups* (rectangles of `group_w x group_h` cores); inside a group the
n*C reduction axis is split by stream-column slice over the group's *ranks*. Each rank keeps its X block
resident from the projection to the y-mix (X crosses DRAM once), sends its (mix, sum x^2) partial pair to
the group root (push gather + monotonic semaphore), the root folds the partials in rank order and
multicasts the combined pair back (mcast_pipe), every rank computes the coefficients itself and mixes its
own y columns, and the Sinkhorn + post/comb output of each token row is owned round-robin by one rank.

Every block knob is derived here ONCE and handed to the kernels as a CT/RT arg:
    group geometry  -> group_w, group_h, group_cores, num_groups (core-assignment knobs)
    block_token_tiles, x_block_depth (L1 selection function)
    y_chunk_tiles, y_depth           (writer window knobs)
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
F32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# ---- CB indices (semantic names; the slot numbers are arbitrary) ----
CB_X_RESIDENT = 0
CB_WEIGHT = 1
CB_BIAS_COEF = 2
CB_REDUCE_SCALER = 3
CB_SQ_ACC = 4
CB_PARTIAL = 5
CB_GATHERED = 6
CB_COMBINED = 7
CB_COEF_IN = 8
CB_COEF_OUT = 9
CB_LOGITS_COEF = 10
CB_COMB_COEF = 11
CB_PRE_COLS = 12
CB_Y_OUT = 13
CB_OUT_STAGE = 14
NUM_CB_SLOTS = 64
UNPACK_TO_DEST_FP32_CBS = (CB_BIAS_COEF, CB_SQ_ACC, CB_GATHERED, CB_COEF_IN, CB_LOGITS_COEF)

# ---- semaphores ----
SEM_GATHER = 0  # monotonic partial-arrival counter on the group root
SEM_MCAST_READY = 1  # mcast_pipe data-ready flag
SEM_MCAST_CONSUMED = 2  # mcast_pipe consumer-ready (pre-handshake) counter

# ---- host constants (tunable knobs, single source) ----
GROUP_CORES_CAP = 32  # flat-root gather cap (cb_gathered grows with group_cores)
X_BLOCK_DEPTH_DEFAULT = 2  # prefetch depth of cb_x_resident (stall shadow of the combine round trip)
Y_DEPTH = 2  # cb_y_out windows in flight
Y_CHUNK_TILES_CAP = 8  # 4-8 writes in flight per barrier saturate (catalog: double_buffer)
OUT_STAGE_PAGES = 2  # one post + one comb staging tile
L1_SAFETY_MARGIN = 64 * 1024  # headroom below the allocator's unreserved L1 (kernel config, stack)
# Upper bound on block_token_tiles (the selection function takes min(this, core share, L1 fit)).
# Measured on BH p150 (fp32, device kernel ns, bt=coarsest-fit -> bt=1): 640x7168 384->383 us,
# 640x1792 133->128, 1280x4096 420->381, 4096x1792 591->521 (bt 2/4/7 in between). The whole K slice
# is still one block; finer token blocks let the X read of block b+1 overlap block b's combine round
# trip, y-mix and y stores (design perf lamp L1). Raise it to trade that overlap for fewer per-block
# fixed costs.
BLOCK_TOKEN_TILES_CAP = 1


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _split(total, parts):
    """ceil/floor split: the first `total % parts` parts get the ceil."""
    base, rem = divmod(total, parts)
    sizes = [base + (1 if p < rem else 0) for p in range(parts)]
    starts = [0] * parts
    for p in range(1, parts):
        starts[p] = starts[p - 1] + sizes[p - 1]
    return sizes, starts


@dataclass
class Plan:
    n: int
    Mt: int
    Ct: int
    Kt: int
    grid_x: int
    grid_y: int
    group_w: int
    group_h: int
    group_cores: int
    groups_x: int
    groups_y: int
    num_groups: int
    core_c_tiles: list
    c_start: list
    core_token_tiles: list  # per group
    t_start: list  # per group
    core_k_tiles_max: int
    block_token_tiles: int
    x_block_depth: int
    y_chunk_tiles: int
    y_depth: int


def _l1_bytes(plan_like, bt, depth, x_tile, w_tile, y_tile, n):
    """Per-core L1 footprint (identical on every core; see l1_ledger.md 'Total per-core footprint')."""
    kmax, G, y_chunk = plan_like
    per_token = depth * kmax * x_tile + F32_TILE_BYTES * (2 + 2 * G + 2 + 1 + 1 + 1 + 1 + n)
    fixed = kmax * w_tile + Y_DEPTH * y_chunk * y_tile + F32_TILE_BYTES * (1 + 1 + OUT_STAGE_PAGES) + BF16_TILE_BYTES
    return bt * per_token + fixed


def make_plan(device, x_tensor, w_tensor, n):
    padded = list(x_tensor.padded_shape)
    lead_tiles = 1
    for d in padded[:-2]:
        lead_tiles *= d
    Mt = lead_tiles * (padded[-2] // TILE)  # per-image tile padding is already in the padded shape
    C = x_tensor.shape[-1] // n
    Ct = C // TILE
    Kt = n * Ct

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y

    x_tile = x_tensor.buffer_page_size()
    w_tile = w_tensor.buffer_page_size()
    y_tile = x_tile
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_SAFETY_MARGIN

    group_w = min(grid_x, Ct)
    if Mt >= grid_y:
        group_h = 1
    else:
        group_h = max(1, min(grid_y // Mt, Ct // group_w, GROUP_CORES_CAP // group_w))

    while True:
        group_cores = group_w * group_h
        kmax = n * math.ceil(Ct / group_cores)
        y_chunk = min(math.ceil(Ct / group_cores), Y_CHUNK_TILES_CAP)
        groups_x, groups_y = grid_x // group_w, grid_y // group_h
        num_groups = groups_x * groups_y
        core_token_tiles, t_start = _split(Mt, num_groups)
        ctt_max = max(core_token_tiles)

        depth = X_BLOCK_DEPTH_DEFAULT
        knobs = (kmax, group_cores, y_chunk)
        per_bt = _l1_bytes(knobs, 1, depth, x_tile, w_tile, y_tile, n) - _l1_bytes(
            knobs, 0, depth, x_tile, w_tile, y_tile, n
        )
        fixed = _l1_bytes(knobs, 0, depth, x_tile, w_tile, y_tile, n)
        bt_cap = min(ctt_max, BLOCK_TOKEN_TILES_CAP)
        bt = min(bt_cap, (budget - fixed) // per_bt) if budget > fixed else 0
        if bt < 1:
            depth = 1
            per_bt = _l1_bytes(knobs, 1, depth, x_tile, w_tile, y_tile, n) - fixed
            bt = min(bt_cap, (budget - fixed) // per_bt) if budget > fixed else 0
        if bt >= 1:
            break
        # Still does not fit: grow the group (smaller per-rank K slice).
        if group_h * 2 <= grid_y and group_w * group_h * 2 <= GROUP_CORES_CAP and group_w * group_h * 2 <= Ct:
            group_h *= 2
            continue
        raise RuntimeError(f"mhc_pre: no blocking fits L1 (C={C}, Mt={Mt}, grid={grid_x}x{grid_y}, budget={budget} B)")

    core_c_tiles, c_start = _split(Ct, group_cores)
    return Plan(
        n=n,
        Mt=Mt,
        Ct=Ct,
        Kt=Kt,
        grid_x=grid_x,
        grid_y=grid_y,
        group_w=group_w,
        group_h=group_h,
        group_cores=group_cores,
        groups_x=groups_x,
        groups_y=groups_y,
        num_groups=num_groups,
        core_c_tiles=core_c_tiles,
        c_start=c_start,
        core_token_tiles=core_token_tiles,
        t_start=t_start,
        core_k_tiles_max=kmax,
        block_token_tiles=bt,
        x_block_depth=depth,
        y_chunk_tiles=y_chunk,
        y_depth=Y_DEPTH,
    )


def _cb(index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def create_program_descriptor(
    x_tensor, w_tensor, b_tensor, y_tensor, post_tensor, comb_tensor, *, n, scale, sinkhorn_iters, eps, norm_eps, cfg
):
    device = x_tensor.device()
    plan = make_plan(device, x_tensor, w_tensor, n)
    G = plan.group_cores
    bt = plan.block_token_tiles
    kmax = plan.core_k_tiles_max

    # ---- active groups (a group with 0 token rows is not launched) ----
    groups = []
    for g in range(plan.num_groups):
        if plan.core_token_tiles[g] == 0:
            continue
        gx0 = (g % plan.groups_x) * plan.group_w
        gy0 = (g // plan.groups_x) * plan.group_h
        groups.append((g, gx0, gy0))
    ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(gx0, gy0), ttnn.CoreCoord(gx0 + plan.group_w - 1, gy0 + plan.group_h - 1))
        for _, gx0, gy0 in groups
    ]
    all_cores = ttnn.CoreRangeSet(ranges)

    x_tile = x_tensor.buffer_page_size()
    w_tile = w_tensor.buffer_page_size()
    y_tile = y_tensor.buffer_page_size()
    f32 = ttnn.float32

    # ---- CBs: identical descriptors on every launched core (uniform L1 addresses for remote writes) ----
    cbs = [
        _cb(CB_X_RESIDENT, all_cores, plan.x_block_depth * bt * kmax, x_tile, x_tensor.dtype),
        _cb(CB_WEIGHT, all_cores, kmax, w_tile, w_tensor.dtype),
        _cb(CB_BIAS_COEF, all_cores, 1, F32_TILE_BYTES, f32),
        _cb(CB_REDUCE_SCALER, all_cores, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_SQ_ACC, all_cores, 1, F32_TILE_BYTES, f32),
        _cb(CB_PARTIAL, all_cores, 2 * bt, F32_TILE_BYTES, f32),
        _cb(CB_GATHERED, all_cores, G * 2 * bt, F32_TILE_BYTES, f32),
        _cb(CB_COMBINED, all_cores, 2 * bt, F32_TILE_BYTES, f32),
        _cb(CB_COEF_IN, all_cores, bt, F32_TILE_BYTES, f32),
        _cb(CB_COEF_OUT, all_cores, bt, F32_TILE_BYTES, f32),
        _cb(CB_LOGITS_COEF, all_cores, bt, F32_TILE_BYTES, f32),
        _cb(CB_COMB_COEF, all_cores, bt, F32_TILE_BYTES, f32),
        _cb(CB_PRE_COLS, all_cores, n * bt, F32_TILE_BYTES, f32),
        _cb(CB_Y_OUT, all_cores, plan.y_depth * plan.y_chunk_tiles, y_tile, y_tensor.dtype),
        _cb(CB_OUT_STAGE, all_cores, OUT_STAGE_PAGES, F32_TILE_BYTES, f32),
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_cores, initial_value=0),
    ]

    # ---- group combine mcast (one Mcast2D per group; identical CT wire across groups) ----
    mcast_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
    helpers = {}
    mcast_ct = None
    if G > 1:
        for g, gx0, gy0 in groups:
            rect = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(gx0, gy0), ttnn.CoreCoord(gx0 + plan.group_w - 1, gy0 + plan.group_h - 1)
                    )
                ]
            )
            helpers[g] = ttnn.Mcast2D(device, rect, ttnn.CoreCoord(gx0, gy0), mcast_cfg)
            ct = list(helpers[g].compile_time_args())
            assert mcast_ct is None or ct == mcast_ct, "mcast CT wire must be identical across groups"
            mcast_ct = ct
    else:
        # group_cores == 1: no receivers, the pipe is never used. Placeholder wire with real sem ids.
        mcast_ct = [0, SEM_MCAST_READY, SEM_MCAST_CONSUMED, 0, 1, 0]

    # ---- kernel CT args ----
    reader_ct = [
        CB_X_RESIDENT,
        CB_WEIGHT,
        CB_BIAS_COEF,
        CB_REDUCE_SCALER,
        n,
        bt,
        kmax,
        plan.Ct,
        n * (n + 2),
    ]
    reader_ct += ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()

    compute_ct = [
        CB_X_RESIDENT,
        CB_WEIGHT,
        CB_BIAS_COEF,
        CB_REDUCE_SCALER,
        CB_SQ_ACC,
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COEF_OUT,
        CB_LOGITS_COEF,
        CB_COMB_COEF,
        CB_PRE_COLS,
        CB_Y_OUT,
        n,
        bt,
        kmax,
        G,
    ]

    writer_ct = [
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COEF_OUT,
        CB_COMB_COEF,
        CB_PRE_COLS,
        CB_Y_OUT,
        CB_OUT_STAGE,
        n,
        bt,
        plan.Ct,
        G,
        plan.y_chunk_tiles,
        plan.y_depth * plan.y_chunk_tiles,
        SEM_GATHER,
        n * (n + 2),
    ]
    assert len(writer_ct) == 17  # MCAST_CT_BASE in the writer
    writer_ct += mcast_ct
    writer_ct += ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args()
    writer_ct += ttnn.TensorAccessorArgs(post_tensor).get_compile_time_args()
    writer_ct += ttnn.TensorAccessorArgs(comb_tensor).get_compile_time_args()

    # ---- per-core RT args ----
    a_pre, a_post, a_res = scale
    C = plan.Ct * TILE
    scalar_bits = [
        _f32_bits(a_pre),
        _f32_bits(a_post),
        _f32_bits(a_res),
        _f32_bits(eps),
        _f32_bits(norm_eps),
        _f32_bits(1.0 / (n * C)),
        int(sinkhorn_iters),
    ]
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for g, gx0, gy0 in groups:
        ctt = plan.core_token_tiles[g]
        ts = plan.t_start[g]
        num_blocks = math.ceil(ctt / bt)
        root_virtual = device.worker_core_from_logical_core(ttnn.CoreCoord(gx0, gy0))
        for dy in range(plan.group_h):
            for dx in range(plan.group_w):
                x, y = gx0 + dx, gy0 + dy
                rank = dy * plan.group_w + dx
                cc = plan.core_c_tiles[rank]
                cs = plan.c_start[rank]
                reader_rt[x][y] = [
                    x_tensor.buffer_address(),
                    w_tensor.buffer_address(),
                    b_tensor.buffer_address(),
                    ts,
                    ctt,
                    cs,
                    cc,
                    num_blocks,
                ]
                mcast_rt = list(helpers[g].runtime_args(ttnn.CoreCoord(x, y))) if G > 1 else [0, 0, 0, 0]
                writer_rt[x][y] = [
                    y_tensor.buffer_address(),
                    post_tensor.buffer_address(),
                    comb_tensor.buffer_address(),
                    ts,
                    ctt,
                    cs,
                    cc,
                    num_blocks,
                    rank,
                    root_virtual.x,
                    root_virtual.y,
                ] + mcast_rt
                compute_rt[x][y] = [num_blocks, ctt, cc, rank] + scalar_bits

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_pre_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_pre_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=cfg.math_approx_mode,
    )
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for idx in UNPACK_TO_DEST_FP32_CBS:
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_pre_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_cfg,
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs), plan
