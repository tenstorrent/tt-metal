# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for groupnorm_sc_N_1_HW_C_b (op_design.md, l1_ledger.md).

Host-side planning:
  * _choose_gx(): column-slice count per image (span-aligned, G_core <= 32, L1-feasible),
    minimizing the max per-core tile count.
  * work split: units u = n*gx + xs; U < P -> cpi = min(P // U, Ht) cores per unit with a
    balanced row split and a gather/mcast combine on slot 0; U >= P -> cpi = 1 and whole-unit
    blocks round-robined contiguously over the P cores.
  * residency: K_alloc = min(B_max, K_max) resident chunks (or the TUNING_OVERRIDES value,
    clamped to [0, K_max]); the input ring adds the streaming depth iff B_max > K_alloc.

L1 BLOCK_SHARDED input (Refinement 1, even shards): a core's shard is its block. The unit is one image's
grid column (grid columns split C, grid rows split N*HW, cpi = gy / N shards per image); the gather goes up
the column to its root (shard row 0 of the image) and the stats mcast comes back down in ColumnMajor groups.
TILE: the input ring is the shard (two views, K = B), the output tiles are the output shard. RM: k = lcm(sw,
32) / sw consecutive sticks form one virtual row of W = lcm(sw, 32) / 32 tiles (lane v holds channel v mod sw),
compute tilizes chunks from the shard into the resident ring and untilizes the result into the output shard.

Every block knob (gx, chunk_ht, K, buffer depths, max_cores) is defined once here and flows into
CB sizes and kernel args.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Tunable constants (single source of truth; TUNING_OVERRIDES may override some)
# ---------------------------------------------------------------------------
CHUNK_TILES_TARGET = 32  # streaming quantum target (tiles per chunk)
D_IN = 2  # TILE input stream depth (chunks)
D_RM = 2  # RM stick CB depth (chunks)
D_OUT = 2  # output CB depth (chunks); group B total = max(4 KB*T, D_OUT*T*2 KB)
MAX_GROUP_LANES = 32  # group lanes in one tile row

BF16_TILE_BYTES = 2048
F32_TILE_BYTES = 4096
GATHER_SEMAPHORE_ID = 0
MCAST_FIRST_SEMAPHORE_ID = 1

# ---------------------------------------------------------------------------
# CB indices (semantic names; see op_design.md "Circular Buffers")
# ---------------------------------------------------------------------------
CB_INPUT_RM = 0
CB_OUTPUT_RM = 1  # sharded RM: untilized output, backed on the output shard
CB_INPUT_TILES = 2  # input ring, pass-1 view (alias group A)
CB_INPUT_KEEP = 3  # input ring, pass-2 view (alias group A)
CB_X_SQUARED = 4
CB_OUTPUT_TILES = 5
CB_REDUCE_SCALER = 6
CB_SUM_ACC = 7
CB_SUMSQ_ACC = 8
CB_CHAN_SCALE = 9
CB_CHAN_WORK = 10
CB_CHAN_SHIFT = 11
CB_GROUP_MAP = 12
CB_GAMMA_ROWS = 13
CB_BETA_ROWS = 14
CB_LOCAL_PARTIALS = 15
CB_GATHERED_PARTIALS = 16
CB_ONES_TILES = 17
CB_GROUP_SUMS = 18
CB_STATS_OUT = 19
CB_GROUP_STATS = 20
CB_AFFINE_STAGING = 21  # reader-only staging: pilot row (step 0) / affine slices (step B), disjoint in time
CB_PILOT_ROWS = 22  # shift k per channel (row 0, bf16), one per block, held for pass 1
CB_SUM_SEED = 23  # -n*k per channel (row 0, Float32 unpacked to DEST): the Sigma x accumulator's start
CB_GROUP_SHIFT = 24  # unit root: k per group lane (Float32, rows replicated), read by the finalize

NUM_CB_SLOTS = 64  # unpack_to_dest_mode vector length (covers WH 32 / BH 64)


def _ceil_div(a, b):
    return -(-a // b)


def _align_up(a, b):
    return _ceil_div(a, b) * b


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def l1_cb_budget(device):
    """Bytes available to this program's statically allocated CBs on every worker core.

    CBs are placed from the L1 allocator base upward, so the region is [allocator base, lowest live L1
    buffer). ``ttnn.get_max_worker_l1_unreserved_size()`` is NOT this: it is measured from the
    kernel-config base, so it also counts the kernel-config ring (70688 B on BH) that sits below the
    allocator base, and the CBs it admits overflow L1 by exactly that much. The memory view's per-bank
    size is measured from the allocator base, and its block table (offsets relative to that base)
    gives the lowest live L1 buffer, so caller-owned L1 tensors (e.g. an R1 shard) are excluded too.
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    budget = int(view.total_bytes_per_bank)
    for blk in view.block_table:
        if blk.get("allocated") == "yes":
            budget = min(budget, int(blk["address"]))
    return budget


def _tuning():
    from .groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES

    return dict(TUNING_OVERRIDES)


def input_ring_pages(*, T, K_alloc, stream, is_rm):
    """Input ring capacity: the resident tail plus the streaming depth (TILE: D_IN, RM: 1 tilized chunk)."""
    return K_alloc * T + int(stream) * (1 if is_rm else D_IN) * T


def group_b_bytes(T):
    """Alias group B (x^2 Float32 pass-1 intermediate / D_OUT-deep bf16 output ring): one allocation."""
    return max(F32_TILE_BYTES * T, D_OUT * T * BF16_TILE_BYTES)


def staging_bytes(*, W, map_period, is_rm, n_affine):
    """Reader staging: the pilot row (TILE: rows 0-1 of faces 0/1 per tile; RM: the slice of stick 0) or the
    n_affine gamma/beta slices (P channels from a 64 B-aligned start), whichever is larger."""
    pilot = map_period * 2 if is_rm else W * 128
    affine = n_affine * (_align_up(map_period * 2, 64) + 64)
    return _align_up(max(pilot, affine), 64)


def footprint_bytes(*, W, T, R, K_alloc, stream, is_rm, has_gamma, has_beta, map_period=None, sharded=False):
    """Per-core static L1 footprint of every CB (l1_ledger.md "Per-core footprint"). Sharded: the input
    sticks (RM) / the input ring (TILE) and the output (RM sticks / TILE tiles) are the shards, not static CBs."""
    hg, hb = int(has_gamma), int(has_beta)
    map_period = W * 32 if map_period is None else map_period
    tile_shard = sharded and not is_rm
    bf16_pages = (
        (D_RM * T if is_rm and not sharded else 0)
        + (0 if tile_shard else input_ring_pages(T=T, K_alloc=K_alloc, stream=stream, is_rm=is_rm))
        + 1  # reduce scaler
        + R  # ones tiles
        + W * (2 + hg + hb)  # map, pilot rows, gamma rows, beta rows
    )
    # Sharded TILE: x^2 alone (its group-B partner, the output tiles, is the output shard).
    group_b = F32_TILE_BYTES * T if tile_shard else group_b_bytes(T)
    # 3W: alias groups E (sum acc / chan scale) and C' (sumsq acc / chan work / chan shift), the sum seed.
    # 2R + 7: gathered partials, group D (2), stats out (2), group stats (2), group shift (1).
    f32_bytes = group_b + F32_TILE_BYTES * (3 * W + 2 * R + 7)
    staging = staging_bytes(W=W, map_period=map_period, is_rm=is_rm, n_affine=hg + hb)
    return BF16_TILE_BYTES * bf16_pages + f32_bytes + staging


@dataclass
class CoreWork:
    core: object  # ttnn.CoreCoord
    unit_start: int
    num_blocks: int
    row_start: int
    rows_core: int
    is_root: int
    slot: int
    root: object  # ttnn.CoreCoord (logical)


@dataclass
class GroupNormPlan:
    N: int
    HW: int
    C: int
    G: int
    Ht: int
    Wt: int
    Cg: int
    span_tiles: int
    gps: int
    S: int
    gx: int
    W: int
    G_core: int
    U: int
    P: int
    cpi: int
    h: int
    T: int
    R: int
    rows_core_max: int
    B_max: int
    K_max: int
    K_alloc: int
    stream: bool
    is_rm: bool
    has_gamma: bool
    has_beta: bool
    grid_x: int
    grid_y: int
    l1_budget: int
    footprint: int
    work: list = field(default_factory=list)
    sharded: bool = False
    map_period: int = 0  # channels per slice (lane v holds channel v mod map_period)
    shard_grid: object = None  # sharded: the shard grid (ttnn.CoreRangeSet)


def _evaluate_gx(gx, *, N, Ht, S, span_tiles, gps, P, chunk_target, is_rm, has_gamma, has_beta, l1_budget):
    W = (S // gx) * span_tiles
    G_core = (S // gx) * gps
    if G_core > MAX_GROUP_LANES:
        return None
    U = N * gx
    cpi = min(P // U, Ht) if U < P else 1
    rows_core_max = _ceil_div(Ht, cpi)
    h = max(1, min(chunk_target // W, rows_core_max))
    # Balance the chunks (no small ragged tail) and keep >= 2 per block so pass 1 overlaps its own read.
    n_chunks = max(_ceil_div(rows_core_max, h), min(rows_core_max, 2))
    h = _ceil_div(rows_core_max, n_chunks)
    T = h * W
    R = _ceil_div(cpi, 32)
    f_fixed = footprint_bytes(
        W=W, T=T, R=R, K_alloc=0, stream=True, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta
    )
    if f_fixed > l1_budget:
        return None
    if U < P:
        max_core_tiles = rows_core_max * W
        cores_used = U * cpi
    else:
        max_core_tiles = _ceil_div(U, P) * Ht * W
        cores_used = P
    return dict(
        gx=gx,
        W=W,
        G_core=G_core,
        U=U,
        cpi=cpi,
        rows_core_max=rows_core_max,
        h=h,
        T=T,
        R=R,
        f_fixed=f_fixed,
        max_core_tiles=max_core_tiles,
        cores_used=cores_used,
    )


def _choose_gx(*, N, Ht, S, span_tiles, gps, P, chunk_target, is_rm, has_gamma, has_beta, l1_budget):
    best = None
    best_key = None
    for gx in _divisors(S):
        cand = _evaluate_gx(
            gx,
            N=N,
            Ht=Ht,
            S=S,
            span_tiles=span_tiles,
            gps=gps,
            P=P,
            chunk_target=chunk_target,
            is_rm=is_rm,
            has_gamma=has_gamma,
            has_beta=has_beta,
            l1_budget=l1_budget,
        )
        if cand is None:
            continue
        # Minimize max per-core tiles; ties: more cores, then larger gx (TILE) / smaller gx (RM).
        key = (cand["max_core_tiles"], -cand["cores_used"], gx if is_rm else -gx)
        if best_key is None or key < best_key:
            best, best_key = cand, key
    return best


def _distribute(plan_fields, grid_x):
    """Per-core work list (row-major core order, so Mcast RowMajor groups == units)."""
    U, P, cpi, Ht = plan_fields["U"], plan_fields["P"], plan_fields["cpi"], plan_fields["Ht"]

    def coord(c):
        return ttnn.CoreCoord(c % grid_x, c // grid_x)

    work = []
    if U < P:
        q, rem = divmod(Ht, cpi)
        for c in range(U * cpi):
            unit, slot = divmod(c, cpi)
            row_start = slot * q + min(slot, rem)
            rows = q + (1 if slot < rem else 0)
            work.append(
                CoreWork(
                    core=coord(c),
                    unit_start=unit,
                    num_blocks=1,
                    row_start=row_start,
                    rows_core=rows,
                    is_root=int(slot == 0),
                    slot=slot,
                    root=coord(unit * cpi),
                )
            )
    else:
        base, extra = divmod(U, P)
        for c in range(P):
            n_units = base + (1 if c < extra else 0)
            start = c * base + min(c, extra)
            work.append(
                CoreWork(
                    core=coord(c),
                    unit_start=start,
                    num_blocks=n_units,
                    row_start=0,
                    rows_core=Ht,
                    is_root=1,
                    slot=0,
                    root=coord(c),
                )
            )
    return work


def plan_or_raise(input_tensor, num_groups, *, has_gamma, has_beta):
    """Build the blocking plan; NotImplementedError when no feasible column-slice count exists."""
    tuning = _tuning()
    if input_tensor.is_sharded():
        return _plan_sharded(input_tensor, num_groups, has_gamma=has_gamma, has_beta=has_beta, tuning=tuning)
    N, _, HW, C = tuple(input_tensor.shape)
    G = num_groups
    Ht, Wt = HW // 32, C // 32
    Cg = C // G
    L = _lcm(Cg, 32)
    if C % L != 0:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C_b: lcm(Cg={Cg}, 32)={L} does not divide C={C} "
            "(cross_column_group_combine regime, not built)"
        )
    span_tiles, gps = L // 32, L // Cg
    S = Wt // span_tiles
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    P = grid.x * grid.y
    if tuning.get("max_cores") is not None:
        P = max(1, min(P, int(tuning["max_cores"])))
    chunk_target = CHUNK_TILES_TARGET if tuning.get("chunk_tiles_target") is None else int(tuning["chunk_tiles_target"])
    chunk_target = max(1, chunk_target)
    l1_budget = l1_cb_budget(device)

    best = _choose_gx(
        N=N,
        Ht=Ht,
        S=S,
        span_tiles=span_tiles,
        gps=gps,
        P=P,
        chunk_target=chunk_target,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        l1_budget=l1_budget,
    )
    if best is None:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C_b: no feasible column-slice count for C={C}, G={G} "
            "(G_core <= 32 and L1 fit); cross_column_group_combine regime not built"
        )

    W, T, R, h = best["W"], best["T"], best["R"], best["h"]
    B_max = _ceil_div(best["rows_core_max"], h)
    K_max = (l1_budget - best["f_fixed"]) // (BF16_TILE_BYTES * T)
    override = tuning.get("max_resident_chunks")
    if override is None:
        K_alloc = min(B_max, K_max)
    else:
        K_alloc = max(0, min(int(override), K_max, B_max))
    stream = B_max > K_alloc

    fields = dict(U=best["U"], P=P, cpi=best["cpi"], Ht=Ht)
    plan = GroupNormPlan(
        N=N,
        HW=HW,
        C=C,
        G=G,
        Ht=Ht,
        Wt=Wt,
        Cg=Cg,
        span_tiles=span_tiles,
        gps=gps,
        S=S,
        gx=best["gx"],
        W=W,
        G_core=best["G_core"],
        U=best["U"],
        P=P,
        cpi=best["cpi"],
        h=h,
        T=T,
        R=R,
        rows_core_max=best["rows_core_max"],
        B_max=B_max,
        K_max=K_max,
        K_alloc=K_alloc,
        stream=stream,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        grid_x=grid.x,
        grid_y=grid.y,
        l1_budget=l1_budget,
        footprint=footprint_bytes(
            W=W, T=T, R=R, K_alloc=K_alloc, stream=stream, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta
        ),
        map_period=W * 32,
    )
    plan.work = _distribute(fields, grid.x)
    return plan


def _plan_sharded(input_tensor, num_groups, *, has_gamma, has_beta, tuning):
    """Blocking plan of an L1 BLOCK_SHARDED input: one block per core, the core's own shard."""
    mem = input_tensor.memory_config()
    spec = mem.shard_spec
    N, _, HW, C = tuple(input_tensor.shape)
    G = num_groups
    Cg = C // G
    sh, sw = int(spec.shape[0]), int(spec.shape[1])
    bbox = spec.grid.bounding_box()
    x0, y0 = bbox.start.x, bbox.start.y
    gx, gy = bbox.end.x - x0 + 1, bbox.end.y - y0 + 1
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    def refuse(why):
        raise NotImplementedError(f"groupnorm_sc_N_1_HW_C_b: block-sharded input: {why}")

    if mem.buffer_type != ttnn.BufferType.L1:
        refuse("only L1 shards are supported")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or spec.grid.num_cores() != gx * gy:
        refuse("the shard grid must be one ROW_MAJOR rectangle")
    if sh * gy != N * HW or sw * gx != C:
        refuse(f"shard {sh}x{sw} on a {gx}x{gy} grid is not an even split (ragged shards: Refinement 3)")
    if HW % sh != 0:
        refuse(f"shard height {sh} straddles images (HW={HW})")
    if sw % Cg != 0:
        refuse(f"shard width {sw} splits a group (C/G={Cg})")
    G_core = sw // Cg
    if G_core > MAX_GROUP_LANES:
        refuse(f"{G_core} groups per core column exceed one tile row of group lanes")
    if is_rm:
        L = _lcm(sw, 32)
        k, W = L // sw, L // 32
    else:
        k, W = 1, sw // 32
    if sh % (32 * k) != 0:
        refuse(f"shard height {sh} is not a whole number of {k}-stick tile rows")
    rows_core = sh // (32 * k)
    cpi = HW // sh
    R = _ceil_div(cpi, 32)

    chunk_target = CHUNK_TILES_TARGET if tuning.get("chunk_tiles_target") is None else int(tuning["chunk_tiles_target"])
    # A chunk is a whole number of the shard's tile rows (no ragged tail: the shard-backed FIFOs wrap exactly).
    h = _largest_divisor_at_most(rows_core, max(1, max(1, chunk_target) // W))
    T = h * W
    B = rows_core // h

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    # The output shard is allocated after this plan, below the live L1 buffers the budget is measured against.
    out_shard_bytes = _align_up(sh * sw * 2, 64)
    l1_budget = l1_cb_budget(device) - out_shard_bytes
    fp_kw = dict(W=W, T=T, R=R, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta, map_period=sw, sharded=True)
    if is_rm:
        f_fixed = footprint_bytes(K_alloc=0, stream=True, **fp_kw)
        K_max = (l1_budget - f_fixed) // (BF16_TILE_BYTES * T)
        if K_max < 0:
            refuse(f"the static CBs ({f_fixed} B) exceed the L1 left beside the shards ({l1_budget} B)")
        override = tuning.get("max_resident_chunks")
        K_alloc = min(B, K_max) if override is None else max(0, min(int(override), K_max, B))
    else:
        K_max = K_alloc = B  # the shard itself is the resident ring
    stream = B > K_alloc
    footprint = footprint_bytes(K_alloc=K_alloc, stream=stream, **fp_kw)
    if footprint > l1_budget:
        refuse(f"the static CBs ({footprint} B) exceed the L1 left beside the shards ({l1_budget} B)")

    plan = GroupNormPlan(
        N=N,
        HW=HW,
        C=C,
        G=G,
        Ht=HW // 32,
        Wt=C // 32,
        Cg=Cg,
        span_tiles=0,
        gps=0,
        S=gx,
        gx=gx,
        W=W,
        G_core=G_core,
        U=N * gx,
        P=gx * gy,
        cpi=cpi,
        h=h,
        T=T,
        R=R,
        rows_core_max=rows_core,
        B_max=B,
        K_max=K_max,
        K_alloc=K_alloc,
        stream=stream,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        grid_x=grid.x,
        grid_y=grid.y,
        l1_budget=l1_budget,
        footprint=footprint,
        sharded=True,
        map_period=sw,
        shard_grid=spec.grid,
    )
    for y in range(gy):
        n, slot = divmod(y, cpi)
        for x in range(gx):
            plan.work.append(
                CoreWork(
                    core=ttnn.CoreCoord(x0 + x, y0 + y),
                    unit_start=n * gx + x,
                    num_blocks=1,
                    row_start=slot * rows_core,
                    rows_core=rows_core,
                    is_root=int(slot == 0),
                    slot=slot,
                    root=ttnn.CoreCoord(x0 + x, y0 + n * cpi),
                )
            )
    return plan


def _used_core_range_set(num_cores, grid_x):
    """The first num_cores cores of the grid in row-major order."""
    ranges = []
    full_rows, rem = divmod(num_cores, grid_x)
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(total_size, cores, formats):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page) for idx, fmt, page in formats
        ],
    )


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def create_program_descriptor(plan, input_tensor, output_tensor, *, gamma=None, beta=None, eps=1e-5):
    device = input_tensor.device()
    W, T, R, h = plan.W, plan.T, plan.R, plan.h
    is_rm = plan.is_rm
    hg, hb = int(plan.has_gamma), int(plan.has_beta)
    bf16 = ttnn.bfloat16
    f32 = ttnn.float32

    sharded = plan.sharded
    cores = plan.shard_grid if sharded else _used_core_range_set(len(plan.work), plan.grid_x)
    map_period = plan.map_period

    def shard_cb(tensor, formats):
        # Zero-copy: the CB is the core's own shard (no NoC traffic); the views are tile-sized pages over it.
        d = ttnn.cb_descriptor_from_sharded_tensor(formats[0][0], tensor)
        d.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page) for idx, fmt, page in formats
        ]
        return d

    # ---------------- Circular buffers ----------------
    cbs = []
    if is_rm:
        if sharded:
            cbs.append(shard_cb(input_tensor, [(CB_INPUT_RM, bf16, BF16_TILE_BYTES)]))
            cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_RM, bf16, BF16_TILE_BYTES)]))
        else:
            cbs.append(_cb(D_RM * T * BF16_TILE_BYTES, cores, [(CB_INPUT_RM, bf16, BF16_TILE_BYTES)]))
    # Alias group A: the input ring, K_alloc resident chunks + the streaming depth, as two FIFO views pushed in
    # lockstep. Pass 1 pops every chunk from CB_INPUT_TILES; CB_INPUT_KEEP drops the streamed head in pass 1 and
    # holds the resident tail (then the re-read head) for pass 2. Sharded TILE: the ring is the shard (K = B).
    ring_views = [(CB_INPUT_TILES, bf16, BF16_TILE_BYTES), (CB_INPUT_KEEP, bf16, BF16_TILE_BYTES)]
    if sharded and not is_rm:
        cbs.append(shard_cb(input_tensor, ring_views))
        # x^2 alone; the output tiles are the output shard.
        cbs.append(_cb(F32_TILE_BYTES * T, cores, [(CB_X_SQUARED, f32, F32_TILE_BYTES)]))
        cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_TILES, bf16, BF16_TILE_BYTES)]))
    else:
        cbs.append(
            _cb(
                input_ring_pages(T=T, K_alloc=plan.K_alloc, stream=plan.stream, is_rm=is_rm) * BF16_TILE_BYTES,
                cores,
                ring_views,
            )
        )
        # Alias group B: x^2 (pass 1) / output (pass 2), disjoint lifetimes.
        cbs.append(
            _cb(
                group_b_bytes(T),
                cores,
                [(CB_X_SQUARED, f32, F32_TILE_BYTES), (CB_OUTPUT_TILES, bf16, BF16_TILE_BYTES)],
            )
        )
    cbs.append(_cb(BF16_TILE_BYTES, cores, [(CB_REDUCE_SCALER, bf16, BF16_TILE_BYTES)]))
    # Alias group E: the Sigma x accumulator (UnpackToDestFp32, pass 1) / the FPU-readable chan_scale. The publish
    # copy reads accumulator tile i and packs chan tile i over it (in-place lockstep; every view returns to its
    # base each block), so the accumulator costs no L1 of its own.
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [(CB_SUM_ACC, f32, F32_TILE_BYTES), (CB_CHAN_SCALE, f32, F32_TILE_BYTES)],
        )
    )
    # Alias group C: the Sigma (x-k)^2 accumulator / chan_work / chan_shift, in-place lockstep hand-offs
    # (publish, 5d/5e).
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [
                (CB_SUMSQ_ACC, f32, F32_TILE_BYTES),
                (CB_CHAN_WORK, f32, F32_TILE_BYTES),
                (CB_CHAN_SHIFT, f32, F32_TILE_BYTES),
            ],
        )
    )
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_GROUP_MAP, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_PILOT_ROWS, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(W * F32_TILE_BYTES, cores, [(CB_SUM_SEED, f32, F32_TILE_BYTES)]))
    cbs.append(_cb(F32_TILE_BYTES, cores, [(CB_GROUP_SHIFT, f32, F32_TILE_BYTES)]))
    if hg:
        cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_GAMMA_ROWS, bf16, BF16_TILE_BYTES)]))
    if hb:
        cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_BETA_ROWS, bf16, BF16_TILE_BYTES)]))
    # Alias group D: local partials (collapse -> gather) / group sums (root), disjoint lifetimes.
    cbs.append(
        _cb(
            2 * F32_TILE_BYTES,
            cores,
            [(CB_LOCAL_PARTIALS, f32, F32_TILE_BYTES), (CB_GROUP_SUMS, f32, F32_TILE_BYTES)],
        )
    )
    # Remote-written landing zones (allocated on every used core so the base address is identical).
    cbs.append(_cb(2 * R * F32_TILE_BYTES, cores, [(CB_GATHERED_PARTIALS, f32, F32_TILE_BYTES)]))
    cbs.append(_cb(R * BF16_TILE_BYTES, cores, [(CB_ONES_TILES, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(2 * F32_TILE_BYTES, cores, [(CB_STATS_OUT, f32, F32_TILE_BYTES)]))
    cbs.append(_cb(2 * F32_TILE_BYTES, cores, [(CB_GROUP_STATS, f32, F32_TILE_BYTES)]))
    stage = staging_bytes(W=W, map_period=map_period, is_rm=is_rm, n_affine=hg + hb)
    cbs.append(_cb(stage, cores, [(CB_AFFINE_STAGING, bf16, stage)]))

    # ---------------- Kernel args ----------------
    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    reader_ct = [W, h, int(is_rm), hg, hb, R, plan.Cg, plan.Wt, plan.Ht, plan.gx, in_page_bytes, map_period]
    reader_ct.append(int(sharded))
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if gamma is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(beta) if beta is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    exp_sbw = _largest_divisor_at_most(W, 4)  # expand matmul out_subblock_w (<= DEST_AUTO_LIMIT fp32 half-sync)
    out_cb_pages = group_b_bytes(T) // BF16_TILE_BYTES  # cb_output_tiles capacity (drain barrier)
    compute_ct = [W, h, int(is_rm), hg, hb, R, exp_sbw, out_cb_pages, int(sharded)]

    # The writer is built once per role (IS_ROOT is compile-time): each binary carries only its own mcast pipe
    # (sender on unit roots, receiver elsewhere) and only the root's gather-wait / zero-fill code.
    def writer_ct(is_root):
        ct = [W, h, R, plan.Wt, plan.Ht, plan.gx, out_page_bytes, int(is_root), int(is_rm), int(sharded)]
        ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        return ct

    inv_n_bits = _f32_bits(1.0 / (plan.HW * plan.Cg))
    eps_bits = _f32_bits(eps)

    reader_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = {1: ttnn.RuntimeArgs(), 0: ttnn.RuntimeArgs()}
    writer_cores = {1: [], 0: []}
    for wk in plan.work:
        x, y = wk.core.x, wk.core.y
        B_core = _ceil_div(wk.rows_core, h)
        K_core = min(plan.K_alloc, B_core)
        root_virt = device.worker_core_from_logical_core(wk.root)
        reader_rt[x][y] = [
            input_tensor.buffer_address(),
            gamma_addr,
            beta_addr,
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            wk.is_root,
            root_virt.x,
            root_virt.y,
        ]
        compute_rt[x][y] = [wk.num_blocks, wk.rows_core, K_core, wk.is_root, inv_n_bits, eps_bits]
        writer_cores[wk.is_root].append(wk.core)
        writer_rt[wk.is_root][x][y] = [
            output_tensor.buffer_address(),
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            wk.slot,
            root_virt.x,
            root_virt.y,
            plan.cpi,
        ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )
    writer_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_writer.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writer_cores[role]]),
            compile_time_args=writer_ct(role),
            runtime_args=writer_rt[role],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        )
        for role in (1, 0)
        if writer_cores[role]
    ]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for idx in (CB_SUM_ACC, CB_SUMSQ_ACC, CB_GROUP_SUMS, CB_SUM_SEED, CB_GROUP_SHIFT):
        unpack_modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = unpack_modes
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    gather_sem = ttnn.SemaphoreDescriptor(id=GATHER_SEMAPHORE_ID, core_ranges=cores, initial_value=0)
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[gather_sem], cbs=cbs)

    # Stats multicast: one channel over all used cores, groups of cpi consecutive cores, sender = slot 0 (the
    # unit root). DRAM: units are row-major core runs; sharded: a unit is a grid-column segment (ColumnMajor).
    # cpi == 1 degenerates to the sender's local copy.
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
        cores,
        plan.cpi,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.ColumnMajor if sharded else ttnn.McastCoreOrder.RowMajor,
    )
    mcast.attach(descriptor, "mcast", writer_kernels, MCAST_FIRST_SEMAPHORE_ID)
    descriptor.kernels = [reader_kernel, *writer_kernels, compute_kernel]
    return descriptor
