# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — ProgramDescriptor (host derivation, work distribution, CBs, kernels).

Realizes op_design.md's Blocking Model:

  * block = `block_rows x core_w_tiles` tiles; a block always spans the core's WHOLE W slice.
  * rows are split contiguously over `num_row_groups` groups of cores; each group is an `a x b`
    rectangle of `num_w_splits` cores that split W and combine their per-row sum-of-squares
    partials on a root core (unicast gather -> root combine -> rstd multicast).
  * regimes: R1 row_split (num_w_splits == 1), R2 w_split_root_combine (interleaved, > 1),
    R3 width_sharded_resident (WIDTH_SHARDED input: the shard IS the core assignment, x/out
    are zero-copy CBs on the shard buffers).

Every knob is defined once here (§H0); every CB size, loop bound and kernel argument is
derived from it. `derive_blocking()` is pure and host-checkable.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# =============================================================================
# §H0 — knobs (single source of truth)
# =============================================================================

W_TILES_PER_CORE_TARGET = 16  # grid-synchronization lamp: W tiles per core when rows under-fill the grid
DEPTH_X = 2  # TILE-interleaved x buffer depth, in blocks (reader prefetch of block k+1)
DEPTH_OUT = 2  # TILE-interleaved output buffer depth, in blocks (writer drains k while compute packs k+1)
DEPTH_X_STICKS_BLOCKS = 2  # ROW_MAJOR x stick buffer depth, in blocks
DEPTH_OUT_STICKS_ROWS = 2  # ROW_MAJOR output stick buffer depth, in tile-rows
L1_MARGIN_BYTES = 64 * 1024

TILE = 32
P32 = ttnn.tile_size(ttnn.float32)  # fp32 intermediate tile bytes (all accumulated intermediates)
T_SCALER = ttnn.tile_size(ttnn.bfloat16)  # bf16 reduce-scaler tile bytes

# Circular-buffer slots. The kernels receive these as NAMED compile-time args (same names).
CB_X_TILES = 0
CB_X_STICKS = 1
CB_SCALER = 2
CB_SUMSQ_PARTIAL = 3
CB_PARTIAL_COLLAPSED = 4
CB_GATHER = 5
CB_RSTD_HANDOFF = 6
CB_RSTD = 7
CB_GAMMA_TILES = 8
CB_GAMMA_STICKS = 9
CB_NORMED = 10
CB_OUTPUT_TILES = 11
CB_OUT_STICKS = 12

# Semaphores (declared on every program core).
SEM_GATHER = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2

GAMMA_MODE_NONE = 0
GAMMA_MODE_TILE = 1
GAMMA_MODE_RM = 2

# Runtime-arg sentinel for a core that owns no W slice (passive bounding-box core in R3).
NO_W_SPLIT = 0xFFFFFFFF


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# =============================================================================
# Footprint model (§H1)
# =============================================================================


@dataclass(frozen=True)
class Footprint:
    t_in: int  # input tile bytes
    t_out: int  # output tile bytes
    t_g: int  # gamma tile bytes (0 without gamma)
    has_gamma: bool
    input_rm: bool  # ROW_MAJOR input (output layout == input layout)
    sharded: bool  # R3: x/out already resident, not allocated by the op

    def per_block(self, wc: int, cw: int) -> int:
        """Bytes per block-row unit: everything that scales with block_rows."""
        if self.sharded:
            x_term, out_term = 0, 0
        elif self.input_rm:
            x_term = (DEPTH_X_STICKS_BLOCKS + 1) * self.t_in  # stick blocks + one tilized block
            out_term = self.t_out  # one block feeding the untilize
        else:
            x_term = DEPTH_X * self.t_in
            out_term = DEPTH_OUT * self.t_out
        normed_term = P32 if self.has_gamma else 0
        collective = cw + 2 + (2 if cw > 1 else 0)  # gather slots + sumsq/rstd (+ collapsed/handoff)
        return wc * (x_term + normed_term + out_term) + P32 * collective

    def fixed(self, wc: int) -> int:
        """Bytes independent of block_rows: resident gamma slice, scaler, RM output stick window."""
        total = T_SCALER
        if self.has_gamma:
            total += wc * self.t_g
        if self.input_rm:
            total += DEPTH_OUT_STICKS_ROWS * wc * self.t_out
        return total

    def block_rows_max(self, wc: int, cw: int, budget: int) -> int:
        free = budget - self.fixed(wc)
        return free // self.per_block(wc, cw) if free > 0 else 0


# =============================================================================
# Work distribution
# =============================================================================


@dataclass(frozen=True)
class CoreRole:
    x: int
    y: int
    group: int
    w_split_index: int  # index inside the group rectangle (row-wise); NO_W_SPLIT for passive cores
    core_w_tiles: int  # 0 for passive cores
    w_tile_start: int
    row_tile_start: int
    core_row_tiles: int

    @property
    def is_active(self) -> bool:
        return self.core_w_tiles > 0

    @property
    def is_root(self) -> bool:
        return self.is_active and self.w_split_index == 0

    @property
    def coord(self) -> ttnn.CoreCoord:
        return ttnn.CoreCoord(self.x, self.y)


@dataclass(frozen=True)
class Group:
    index: int
    rect: ttnn.CoreRangeSet  # the a x b rectangle (mcast rect)
    root: ttnn.CoreCoord  # logical
    cores: tuple  # tuple[CoreRole, ...]
    row_tile_start: int
    core_row_tiles: int

    @property
    def num_active(self) -> int:
        return sum(1 for c in self.cores if c.is_active)


@dataclass(frozen=True)
class Blocking:
    regime: str
    tensor_row_tiles: int
    tensor_w_tiles: int
    num_w_splits: int
    rect_a: int
    rect_b: int
    num_row_groups: int
    core_w_tiles_max: int
    block_rows: int
    groups: tuple  # tuple[Group, ...]
    l1_cb_budget: int

    @property
    def all_roles(self):
        return [c for g in self.groups for c in g.cores]

    @property
    def num_blocks_max(self) -> int:
        return max(_ceil_div(g.core_row_tiles, self.block_rows) for g in self.groups)


def _core_range_set(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in cores])


def _rect(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _derive_w_splits(wt: int, rt: int, grid_x: int, grid_y: int, fp: Footprint, budget: int):
    """§H2: number of W splits (as an a x b rectangle) for an interleaved input."""
    # residency floor — the widest per-core slice whose block footprint fits at least one tile-row
    core_w_tiles_max_l1 = 0
    for wc in range(wt, 0, -1):
        if fp.block_rows_max(wc, _ceil_div(wt, wc), budget) >= 1:
            core_w_tiles_max_l1 = wc
            break
    if core_w_tiles_max_l1 == 0:
        raise ValueError("rms_norm: a single tile column does not fit the per-core L1 budget")
    cw_res = _ceil_div(wt, core_w_tiles_max_l1)
    # occupancy term — rows alone under-fill the grid, so split W to W_TILES_PER_CORE_TARGET tiles/core
    cw_occ = _ceil_div(wt, W_TILES_PER_CORE_TARGET) if rt < grid_x * grid_y else 1
    cw_req = max(1, min(wt, max(cw_res, cw_occ)))
    # shape into an a x b rectangle: a*b >= cw_req, a <= grid_x, b <= grid_y, a*b <= wt
    for a in range(min(cw_req, grid_x), 0, -1):
        b = _ceil_div(cw_req, a)
        if b <= grid_y and a * b <= wt:
            return a * b, a, b
    raise ValueError(f"rms_norm: cannot shape {cw_req} W-splits into the {grid_x}x{grid_y} grid for W tiles={wt}")


def _split_two_group(total: int, parts: int):
    """ceil/floor split: the first `total % parts` parts get one extra unit."""
    q, r = divmod(total, parts)
    starts, counts = [], []
    acc = 0
    for i in range(parts):
        n = q + (1 if i < r else 0)
        starts.append(acc)
        counts.append(n)
        acc += n
    return starts, counts


def derive_blocking(input_tensor: ttnn.Tensor, gamma: Optional[ttnn.Tensor], grid_x: int, grid_y: int) -> Blocking:
    shape = list(input_tensor.padded_shape)
    w = shape[-1]
    rt = (math.prod(shape[:-2]) if len(shape) > 2 else 1) * _ceil_div(shape[-2], TILE)
    wt = w // TILE
    input_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    sharded = input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    t_in = ttnn.tile_size(input_tensor.dtype)
    fp = Footprint(
        t_in=t_in,
        t_out=t_in,
        t_g=ttnn.tile_size(gamma.dtype) if gamma is not None else 0,
        has_gamma=gamma is not None,
        input_rm=input_rm,
        sharded=sharded,
    )
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_MARGIN_BYTES

    if sharded:
        shard_spec = input_tensor.memory_config().shard_spec
        shard_cores = ttnn.corerange_to_cores(shard_spec.grid, None, True)
        shard_w = shard_spec.shape[1]
        if shard_w % TILE != 0:
            raise ValueError(f"rms_norm: shard width {shard_w} must be a multiple of {TILE}")
        core_w_tiles = shard_w // TILE
        if len(shard_cores) * core_w_tiles != wt:
            raise ValueError(
                f"rms_norm: shard grid ({len(shard_cores)} cores x {core_w_tiles} tiles) does not cover W tiles={wt}"
            )
        shard_bytes = rt * core_w_tiles * t_in
        budget -= 2 * shard_bytes  # input + output shards are already resident
        xs = [c.x for c in shard_cores]
        ys = [c.y for c in shard_cores]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        a, b = x1 - x0 + 1, y1 - y0 + 1
        cw = a * b
        block_rows_max = fp.block_rows_max(core_w_tiles, cw, budget)
        if block_rows_max < 1:
            raise ValueError("rms_norm: the WIDTH_SHARDED intermediates do not fit next to the resident shards")
        block_rows = min(rt, block_rows_max)
        index_of = {(c.x, c.y): i for i, c in enumerate(shard_cores)}
        roles = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                i = index_of.get((x, y))
                if i is None:
                    roles.append(CoreRole(x, y, 0, NO_W_SPLIT, 0, 0, 0, rt))
                else:
                    roles.append(CoreRole(x, y, 0, i, core_w_tiles, i * core_w_tiles, 0, rt))
        group = Group(0, _rect(x0, y0, x1, y1), shard_cores[0], tuple(roles), 0, rt)
        return Blocking(
            regime="R3_width_sharded_resident",
            tensor_row_tiles=rt,
            tensor_w_tiles=wt,
            num_w_splits=cw,
            rect_a=a,
            rect_b=b,
            num_row_groups=1,
            core_w_tiles_max=core_w_tiles,
            block_rows=block_rows,
            groups=(group,),
            l1_cb_budget=budget,
        )

    cw, a, b = _derive_w_splits(wt, rt, grid_x, grid_y, fp, budget)
    groups_x, groups_y = grid_x // a, grid_y // b
    num_row_groups = min(groups_x * groups_y, rt)
    row_starts, row_counts = _split_two_group(rt, num_row_groups)
    qw, rw = divmod(wt, cw)
    core_w_tiles_max = qw + (1 if rw > 0 else 0)

    block_rows_max = fp.block_rows_max(core_w_tiles_max, cw, budget)
    if block_rows_max < 1:
        raise ValueError("rms_norm: derived W split does not fit the per-core L1 budget")
    block_rows = min(max(row_counts), block_rows_max)

    groups = []
    for g in range(num_row_groups):
        ox, oy = (g % groups_x) * a, (g // groups_x) * b
        roles = []
        for ly in range(b):
            for lx in range(a):
                c = ly * a + lx
                wc = qw + (1 if c < rw else 0)
                roles.append(CoreRole(ox + lx, oy + ly, g, c, wc, c * qw + min(c, rw), row_starts[g], row_counts[g]))
        groups.append(
            Group(
                g,
                _rect(ox, oy, ox + a - 1, oy + b - 1),
                ttnn.CoreCoord(ox, oy),
                tuple(roles),
                row_starts[g],
                row_counts[g],
            )
        )
    return Blocking(
        regime="R1_row_split" if cw == 1 else "R2_w_split_root_combine",
        tensor_row_tiles=rt,
        tensor_w_tiles=wt,
        num_w_splits=cw,
        rect_a=a,
        rect_b=b,
        num_row_groups=num_row_groups,
        core_w_tiles_max=core_w_tiles_max,
        block_rows=block_rows,
        groups=tuple(groups),
        l1_cb_budget=budget,
    )


# =============================================================================
# Program descriptor
# =============================================================================


def _cb(index: int, core_ranges, page_size: int, num_pages: int, dtype) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma: Optional[ttnn.Tensor],
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    blocking = derive_blocking(input_tensor, gamma, grid.x, grid.y)

    input_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    sharded = blocking.regime.startswith("R3")
    has_gamma = gamma is not None
    if not has_gamma:
        gamma_mode = GAMMA_MODE_NONE
    elif gamma.layout == ttnn.ROW_MAJOR_LAYOUT:
        gamma_mode = GAMMA_MODE_RM
    else:
        gamma_mode = GAMMA_MODE_TILE

    in_dtype, out_dtype = input_tensor.dtype, output_tensor.dtype
    t_in, t_out = ttnn.tile_size(in_dtype), ttnn.tile_size(out_dtype)
    e_in, e_out = input_tensor.element_size(), output_tensor.element_size()
    t_g = ttnn.tile_size(gamma.dtype) if has_gamma else 0
    e_g = gamma.element_size() if has_gamma else 0

    B = blocking.block_rows
    Cw = blocking.num_w_splits
    Wt = blocking.tensor_w_tiles
    Rt = blocking.tensor_row_tiles
    W = Wt * TILE
    inv_w_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    roles = blocking.all_roles
    all_cores = ttnn.CoreRangeSet([r for g in blocking.groups for r in g.rect.ranges()])
    active_roles = [r for r in roles if r.is_active]
    active_cores = _core_range_set([(r.x, r.y) for r in active_roles])

    # ---------------- circular buffers ----------------
    # The collective CBs (gather slots, rstd landing) are addressed by REMOTE cores via base +
    # slot offset, so they must sit at the same L1 address on every core: they are created first,
    # uniformly over the whole program range, ahead of any per-W-group (ragged-size) CB.
    cbs = [
        _cb(CB_GATHER, all_cores, P32, B * Cw, ttnn.float32),  # exactly one round (mechanism cap)
        _cb(CB_RSTD, all_cores, P32, B, ttnn.float32),  # exactly one round (mechanism cap)
        _cb(CB_SCALER, all_cores, T_SCALER, 1, ttnn.bfloat16),
        _cb(CB_SUMSQ_PARTIAL, all_cores, P32, B, ttnn.float32),
    ]
    if Cw > 1:
        cbs.append(_cb(CB_PARTIAL_COLLAPSED, all_cores, P32, B, ttnn.float32))
        cbs.append(_cb(CB_RSTD_HANDOFF, all_cores, P32, B, ttnn.float32))

    if sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor))

    # Per-W-group CBs: sized with that group's core_w_tiles so every CB capacity is an exact
    # multiple of its push/pop quantum (a ragged quantum wraps a CB illegally).
    w_groups = {}
    for r in active_roles:
        w_groups.setdefault(r.core_w_tiles, []).append(r)
    for wc, wroles in sorted(w_groups.items(), reverse=True):
        cr = _core_range_set([(r.x, r.y) for r in wroles])
        if not sharded:
            if input_rm:
                cbs.append(_cb(CB_X_STICKS, cr, t_in, DEPTH_X_STICKS_BLOCKS * B * wc, in_dtype))
                cbs.append(_cb(CB_X_TILES, cr, t_in, B * wc, in_dtype))
                cbs.append(_cb(CB_OUTPUT_TILES, cr, t_out, B * wc, out_dtype))
                cbs.append(_cb(CB_OUT_STICKS, cr, t_out, DEPTH_OUT_STICKS_ROWS * wc, out_dtype))
            else:
                cbs.append(_cb(CB_X_TILES, cr, t_in, DEPTH_X * B * wc, in_dtype))
                cbs.append(_cb(CB_OUTPUT_TILES, cr, t_out, DEPTH_OUT * B * wc, out_dtype))
        if has_gamma:
            cbs.append(_cb(CB_GAMMA_TILES, cr, t_g, wc, gamma.dtype))
            normed_fds = [ttnn.CBFormatDescriptor(buffer_index=CB_NORMED, data_format=ttnn.float32, page_size=P32)]
            if gamma_mode == GAMMA_MODE_RM:
                # cb_gamma_sticks aliases cb_normed's allocation: start-up-only vs per-block lifetimes.
                normed_fds.append(
                    ttnn.CBFormatDescriptor(buffer_index=CB_GAMMA_STICKS, data_format=gamma.dtype, page_size=t_g)
                )
            cbs.append(ttnn.CBDescriptor(total_size=B * wc * P32, core_ranges=cr, format_descriptors=normed_fds))

    # ---------------- semaphores + multicast families ----------------
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_cores, initial_value=0),
    ]
    mcast_helpers = {}
    if Cw > 1:
        mcast_config = ttnn.McastConfig(rotating_sender=False, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
        for g in blocking.groups:
            mcast_helpers[g.index] = ttnn.Mcast2D(device, g.rect, g.root, mcast_config)
        mcast_ct = list(mcast_helpers[0].compile_time_args())
        for g in blocking.groups:
            assert list(mcast_helpers[g.index].compile_time_args()) == mcast_ct, "mcast CT args differ across groups"
    else:
        mcast_ct = [0] * 6  # McastArgs is never instantiated in the NUM_W_SPLITS == 1 build
    root_virtual = {g.index: device.worker_core_from_logical_core(g.root) for g in blocking.groups}

    def blocks_of(role: CoreRole):
        num_blocks = _ceil_div(role.core_row_tiles, B)
        last_rows = role.core_row_tiles - (num_blocks - 1) * B
        return num_blocks, last_rows

    named_common = [
        ("CB_X_TILES", CB_X_TILES),
        ("CB_X_STICKS", CB_X_STICKS),
        ("CB_SCALER", CB_SCALER),
        ("CB_SUMSQ_PARTIAL", CB_SUMSQ_PARTIAL),
        ("CB_PARTIAL_COLLAPSED", CB_PARTIAL_COLLAPSED),
        ("CB_GATHER", CB_GATHER),
        ("CB_RSTD_HANDOFF", CB_RSTD_HANDOFF),
        ("CB_RSTD", CB_RSTD),
        ("CB_GAMMA_TILES", CB_GAMMA_TILES),
        ("CB_GAMMA_STICKS", CB_GAMMA_STICKS),
        ("CB_NORMED", CB_NORMED),
        ("CB_OUTPUT_TILES", CB_OUTPUT_TILES),
        ("CB_OUT_STICKS", CB_OUT_STICKS),
        ("INPUT_RM", 1 if input_rm else 0),
        ("GAMMA_MODE", gamma_mode),
        ("SHARDED", 1 if sharded else 0),
        ("NUM_W_SPLITS", Cw),
        ("SEM_GATHER", SEM_GATHER),
        ("IN_TILE_BYTES", t_in),
        ("IN_ELEM_BYTES", e_in),
        ("OUT_TILE_BYTES", t_out),
        ("OUT_ELEM_BYTES", e_out),
        ("GAMMA_TILE_BYTES", t_g),
        ("GAMMA_ELEM_BYTES", e_g),
        ("P32_BYTES", P32),
    ]

    # ---------------- reader (active cores) ----------------
    reader_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct += list(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    reader_named = named_common + [
        ("IN_PAGE_BYTES", input_tensor.buffer_page_size()),
        ("GAMMA_PAGE_BYTES", gamma.buffer_page_size() if has_gamma else 0),
    ]
    reader_rt = ttnn.RuntimeArgs()
    for r in active_roles:
        num_blocks, last_rows = blocks_of(r)
        reader_rt[r.x][r.y] = [
            input_tensor.buffer_address(),
            gamma.buffer_address() if has_gamma else 0,
            r.row_tile_start,
            num_blocks,
            B,
            last_rows,
            r.w_tile_start,
            r.core_w_tiles,
            Wt,
            Rt,
        ]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=active_cores,
        compile_time_args=reader_ct,
        named_compile_time_args=reader_named,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------------- compute (one descriptor per distinct core_w_tiles: tilize/untilize width is CT) ----
    compute_kernels = []
    for wc, wroles in sorted(w_groups.items(), reverse=True):
        compute_rt = ttnn.RuntimeArgs()
        for r in wroles:
            num_blocks, last_rows = blocks_of(r)
            compute_rt[r.x][r.y] = [
                1 if r.is_root else 0,
                num_blocks,
                B,
                last_rows,
                inv_w_bits,
                eps_bits,
                Rt,
            ]
        compute_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
                core_ranges=_core_range_set([(r.x, r.y) for r in wroles]),
                compile_time_args=[],
                named_compile_time_args=named_common + [("CORE_W_TILES", wc)],
                runtime_args=compute_rt,
                config=compute_kernel_config,
            )
        )

    # ---------------- writer (every program core: passive cores still ack the rstd multicast) ----
    writer_ct = list(mcast_ct) + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for g in blocking.groups:
        num_partials_expected = g.num_active - 1
        for r in g.cores:
            num_blocks, last_rows = blocks_of(r)
            rv = root_virtual[g.index]
            mcast_rt = list(mcast_helpers[g.index].runtime_args(r.coord)) if Cw > 1 else [0] * 4
            writer_rt[r.x][r.y] = [
                output_tensor.buffer_address(),
                r.row_tile_start,
                num_blocks,
                B,
                last_rows,
                r.w_tile_start,
                r.core_w_tiles,
                Wt,
                1 if r.is_active else 0,
                1 if r.is_root else 0,
                r.w_split_index,
                rv.x,
                rv.y,
                num_partials_expected,
            ] + mcast_rt
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        named_compile_time_args=named_common + [("MCAST_CT_BASE", 0), ("MCAST_RT_BASE", 14)],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel] + compute_kernels,
        semaphores=semaphores,
        cbs=cbs,
    )
