# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for max_pool2d (op_design.md: Blocking Model, Work Distribution, Circular Buffers).

The two ideas the program rests on:

  stream tiles   For one window tap, the input sticks a core's consecutive output positions need, laid end to end
                 (`stick_pitch` elements per position), have the layout of the core's output rows. Both streams are
                 cut into stream tiles of 1024 elements; output stream tile u is the elementwise max over the K taps
                 of tap stream tile (u, k). No tilize, no untilize, no within-tile reduce.
  staging        A core's receptive field (the input rows its positions need) is read into a scratchpad once per
                 block (or used where it lies, when it is the core's own width shard), and the K taps are gathered
                 from there by core-local copies. A padded tap re-reads a real tap of the same window, so padding is
                 never stored and never wins (rule R1).

Work split (op_design.md "Work Distribution"): the output shards rule R9 fixes (channels for a width-sharded or
interleaved input, position ranges for a height-sharded one) are the primary split; each shard's positions are then
cut into `position_splits` sub-ranges of whole rounds (C stream tiles), sub-range 0 on the shard's owner and the
others on spare cores. Inside a core the thread plan THREAD_PLAN = (R, C, W) deals stream tiles to NEOs: in block b,
NEO t owns tiles [b*C*bt + t*bt_b, + bt_b); reader thread t feeds NEO t (and NEOs t+R, ...), writer thread w drains
NEOs w, w+W, ... Every NEO takes the same count; a NEO past the last real tile recomputes it (padded round).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

from eval import memory

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------
# Knobs (op_design.md: every block factor, depth and assignment is one named host constant)
# ---------------------------------------------------------------------------

# Thread plan (R readers, C NEOs, W writers) per cluster. Bound by compute (K unpack + datacopy + SFPU max passes
# per output tile), so C = 4; the op reads K tiles per tile it writes, so R = 4 and W = 2 (the target page).
THREAD_PLAN = (4, 4, 2)
# The target page's cluster model (Python has no query for these).
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 is the remapper
NEOS = 4
DFB_MAX_RING_BYTES = 1_048_560  # one ring, compute side (1 MiB - 16 B)

TAP_RING_DEPTH = 2  # tap windows per NEO in flight in `window_taps`: the reader lands tile u+1 while u is folded
OUT_RING_DEPTH = 2  # entries per NEO in `pooled_tiles`: the NEO packs tile u+1 while the writer stores u
STAGING_DEPTH = 1  # staging passes in flight; 1 is the built value (2 = the overlap lamp of op_design.md)
MIN_ROUNDS_PER_CORE = 1  # position sub-ranges hold at least this many rounds (C stream tiles)
MAX_POSITION_SPLITS = None  # cap on sub-ranges per output shard; None = as many as the grid allows, 1 = owners only
L1_RESERVE_BYTES = 65_536  # L1 left to everything else on a core
_STAGING_BUDGET_OVERRIDE = None  # tests lower the staging budget to pin the staged-streamed regime

TILE = 32
TILE_ELEMS = TILE * TILE  # elements of one stream tile (one ring entry)


def _ceil_div(a, b):
    return -(-a // b)


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in (1, 2, 4), R + W <= {FREE_DM_HARTS}, R | C, W | C")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    N: int
    H: int
    W: int
    C: int
    kh: int
    kw: int
    sh: int
    sw: int
    ph: int
    pw: int
    dh: int
    dw: int
    H_out: int
    W_out: int

    @classmethod
    def of(cls, n, h, w, c, kernel, stride, padding, dilation):
        def pooled(size, k, s, p, d):
            return (size + 2 * p - d * (k - 1) - 1) // s + 1

        return cls(n, h, w, c, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1],
                   dilation[0], dilation[1],
                   pooled(h, kernel[0], stride[0], padding[0], dilation[0]),
                   pooled(w, kernel[1], stride[1], padding[1], dilation[1]))

    @property
    def P(self):
        return self.N * self.H_out * self.W_out

    @property
    def K(self):
        return self.kh * self.kw

    @property
    def kh_eff(self):
        return self.dh * (self.kh - 1) + 1

    def receptive_sticks(self, q0, q1):
        """[s_lo, s_hi): the input rows (sticks) output positions [q0, q1) read, as whole image rows. The same
        formula is in the reader kernel (receptive_field)."""
        img0, rem0 = divmod(q0, self.H_out * self.W_out)
        img1, rem1 = divmod(q1 - 1, self.H_out * self.W_out)
        i0, i1 = rem0 // self.W_out, rem1 // self.W_out
        r_lo = max(0, i0 * self.sh - self.ph)
        r_hi = min(self.H - 1, i1 * self.sh - self.ph + self.kh_eff - 1)
        return (img0 * self.H + r_lo) * self.W, (img1 * self.H + r_hi + 1) * self.W


# ---------------------------------------------------------------------------
# Output placement (rule R9) and the output shards
# ---------------------------------------------------------------------------


@dataclass
class Shard:
    core: object  # ttnn.CoreCoord of the shard's owner
    index: int  # shard index within the output's shard grid
    pos0: int  # first output position (row)
    npos: int  # output positions in the shard
    chan0: int  # first channel
    in_core: object = None  # width-sharded input: the core holding input shard `index` (may differ from `core`)


@dataclass
class OutputPlan:
    spec: object  # ttnn.TensorSpec of the output
    in_sharding: str  # "interleaved" | "height" | "width" (the input's)
    c_core: int  # channels per shard
    pitch_bytes: int  # bytes per output row in a shard = bytes per position in the streams
    out_page_stride: int  # page ids between consecutive rows of one shard
    shards: list = field(default_factory=list)
    n_in_width_shards: int = 1  # input pages per row (width-sharded row-major input)


def _interleaved_width_cores(channels, total_cores):
    """n_w: the largest divisor of C/32 that is at most the grid's core count (C % 32 != 0: one core). Every
    shard then holds a whole number of 32-channel runs, which keeps DRAM reads 64-B aligned."""
    if channels % TILE:
        return 1
    runs = channels // TILE
    return max(d for d in range(1, min(runs, total_cores) + 1) if runs % d == 0)


def plan_output(x, geo: Geometry, device) -> OutputPlan:
    """Where the result lives (rule R9) and the output shards: height-sharded over a height-sharded input's cores
    (ceil(P / n) rows each, dealt in order), width-sharded over a width-sharded input's cores (all rows, its own
    channels), width-sharded over n_w cores of the op's choosing for an interleaved input."""
    in_sharding = memory.memory_axes(x)["sharding"]
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    esize = x.element_size()
    l1_align = ttnn.get_l1_alignment()
    P, C = geo.P, geo.C

    if in_sharding in ("height", "width"):
        in_spec = x.memory_config().shard_spec
        shard_grid = in_spec.grid
        # Rule R9: the result's shards are row-major on the input's grid, whatever the input's orientation; input
        # shard g sits on in_cores[g] (the input's own order), output shard g on cores[g].
        in_cores = ttnn.corerange_to_cores(shard_grid, None,
                                           in_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        cores = ttnn.corerange_to_cores(shard_grid, None, True)
    if in_sharding == "height":
        n = len(cores)
        rows = _ceil_div(P, n)
        c_core, layout, shard_shape = C, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [rows, C]
        shards = [Shard(cores[s], s, s * rows, min(P, (s + 1) * rows) - s * rows, 0)
                  for s in range(n) if s * rows < P]
        out_page_stride = 1
    else:
        if in_sharding == "width":
            c_core = int(in_spec.shape[1])
            n = _ceil_div(C, c_core)
        else:
            n = _interleaved_width_cores(C, total_cores)
            c_core = C // n
            shard_grid = ttnn.num_cores_to_corerangeset(n, grid, True)
            cores = ttnn.corerange_to_cores(shard_grid, None, True)
            in_cores = cores  # unused: an interleaved input has no shards
        layout, shard_shape = ttnn.TensorMemoryLayout.WIDTH_SHARDED, [P, c_core]
        shards = [Shard(cores[g], g, 0, P, g * c_core, in_cores[g]) for g in range(n)]
        out_page_stride = n
    out_config = ttnn.MemoryConfig(layout, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR))
    spec = ttnn.TensorSpec(ttnn.Shape([1, 1, P, C]), x.dtype, ttnn.ROW_MAJOR_LAYOUT, out_config)
    n_in_width = len(in_cores) if in_sharding == "width" else 1
    return OutputPlan(spec, in_sharding, c_core, _align_up(c_core * esize, l1_align), out_page_stride, shards,
                      n_in_width)


# ---------------------------------------------------------------------------
# Assignments: output shards x whole-round position sub-ranges
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    core: object
    shard: Shard
    pos0: int
    npos: int
    in_place: bool  # the core holds the receptive field in its own input shard (in-place-shard regime)
    num_tiles: int = 0  # T_core: stream tiles of the assignment
    block_tiles: int = 0  # stream tiles per NEO per staging block
    staging_bytes: int = 0


def split_assignments(out: OutputPlan, device, C, esize):
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    tile_bytes = TILE_ELEMS * esize
    t_max = max(_ceil_div(s.npos * out.pitch_bytes, tile_bytes) for s in out.shards)
    splits = max(1, min(total_cores // len(out.shards), _ceil_div(t_max, C * MIN_ROUNDS_PER_CORE)))
    if MAX_POSITION_SPLITS is not None:
        splits = max(1, min(splits, MAX_POSITION_SPLITS))
    # A whole round is C stream tiles; a cut must fall on a position boundary and on a round boundary.
    round_bytes = C * tile_bytes
    quantum = round_bytes // math.gcd(round_bytes, out.pitch_bytes)  # positions per quantum
    owners = {(s.core.x, s.core.y) for s in out.shards}
    spare = [c for c in ttnn.corerange_to_cores(ttnn.num_cores_to_corerangeset(total_cores, grid, True), None, True)
             if (c.x, c.y) not in owners]
    width_input = out.in_sharding == "width"
    assignments = []
    for s in out.shards:
        nq = _ceil_div(s.npos, quantum)
        parts = min(splits, nq)
        base, extra = divmod(nq, parts)
        start = 0
        for k in range(parts):
            quanta = base + (1 if k < extra else 0)
            npos = min(s.npos - start, quanta * quantum)
            core = s.core if k == 0 else spare.pop(0)
            # In place only where the output owner also holds input shard s (a col-major input puts it elsewhere).
            in_place = width_input and k == 0 and (s.in_core.x, s.in_core.y) == (core.x, core.y)
            assignments.append(Assignment(core, s, s.pos0 + start, npos, in_place))
            start += npos
    return assignments


# ---------------------------------------------------------------------------
# Blocks and staging
# ---------------------------------------------------------------------------


def block_position_ranges(a: Assignment, pitch_elems, C, block_tiles):
    """[(q0, q1)] absolute output positions each staging block of `a` covers. The reader kernel walks the same
    blocks: block b holds tiles [b*C*bt, min(T, b*C*bt + C*bt_b)), bt_b = min(bt, Q - b*bt)."""
    T = a.num_tiles
    Q = _ceil_div(T, C)
    ranges = []
    for b in range(_ceil_div(Q, block_tiles)):
        u0 = b * C * block_tiles
        u1 = min(T, u0 + C * min(block_tiles, Q - b * block_tiles))
        e0, e1 = u0 * TILE_ELEMS, min(u1 * TILE_ELEMS, a.npos * pitch_elems)
        ranges.append((a.pos0 + e0 // pitch_elems, a.pos0 + _ceil_div(e1, pitch_elems)))
    return ranges


def staging_bytes(geo: Geometry, a: Assignment, pitch_elems, C, block_tiles, *, tile_input, in_tile_bytes,
                  channel_tiles, pitch_bytes):
    """The largest receptive field over the blocks of `a`, in bytes of the staging scratchpad."""
    worst = 0
    for q0, q1 in block_position_ranges(a, pitch_elems, C, block_tiles):
        s_lo, s_hi = geo.receptive_sticks(q0, q1)
        if tile_input:
            size = ((s_hi - 1) // TILE - s_lo // TILE + 1) * channel_tiles * in_tile_bytes
        else:
            size = (s_hi - s_lo) * pitch_bytes
        worst = max(worst, size)
    return worst


def taps_per_batch(K, C, tile_bytes):
    """The largest divisor of K whose double-buffered window fits one ring (target page's ring cap)."""
    cap = max(1, DFB_MAX_RING_BYTES // (TAP_RING_DEPTH * C * tile_bytes))
    return max(d for d in range(1, min(K, cap) + 1) if K % d == 0)


def _staging_budget(ring_bytes):
    if _STAGING_BUDGET_OVERRIDE is not None:
        return _STAGING_BUDGET_OVERRIDE
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES - ring_bytes


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _dm_config(make, explicit_rings):
    # Explicit sync for the rings this kernel assembles (the reader: many core-local copies per entry) or writes out
    # in exact lengths (the writer): opt them out of implicit sync; every NoC transfer then carries trid 0.
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------


def create_program_artifacts(x, out, geo: Geometry, out_plan: OutputPlan, config, *, threads=None):
    threads = THREAD_PLAN if threads is None else tuple(threads)
    validate_thread_plan(threads)
    if STAGING_DEPTH != 1:
        raise NotImplementedError("max_pool2d: only staging_depth = 1 is built")
    R, C, W = threads
    device = x.device()
    esize = x.element_size()
    tile_bytes = TILE_ELEMS * esize  # one stream tile = one ring entry
    pitch_bytes = out_plan.pitch_bytes
    pitch_elems = pitch_bytes // esize
    tile_input = x.layout == ttnn.TILE_LAYOUT
    in_tile_bytes = x.buffer_page_size() if tile_input else 0
    channel_tiles = out_plan.c_core // TILE if tile_input else 0
    in_ct_total = _ceil_div(int(x.padded_shape[-1]), TILE) if tile_input else 0
    if tile_input and out_plan.c_core % TILE:
        raise NotImplementedError("max_pool2d: a tile-layout channel run must be a multiple of 32")

    K = geo.K
    tpb = taps_per_batch(K, C, tile_bytes)
    taps_entries = TAP_RING_DEPTH * tpb * C
    out_entries = OUT_RING_DEPTH * C
    ring_bytes = (taps_entries + out_entries) * tile_bytes
    budget = _staging_budget(ring_bytes)

    assignments = split_assignments(out_plan, device, C, esize)
    max_staging = 0
    for a in assignments:
        a.num_tiles = _ceil_div(a.npos * pitch_elems, TILE_ELEMS)
        Q = _ceil_div(a.num_tiles, C)
        if a.in_place:
            a.block_tiles = Q  # nothing to stage: the whole assignment is one block
            continue

        def size(bt, a=a):
            return staging_bytes(geo, a, pitch_elems, C, bt, tile_input=tile_input, in_tile_bytes=in_tile_bytes,
                                 channel_tiles=channel_tiles, pitch_bytes=pitch_bytes)

        bt = Q
        while bt > 1 and size(bt) > budget:
            bt -= 1
        if size(bt) > budget:
            raise RuntimeError(f"max_pool2d: one stream tile's receptive field ({size(bt)} B) exceeds L1 ({budget} B)")
        a.block_tiles, a.staging_bytes = bt, size(bt)
        max_staging = max(max_staging, a.staging_bytes)

    in_sharding = out_plan.in_sharding
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    geometry = {"H": geo.H, "W": geo.W, "H_out": geo.H_out, "W_out": geo.W_out, "kh": geo.kh, "kw": geo.kw,
                "sh": geo.sh, "sw": geo.sw, "ph": geo.ph, "pw": geo.pw, "dh": geo.dh, "dw": geo.dw}
    reader_ct = {
        **plan, **geometry,
        "tile_bytes": tile_bytes,
        "esize": esize,
        "pitch_bytes": pitch_bytes,
        "taps_per_batch": tpb,
        "tile_input": int(tile_input),
        "in_tile_bytes": in_tile_bytes,
        "channel_tiles": channel_tiles,
        "in_ct_total": in_ct_total,
        "in_page_stride": out_plan.n_in_width_shards,
        "input_sharded": int(in_sharding != "interleaved"),
        "input_width_sharded": int(in_sharding == "width"),
    }
    compute_ct = {**plan, "taps": K}
    writer_ct = {**plan, "tile_bytes": tile_bytes, "pitch_bytes": pitch_bytes}

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "max_pool2d_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["window_taps"]),
        num_threads=R,
        dfb_bindings=[E.producer_of("window_taps", "window_taps")],
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="staging", accessor_name="staging")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=reader_ct,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=[
            "pos_start", "pos_count", "num_tiles", "block_tiles", "in_page_col", "in_chan_offset", "in_ct0",
            "in_place"]),
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "max_pool2d_compute.cpp"),
        # The fidelity is set on every compute kernel; math_approx_mode is not read (R4).
        hw_config=E.ComputeHardwareConfig(fpu_math_fidelity=config.math_fidelity,
                                          enable_32_bit_dest=bool(config.fp32_dest_acc_en)),
        num_threads=C,
        dfb_bindings=[E.consumer_of("window_taps", "window_taps"), E.producer_of("pooled_tiles", "pooled_tiles")],
        compile_time_args=compute_ct,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "max_pool2d_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, ["pooled_tiles"]),
        num_threads=W,
        dfb_bindings=[E.consumer_of("pooled_tiles", "pooled_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=writer_ct,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=[
            "out_page", "pos_count", "num_tiles", "block_tiles"]),
    )
    dfbs = [
        E.DataflowBufferSpec(unique_id="window_taps", entry_size=tile_bytes, num_entries=taps_entries,
                             data_format_metadata=x.dtype),
        E.DataflowBufferSpec(unique_id="pooled_tiles", entry_size=tile_bytes, num_entries=out_entries,
                             data_format_metadata=out.dtype),
    ]
    # DM-only scratchpad: raw input bytes, no format metadata. Allocated on every engaged core (unused on in-place
    # cores); at least one alignment unit so the binding exists when every core is in place.
    scratchpads = [E.ScratchpadSpec(unique_id="staging",
                                    size_per_node=max(max_staging, ttnn.get_l1_alignment()))]
    cores = [a.core for a in assignments]
    core_set = _core_range_set(cores)
    spec = E.ProgramSpec(
        name="max_pool2d",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        scratchpads=scratchpads,
        tensor_parameters=[E.TensorParameter(unique_id="input", spec=x.spec),
                           E.TensorParameter(unique_id="output", spec=out.spec)],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )

    def per_core(fn):
        return {a.core: int(fn(a)) for a in assignments}

    width_input = in_sharding == "width"
    height_output = in_sharding == "height"
    reader_args = {
        "pos_start": per_core(lambda a: a.pos0),
        "pos_count": per_core(lambda a: a.npos),
        "num_tiles": per_core(lambda a: a.num_tiles),
        "block_tiles": per_core(lambda a: a.block_tiles),
        # row-major: input page of stick s = s * in_page_stride + in_page_col, channels at in_chan_offset bytes
        "in_page_col": per_core(lambda a: a.shard.index if width_input else 0),
        "in_chan_offset": per_core(lambda a: 0 if width_input else a.shard.chan0 * esize),
        # tile layout: first channel tile of the run
        "in_ct0": per_core(lambda a: a.shard.chan0 // TILE if tile_input else 0),
        "in_place": per_core(lambda a: a.in_place),
    }
    writer_args = {
        # the first output page of the assignment: its rows are contiguous in the owner's shard
        "out_page": per_core(lambda a: a.pos0 * out_plan.out_page_stride + (0 if height_output else a.shard.index)),
        "pos_count": reader_args["pos_count"],
        "num_tiles": reader_args["num_tiles"],
        "block_tiles": reader_args["block_tiles"],
    }
    run_args = E.ProgramRunArgs(kernel_run_args=[
        E.KernelRunArgs(kernel="reader", runtime_arg_values=reader_args),
        E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": reader_args["num_tiles"]}),
        E.KernelRunArgs(kernel="writer", runtime_arg_values=writer_args),
    ])
    return spec, run_args, {"input": 0, "output": 1}
