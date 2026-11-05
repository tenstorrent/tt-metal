# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_moe_route_dispatch: router scores, top-2, sort into 32-row expert regions and the gathered (tilized)
activation matrix, in one ttnn.generic_op program (regime R1 `nd_push_l1` of op_design.md).

x arrives in the layer norm's ND-sharded L1 layout: token group g (rps tile rows) is held by G cores, member c
holding hidden slice c (ht tiles). Every member computes its partial logits, the G partials are exchanged inside
the group, and every member then computes the same scores and selection. Group leaders send their per-expert
counts to the coordinator, which multicasts the table back; every token core scans it and pushes its tokens'
row slices straight into the owners' shards of `dispatched`, which every core tilizes in place.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; INVALID lives in
eval/golden_tests/nomic_moe_route_dispatch/feature_spec.py.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

# ---------------------------------------------------------------------------
# Model constants and blocking knobs (single source of truth)
# ---------------------------------------------------------------------------

HIDDEN = 768
NUM_EXPERTS = 8
TOP_K = 2
TILE = 32
HIDDEN_TILES = HIDDEN // TILE  # 24
MAX_SPARE_TILE_ROWS = 3
T_MIN_PAD, T_MAX = 1024, 4096
MAX_EXPERTS_ACCEPTED = 32  # one tile column holds every expert (validate: E > 32 is malformed)
MAX_ROWS_PER_SHARD = 4  # packed u16 count table and the fp32 DEST matmul subblock (rps x 1)
GRID_X, GRID_Y = 11, 10
NUM_CORES = GRID_X * GRID_Y
COORDINATOR = NUM_CORES - 1  # receives the count rows, multicasts the table and the landing go
W_SENDER_TOP = NUM_CORES - 2  # sender of W slice c is core W_SENDER_TOP - c
# NoCs the W slice multicasts alternate over: slice c goes out on the writer RISC (NOC1) when c % W_SLICE_NOCS == 1,
# else on the reader (NOC0). Full-grid multicasts on one NoC serialize, so 2 runs two slices at once; 1 = all NOC0.
W_SLICE_NOCS = 2
# Share of a core's tokens (of the rps*32 a shard holds) whose rows the reader RISC pushes; the writer RISC pushes
# the rest (perf lamp L2). Token granular, so rps = 1 shards also use both RISCs / NoCs.
PUSH_READER_SHARE = 0.5
READER_SPLIT_TOKENS = 16  # granule of the reader/writer token split (16 u32 combine_index words = 64 B)
# Bytes of row pushes a RISC keeps in flight (perf lamp L2 / nomic_moe_dispatch's congestion-collapse rule).
PUSH_BYTES_IN_FLIGHT = 4096
# Token offset per group at which a RISC starts its slot walk (perf lamp L7, catalog `bank_stagger`); 0 = lockstep.
PUSH_STAGGER_TOKENS = 0
# NoC write command buffers a pushing RISC round-robins over (1 or 2).
PUSH_CMD_BUFS = 1
# Push whole output pieces (perf lamp L2, member balance): each piece of w tiles is pushed by the group member that
# holds most of it, which first gathers the piece's other tiles from its peers' x shards, so a row costs one write
# per piece instead of one per (slice, piece) intersection. Off: every member pushes its slice's intersections.
PUSH_WHOLE_PIECES = True
# Landing (perf lamp L3): False = global barrier (every pushing RISC increments the coordinator, which multicasts the
# go; every owner then tilizes all its blocks). True = per-owner counted arrival: each pushing RISC, once its writes
# are acked, adds the 64 B row granules it landed in each owner block to that block's semaphore (one atomic per
# (tile row, piece) run, not per write), and an owner tilizes each block as soon as it is complete.
ARRIVAL_PER_BLOCK = False
# combine_weights tiles: False = the group leader writes all of them; True = member c of the group writes every G-th
# tile from tile c, spreading the DRAM writes over the group.
WEIGHTS_SPREAD = True
# A pushed piece's rows: all, or one parity (two members tie on a piece and split its rows).
ROWS_ALL, ROWS_EVEN, ROWS_ODD = 0, 1, 2
# DRAM `dispatched` (regime R4): output blocks (32 rows x w tiles) the tilize -> DRAM-writer CB holds, so the
# tilize of block b+1 overlaps the DRAM writes of block b.
DRAM_OUT_BLOCKS_IN_FLIGHT = 2
FP32_TILE_BYTES = TILE * TILE * 4
BF16_TILE_BYTES = TILE * TILE * 2


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def rows_for(tokens: int, spare_tile_rows: int = 0) -> int:
    """R: every region rounded up to 32 rows always fits, plus the spare tile rows."""
    return TILE * (math.ceil(TOP_K * tokens / TILE) + NUM_EXPERTS - 1) + TILE * spare_tile_rows


def _dispatch():
    import ttnn.operations.nomic_moe_dispatch as dispatch  # noqa: PLC0415

    return dispatch


def output_memory_config(rows, device=None, buffer_type=ttnn.BufferType.L1) -> ttnn.MemoryConfig:
    """The layout `dispatched` is written in: nomic_moe_dispatch's L1 layout, or DRAM interleaved."""
    if buffer_type == ttnn.BufferType.DRAM:
        return ttnn.DRAM_MEMORY_CONFIG
    return _dispatch().output_memory_config(int(rows), device, ttnn.BufferType.L1)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    batch: int
    seq: int
    tokens: int  # T
    t_pad: int
    rows: int  # R
    spare_tile_rows: int
    hidden_split: int  # G
    hidden_tiles: int  # ht = block_hidden_tiles
    rows_per_shard: int  # rps = block_token_tiles
    groups_per_batch: int  # SG
    num_groups: int
    num_token_cores: int  # G * num_groups
    pieces_per_row: int  # k
    block_width_tiles: int  # w
    num_shards: int
    max_blocks_per_core: int
    out_dram: bool  # dispatched DRAM interleaved (R4): tilize streams through a CB to the writer's DRAM writes
    x_interleaved: bool  # R2: x read through a TensorAccessor into a real cb_x_tiles, G = 1
    group_ranges: tuple  # (first global token, real tokens) of every token group (R3: real tokens only)
    push_gather: bool  # a member stages tiles beyond its own slice (gathered from its group peers)
    stage_tiles: int  # width of a staged (untilized) row in tiles, the same on every pushing member
    member_push: tuple  # per member: (first staged tile, ((piece, src B, dst B, bytes, rows), ...))

    def group_tokens(self, group: int) -> tuple[int, int]:
        """(first global token, real tokens) of a token group."""
        return self.group_ranges[group]

    def reader_tokens_of(self, group: int) -> int:
        """Tokens of a group's core the reader RISC selects and pushes (PUSH_READER_SHARE of the core's tile rows),
        a multiple of READER_SPLIT_TOKENS so the two RISCs' combine_index segments stay 64 B aligned."""
        tokens = TILE * math.ceil(self.group_ranges[group][1] / TILE)
        granules = tokens // READER_SPLIT_TOKENS
        return READER_SPLIT_TOKENS * max(0, min(granules, round(PUSH_READER_SHARE * granules)))

    def blocks_of(self, core: int) -> int:
        return max(0, math.ceil((self.num_shards - core) / NUM_CORES))


def _push_layout(hidden_split: int, hidden_tiles: int, pieces: int, width: int) -> tuple[bool, int, tuple]:
    """Which member pushes which part of each output row: (push_gather, stage_tiles, member_push), see Plan."""
    slices = [(m * hidden_tiles, (m + 1) * hidden_tiles) for m in range(hidden_split)]
    spans = [(p * width, (p + 1) * width) for p in range(pieces)]

    def overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    owners = []
    for span in spans:
        best = max(overlap(span, sl) for sl in slices)
        owners.append([m for m, sl in enumerate(slices) if overlap(span, sl) == best])
    if not PUSH_WHOLE_PIECES or all(sum(overlap(span, sl) > 0 for sl in slices) == 1 for span in spans):
        # Intersections: each member pushes its own slice, one write per (slice, piece) intersection.
        member_push = []
        for lo_m, hi_m in slices:
            parts = []
            for p, (lo_p, hi_p) in enumerate(spans):
                lo, hi = max(lo_m, lo_p), min(hi_m, hi_p)
                if lo < hi:
                    parts.append((p, (lo - lo_m) * 64, (lo - lo_p) * 64, (hi - lo) * 64, ROWS_ALL))
            member_push.append((lo_m, tuple(parts)))
        return False, hidden_tiles, tuple(member_push)
    owned = [[] for _ in slices]
    for p, members in enumerate(owners):
        if len(members) > 2:
            raise UnsupportedAxisValue("nomic_moe_route_dispatch: a piece split three ways")
        for i, m in enumerate(members):
            owned[m].append((p, ROWS_ALL if len(members) == 1 else (ROWS_EVEN, ROWS_ODD)[i]))
    member_push, widths = [], set()
    for parts in owned:
        if not parts:
            member_push.append((0, ()))
            continue
        lo_s, hi_s = spans[parts[0][0]][0], spans[parts[-1][0]][1]
        widths.add(hi_s - lo_s)
        member_push.append((lo_s, tuple((p, (spans[p][0] - lo_s) * 64, 0, width * 64, rows) for p, rows in parts)))
    if len(widths) != 1:
        raise UnsupportedAxisValue("nomic_moe_route_dispatch: pushing members stage different widths")
    return True, widths.pop(), tuple(member_push)


def _interleaved_ranges(tokens: int) -> tuple[int, tuple]:
    """R2: x's T_pad/32 tile rows split row-wise over the grid (split_work_to_cores: the first cores take one tile
    row more). (rows_per_shard = the larger per-core count, per-core (first token, real tokens))."""
    tile_rows = math.ceil(tokens / TILE)
    grid = ttnn.CoreCoord(GRID_X, GRID_Y)
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, tile_rows, True)
    counts = [per_core_1] * group_1.num_cores() + [per_core_2] * group_2.num_cores()
    ranges, first = [], 0
    for count in counts:
        ranges.append((first, min(count * TILE, tokens - first)))
        first += count * TILE
    return per_core_1, tuple(ranges)


@lru_cache(maxsize=None)
def _plan(shape: tuple, shard_shape, num_shard_cores: int, spare_tile_rows: int, out_dram: bool) -> Plan:
    """shard_shape None: x interleaved (R2, G = 1, tile rows split over the grid); else x's nd_norm shard (R1)."""
    batch, _, seq, hidden = shape
    tokens = batch * seq
    if shard_shape is None:
        hidden_tiles, hidden_split = HIDDEN_TILES, 1
        rps, group_ranges = _interleaved_ranges(tokens)
        groups_per_batch = 0
        num_groups = num_token_cores = len(group_ranges)
    else:
        shard_h, shard_w = shard_shape[-2], shard_shape[-1]
        if shard_h % TILE or shard_w % TILE or hidden % shard_w:
            raise UnsupportedAxisValue(
                f"nomic_moe_route_dispatch: x shard {shard_shape} is not a tile split of {shape}"
            )
        hidden_tiles = shard_w // TILE
        hidden_split = HIDDEN_TILES // hidden_tiles
        rps = shard_h // TILE
        if hidden_split < 2:
            raise UnsupportedAxisValue("nomic_moe_route_dispatch: an nd_norm shard splits the hidden axis (G >= 2)")
        seq_tiles = math.ceil(seq / TILE)
        groups_per_batch = math.ceil(seq_tiles / rps)
        num_groups = batch * groups_per_batch
        num_token_cores = hidden_split * num_groups
        if num_token_cores != num_shard_cores:
            raise UnsupportedAxisValue("nomic_moe_route_dispatch: x is not one shard a core on the grid")
        group_ranges = tuple(
            (b * seq + sg * rps * TILE, min(rps * TILE, seq - sg * rps * TILE))
            for b in range(batch)
            for sg in range(groups_per_batch)
        )
    if not (1 <= rps <= MAX_ROWS_PER_SHARD):
        raise UnsupportedAxisValue(f"nomic_moe_route_dispatch: {rps} tile rows a core (max {MAX_ROWS_PER_SHARD})")
    if num_token_cores > NUM_CORES:
        raise UnsupportedAxisValue("nomic_moe_route_dispatch: more token cores than the grid holds")
    # A W slice the writer RISC multicasts (slice c with c % W_SLICE_NOCS == 1) must come from a core that holds no
    # tokens: that core's writer would wait for its own multicast, which leaves the source out (the descriptor
    # asserts it). Token cores are grid cores 0 .. num_token_cores - 1, so a shape whose shards reach that sender
    # (5x704: 110 token cores) is refused here and the model runs the three ops instead.
    writer_senders = [W_SENDER_TOP - c for c in range(min(hidden_split, W_SLICE_NOCS)) if c % W_SLICE_NOCS == 1]
    if any(sender < num_token_cores for sender in writer_senders):
        raise UnsupportedAxisValue(
            f"nomic_moe_route_dispatch: {num_token_cores} token cores reach the writer-sent W slice sender "
            f"(core {min(writer_senders)})"
        )
    rows = rows_for(tokens, spare_tile_rows)
    dispatch_plan = _dispatch().plan(rows)
    push_gather, stage_tiles, member_push = _push_layout(
        hidden_split, hidden_tiles, dispatch_plan.pieces_per_row, dispatch_plan.block_width_tiles
    )
    return Plan(
        batch=batch,
        seq=seq,
        tokens=tokens,
        t_pad=TILE * math.ceil(tokens / TILE),
        rows=rows,
        spare_tile_rows=spare_tile_rows,
        hidden_split=hidden_split,
        hidden_tiles=hidden_tiles,
        rows_per_shard=rps,
        groups_per_batch=groups_per_batch,
        num_groups=num_groups,
        num_token_cores=num_token_cores,
        pieces_per_row=dispatch_plan.pieces_per_row,
        block_width_tiles=dispatch_plan.block_width_tiles,
        num_shards=dispatch_plan.num_shards,
        max_blocks_per_core=dispatch_plan.max_blocks_per_core,
        out_dram=out_dram,
        x_interleaved=shard_shape is None,
        group_ranges=group_ranges,
        push_gather=push_gather,
        stage_tiles=stage_tiles,
        member_push=member_push,
    )


def plan(input_tensor, spare_tile_rows: int = 0, out_buffer_type=ttnn.BufferType.L1) -> Plan:
    """The single source of the work split, read off x's ND shard spec (or the interleaved tile-row split) and
    nomic_moe_dispatch.plan(R)."""
    spec = input_tensor.memory_config().nd_shard_spec
    if input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED or spec is None:
        shard_shape, num_shard_cores = None, 0
    else:
        shard_shape = tuple(spec.shard_shape)
        num_shard_cores = len(ttnn.corerange_to_cores(spec.grid, None, True))
    return _plan(
        tuple(input_tensor.shape),
        shard_shape,
        num_shard_cores,
        int(spare_tile_rows),
        out_buffer_type == ttnn.BufferType.DRAM,
    )


def circular_buffer_bytes(input_shape, device, spare_tile_rows: int = 0, buffer_type=ttnn.BufferType.L1) -> int:
    """CB bytes a core for the call (the production nd_norm placement); the sharded CBs are the tensors.
    buffer_type: where `dispatched` goes (DRAM adds the streaming output CB)."""
    from ttnn.operations.nomic_residual_layer_norm import input_memory_config  # noqa: PLC0415

    from .nomic_moe_route_dispatch_program_descriptor import cb_bytes  # noqa: PLC0415

    shape = tuple(int(d) for d in input_shape)
    spec = input_memory_config(shape, device).nd_shard_spec
    num_shard_cores = len(ttnn.corerange_to_cores(spec.grid, None, True))
    return cb_bytes(
        _plan(
            shape,
            tuple(spec.shard_shape),
            num_shard_cores,
            int(spare_tile_rows),
            buffer_type == ttnn.BufferType.DRAM,
        )
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """S % 32 == 0 ("tile_aligned") or not ("ragged"); inputs[0] is x's shape."""
    return "tile_aligned" if inputs[0][-2] % TILE == 0 else "ragged"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "ragged"],
    "fp32_dest_acc_en": [True],
    "in_placement": ["nd_norm", "l1_interleaved", "dram_interleaved"],
    "out_buffer_type": [ttnn.BufferType.L1, ttnn.BufferType.DRAM],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _in_placement(tensor) -> str:
    memory = tensor.memory_config()
    if memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if memory.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    from ttnn.operations.nomic_residual_layer_norm import input_memory_config  # noqa: PLC0415

    try:
        if memory == input_memory_config(tuple(tensor.shape), tensor.device()):
            return "nd_norm"
    except NotImplementedError:
        pass
    return "other"


def _check_malformed(input_tensor, weight, spare_tile_rows):
    shape, w_shape = tuple(input_tensor.shape), tuple(weight.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"nomic_moe_route_dispatch: input_tensor shape {shape} is not (B, 1, S, H)")
    if len(w_shape) != 2 or shape[-1] != w_shape[0]:
        raise ValueError(f"nomic_moe_route_dispatch: weight shape {w_shape} is not (H, E) with H = {shape[-1]}")
    if w_shape[1] > MAX_EXPERTS_ACCEPTED or w_shape[1] < 1:
        raise ValueError(f"nomic_moe_route_dispatch: E = {w_shape[1]} must be in 1..{MAX_EXPERTS_ACCEPTED}")
    if isinstance(spare_tile_rows, bool) or not isinstance(spare_tile_rows, int):
        raise ValueError(f"nomic_moe_route_dispatch: spare_tile_rows {spare_tile_rows!r} is not an int")
    if not 0 <= spare_tile_rows <= MAX_SPARE_TILE_ROWS:
        raise ValueError(
            f"nomic_moe_route_dispatch: spare_tile_rows {spare_tile_rows} outside 0..{MAX_SPARE_TILE_ROWS}"
        )
    if input_tensor.device() is None or weight.device() is None or input_tensor.device() != weight.device():
        raise ValueError("nomic_moe_route_dispatch: input_tensor and weight must be on the same device")


def validate(input_tensor, weight, *, spare_tile_rows=0, compute_kernel_config=None, memory_config=None):
    _check_malformed(input_tensor, weight, spare_tile_rows)

    config = compute_kernel_config or default_compute_kernel_config()
    out_memory = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "in_placement": _in_placement(input_tensor),
        "out_buffer_type": out_memory.buffer_type,
    }
    shapes = (tuple(input_tensor.shape), tuple(weight.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_moe_route_dispatch: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_moe_route_dispatch: unsupported combination (refinement candidate): {exc}")

    # Contract refusals (op_design.md, Host contract).
    batch, _, seq, hidden = shapes[0]
    experts = shapes[1][1]
    if hidden != HIDDEN or experts != NUM_EXPERTS:
        raise UnsupportedAxisValue(f"nomic_moe_route_dispatch: (H, E) = ({hidden}, {experts}) is not (768, 8)")
    t_pad = TILE * math.ceil(batch * seq / TILE)
    if not T_MIN_PAD <= t_pad <= T_MAX:
        raise UnsupportedAxisValue(f"nomic_moe_route_dispatch: T_pad = {t_pad} outside {T_MIN_PAD}..{T_MAX}")
    if batch > 1 and seq % TILE:
        raise UnsupportedAxisValue("nomic_moe_route_dispatch: S % 32 != 0 is defined for B = 1 only")
    if (
        weight.dtype != ttnn.float32
        or weight.layout != ttnn.TILE_LAYOUT
        or weight.memory_config() != ttnn.DRAM_MEMORY_CONFIG
    ):
        raise UnsupportedAxisValue("nomic_moe_route_dispatch: weight must be float32 TILE DRAM interleaved")
    grid = input_tensor.device().compute_with_storage_grid_size()
    if (grid.x, grid.y) != (GRID_X, GRID_Y):
        raise UnsupportedAxisValue(f"nomic_moe_route_dispatch: the plan is built for an {GRID_X}x{GRID_Y} grid")
    plan(
        input_tensor, spare_tile_rows, out_memory.buffer_type
    )  # rps > 4, shard map and nomic_moe_dispatch.plan(R) refusals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _l1_room(device, p: Plan, reserved_bytes: int) -> bool:
    """Whether the op's CBs plus `reserved_bytes` a bank fit between the allocator base and the lowest live buffer."""
    from .nomic_moe_route_dispatch_program_descriptor import cb_bytes  # noqa: PLC0415

    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    live = [int(block["address"]) for block in view.block_table if block["allocated"] == "yes"]
    lowest = min(live, default=view.total_bytes_per_bank)
    return ttnn.get_allocator_base_address(device, ttnn.BufferType.L1) + cb_bytes(p) + reserved_bytes <= lowest


def _refuse_l1():
    raise UnsupportedAxisValue("nomic_moe_route_dispatch: the output and circular buffers do not fit below live L1")


# ---------------------------------------------------------------------------
# Call-state cache (host time)
# ---------------------------------------------------------------------------

# One entry per call-state key: everything validate(), the plan and the program descriptor depend on beyond buffer
# addresses (x's and W's shapes, dtypes, layouts and memory configs, the spare tile rows, the output buffer type, the
# compute config, the device and the ablation defines). validate() and the descriptor build run on a key's first
# call only; every call rebinds the cached descriptor to its own buffers (bind_call), so it never rebuilds the
# 110-core program, which took ~600 of the ~800 us of host time a call. The live-L1 checks run on every call.
# Knobs are read when an entry is built: after patching one, clear this cache.
_CALL_CACHE = {}


@dataclass
class _CallState:
    plan: Plan
    compute: object  # ttnn.ComputeConfigDescriptor
    out_buffer_type: object  # ttnn.BufferType of `dispatched`
    out_l1_bytes: int  # a bank's share of an L1 `dispatched`, 0 in DRAM
    descriptor: object = None  # ttnn.ProgramDescriptor, built on the key's first call (needs the tensors)
    binding: object = None  # the descriptor's address slots (nomic_moe_route_dispatch_program_descriptor.CallBinding)


def _compute_key(config):
    if config is None:
        return None
    return (
        config.math_fidelity,
        config.fp32_dest_acc_en,
        config.math_approx_mode,
        config.dst_full_sync_en,
        config.bfp8_pack_precise,
        tuple(config.unpack_to_dest_mode),
    )


def _call_key(input_tensor, weight, spare_tile_rows, compute_kernel_config, out_buffer_type) -> tuple:
    return (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        repr(input_tensor.memory_config()),
        tuple(weight.shape),
        weight.dtype,
        weight.layout,
        repr(weight.memory_config()),
        spare_tile_rows,
        out_buffer_type,
        _compute_key(compute_kernel_config),
        input_tensor.device().id(),
        os.environ.get("NOMIC_MOE_ROUTE_DISPATCH_ABLATE", ""),
    )


def _call_state(input_tensor, weight, spare_tile_rows, compute_kernel_config, memory_config) -> _CallState:
    """The key's cached state; validate() runs on the key's first call. The malformed-call checks run every call."""
    _check_malformed(input_tensor, weight, spare_tile_rows)
    out_buffer_type = (memory_config or ttnn.L1_MEMORY_CONFIG).buffer_type
    key = _call_key(input_tensor, weight, spare_tile_rows, compute_kernel_config, out_buffer_type)
    state = _CALL_CACHE.get(key)
    if state is None:
        validate(
            input_tensor,
            weight,
            spare_tile_rows=spare_tile_rows,
            compute_kernel_config=compute_kernel_config,
            memory_config=memory_config,
        )
        p = plan(input_tensor, spare_tile_rows, out_buffer_type)
        state = _CallState(
            plan=p,
            compute=compute_kernel_config or default_compute_kernel_config(),
            out_buffer_type=out_buffer_type,
            out_l1_bytes=0 if p.out_dram else p.max_blocks_per_core * p.block_width_tiles * BF16_TILE_BYTES,
        )
        _CALL_CACHE[key] = state
    return state


def nomic_moe_route_dispatch(
    input_tensor, weight, *, spare_tile_rows=0, compute_kernel_config=None, memory_config=None
):
    state = _call_state(input_tensor, weight, spare_tile_rows, compute_kernel_config, memory_config)
    from .nomic_moe_route_dispatch_program_descriptor import (  # noqa: PLC0415
        CallBinding,
        bind_call,
        create_program_descriptor,
    )

    out_buffer_type = state.out_buffer_type
    p = state.plan
    device = input_tensor.device()
    # An L1 dispatched is allocated top-down just below the live buffers; refuse before an allocation that cannot fit.
    if not _l1_room(device, p, state.out_l1_bytes):
        _refuse_l1()
    dram = ttnn.DRAM_MEMORY_CONFIG
    dispatched = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, p.rows, HIDDEN]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config(p.rows, device, out_buffer_type),
    )
    combine_index = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TOP_K, p.t_pad]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, dram
    )
    combine_weights = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, TOP_K, p.t_pad, 1]), ttnn.float32, ttnn.TILE_LAYOUT, device, dram
    )
    expert_counts = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, NUM_EXPERTS]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, dram
    )
    outputs = [dispatched, combine_index, combine_weights, expert_counts]
    if not _l1_room(device, p, 0):
        for tensor in outputs:
            ttnn.deallocate(tensor)
        _refuse_l1()
    if state.descriptor is None:
        state.descriptor = create_program_descriptor(p, input_tensor, weight, *outputs, state.compute)
        state.binding = CallBinding(state.descriptor, p, input_tensor)
    # Every call rebinds the addresses (a cached descriptor never carries a previous call's).
    bind_call(state.binding, input_tensor, weight, *outputs)
    ttnn.generic_op([input_tensor, weight, *outputs], state.descriptor)
    return tuple(outputs)
