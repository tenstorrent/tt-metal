# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_moe_combine: the combine of nomic-embed-text-v2-moe's MoE layers (op_design.md).

    out[t, :] = round_bf16( b + W[0,0,t,0] * Y[0,0,I[0,t],:] + W[0,1,t,0] * Y[0,0,I[1,t],:] )

One ttnn.generic_op program, regime `owner_computes` (op_design.md § Blocking Model): each core of the output's
ND-shard grid computes exactly its own shard and packs it straight into that shard (the output CB is the shard).
Its two data-movement RISCs each gather one choice's row segments (choice 0 on RISCV_0 / NOC_0, choice 1 on
RISCV_1 / NOC_1); no RISC writes anything over the NoC.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; INVALID lives in
eval/golden_tests/nomic_moe_combine/feature_spec.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Contract constants and blocking knobs (single source of truth)
# ---------------------------------------------------------------------------

HIDDEN = 768
TILE = 32
HIDDEN_TILES_TOTAL = HIDDEN // TILE  # 24
TOP_K = 2
TOKENS_MIN, TOKENS_MAX = 1000, 4096
ROWS_MIN, ROWS_MAX = 2240, 8512
GRID = (11, 10)
# One fp32 bias copy per DRAM bank: core i reads copy i % BIAS_COPIES (spreads the bias reads over every bank).
BIAS_COPIES = 8
# Per-core L1 the op's own CBs may take beyond the io tensors (prompt: Memory and L1).
CB_L1_BUDGET = 384 * 1024
# Mechanism cap (op_design § Mechanism caps): the bias slice (W * 32 fp32) is staged in faces 2-3 of bias tile 0
# (2 KB) before its placement reads, so the shard width W (tiles) is at most 2048 / (32 * 4) = 16.
MAX_SHARD_COL_TILES = 16

# Reads in flight per DM RISC (perf lamp L1): BATCH_READS reads a batch under BATCH_DEPTH rotating transaction ids.
# BATCH_READS must divide 32 (a tile row's reads per choice) so a tile row ends on a batch boundary.
# Measured (us, 8x512 / 4x512 / 2x512): 4x2 16.55 / 10.09 / 7.76, 8x2 16.43 / 9.66 / 7.27, 4x4 17.1 / 10.3 / 7.6,
# 8x3 17.2 / 15.0 / 7.6, 4x3 16.9 / 10.3 / 7.7 -> 8x2.
BATCH_READS = 8
BATCH_DEPTH = 2
# Mechanism cap (reader kernel's transaction-id layout): ids 1..BATCH_DEPTH carry the row batches, then one id each
# for the index, the bias staging read and the bias placement reads (SIDE_TRIDS), then one id per tile row of the
# shard for its weight tile. NoC transaction ids are 4-bit (1..NOC_MAX_TRID), so BATCH_DEPTH + SIDE_TRIDS + r <= 15
# (the kernel static_asserts the same bound; validate() refuses a plan that would trip it).
NOC_MAX_TRID = 15
SIDE_TRIDS = 3
# Rows CB depth in tile rows (perf lamp L6). None = the whole shard (r): the reader never waits on CB space.
ROWS_DEPTH_TILE_ROWS = None
# Compute realization (op_design § Key Risks). True = R1: tilize inside the multiply's unpack (unpack_tilizeA_B x
# ELWMUL against expanded weights). False = R2: fast tilize into cb_rows_tilized + mul_tiles_bcast_cols on the raw
# weights. Both correct; R1 measured compute-bound at 8x512 (32.7 us, tilize-A unpacks one 16-datum row per
# instruction) against R2 (see the commit log), so R2 is the default.
FUSED_TILIZE = False
# DEST group size in tiles (perf lamp L3). 0 = DEST_AUTO_LIMIT (4 fp32 tiles at half sync, 8 at full sync).
DEST_GROUP_TILES = 0
# Perf lamp L3: the default compute config's dst_full_sync_en (8 fp32 DEST tiles a group, no pack/math overlap).
# Measured (Refinement 3, 8x512 / 4x512 / 2x512 us): True 15.6 / 9.7 / 7.9 vs False 13.4 / 8.4 / 7.0 at HiFi4, and
# 14.8 / 9.1 / 9.1 vs 11.9 / 7.8 / 6.5 at HiFi3 (the pack no longer overlaps the math) -> False.
DST_FULL_SYNC = False
# Bias seed path (perf lamp L4): True = row-broadcast datacopy through srcB (TF32: exact, the bias is bf16-valued),
# the srcB mode the multiply uses; False = unary_bcast<ROW>, which takes the fp32 bias CB through unpack-to-dest.
# Measured at HiFi3: True 11.85 / 7.85 / 6.41 vs False 11.74-11.87 / 7.74-7.76 / 6.46-7.01 us (neutral) -> False.
BIAS_SEED_SRCB = False
# Math fidelities that meet the precision contract (golden bars: PCC >= 0.999995, rel RMS <= 0.0019, one-hot exact).
# The multiply is rows (bf16, srcA) x weights (fp32 -> TF32, srcB). HiFi3 skips only the last fidelity phase, the
# rows' low mantissa bits x the weights' low bits (under 2^-10 of a product; zero for one-hot weights): measured PCC
# 0.9999986, rel RMS 0.00168, 98.4% within 1 ULP (HiFi4: same PCC / RMS, 98.8%). HiFi2 drops the weights' low bits
# against every row bit: rel RMS 0.0036, refused.
PRECISION_FIDELITIES = (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3)
# The default compute config's math fidelity (Refinement 3). The multiply is FPU-bound per fidelity phase (~35 cycles
# a tile): HiFi3 measured 8x512 13.4 -> 11.8 us, 4x512 8.4 -> 7.9 (2x512 within noise).
MATH_FIDELITY = ttnn.MathFidelity.HiFi3
# Reader start-up order: True = the index read alone, full barrier, then the weight / bias reads (no other DRAM
# request of a core queues ahead of the index reads on the index's banks); False = all issued at once (op_design).
# Measured (8x512 / 4x512 / 2x512 us): True 17.4 / 11.2 / 8.6 vs False 17.0 / 10.7 / 8.4 -> False.
INDEX_FIRST = False
# Bias copy of core i: BIAS_COPY_OFFSET + i % (BIAS_COPIES - BIAS_COPY_OFFSET). combine_index's pages 0 and 1 (choice
# k's ids) sit on DRAM banks 0 and 1 and every core's readers wait on them first; offset TOP_K keeps the bias reads
# off those banks. 0 = every copy (op_design). Measured: TOP_K 17.0 / 10.7 / 8.4 vs 0 16.9 / 10.2 / 8.0 -> 0.
BIAS_COPY_OFFSET = 0
# Weight tiles (tile rows) a reader issues ahead of its row reads; None = all at start-up (op_design). Spreading the
# weights' DRAM requests over the gather was measured neutral (1: 16.7 / 10.4 / 7.3 us, 2: 17.0 / 9.9 / 7.3, None:
# 16.6 / 9.9 / 7.3), so None.
WEIGHTS_LOOKAHEAD_ROWS = None
# Request VC of the readers' DRAM side reads (weights, bias); the row reads use the default read VC 1. 1 = same VC.
# Measured: 1 14.7 / 9.1 / 6.8 us, 0 15.4 / 9.3 / 6.9, 2 15.3 / 9.5 / 7.1, 3 15.3 / 9.4 / 7.0 -> 1.
SIDE_READ_VC = 1
# Weight tile read shape: True = faces 0-2 as one 3 KB DRAM request; False = faces 0 and 2 as two 1 KB requests.
# Measured: True 16.1-16.5 / 9.9-11.8 / 7.2-7.6 us vs False 14.7-14.9 / 9.2 / 6.7-6.9 (bytes, not requests, bind).
WEIGHTS_ONE_READ = False
# Row batches a reader issues before its weight / bias DRAM reads (the index is then waited alone). 0 = side reads at
# start-up (op_design). The first tile row's row reads then get ahead of the grid's DRAM burst, which otherwise
# delays every core's first tile row by 2-5 us (and compute, which runs the tile rows back to back after it).
# Measured (us): 0 15.2 / 8.7 / 7.0, 1 14.3 / 10.2 / 6.7, 2 14.7 / 9.8 / 6.7, 4 15.7 / 10.1 / 6.8, 8 17.0 / 9.7 / 6.9:
# within run-to-run noise (+-0.5 us) and not one-signed across shapes -> 0.
SIDE_READS_AFTER_BATCHES = 0
# group_shared_index_weights (Refinement 2): the G = 24 / W owners of a row group read identical index ids and weight
# faces. True = only the group's leader (leader_member) reads them from DRAM and unicasts them into its peers' CBs (same L1
# offsets) with one semaphore per choice; False = every owner reads its own (op_design Phase 0). No-op at G = 1.
SHARE_INDEX = True
SHARE_WEIGHTS = True
# Leader of choice k: member 0 for both (False), or member 0 for choice 0 and member G-1 for choice 1 (True: the two
# choices' DRAM reads and forwarding writes start from different cores).
STAGGER_LEADERS = False


# Per-core values (tr0, core_row_tiles, g, bias copy) packed into one common runtime arg a core, this many bits a field
# (Refinement 4: the kernels read their record from a table instead of per-core runtime args).
CORE_RECORD_FIELD_BITS = 8


def pack_core_record(*fields):
    record = 0
    for position, value in enumerate(fields):
        assert 0 <= value < 1 << CORE_RECORD_FIELD_BITS, f"core record field {value} overflows"
        record |= value << (position * CORE_RECORD_FIELD_BITS)
    return record


def leader_member(choice, hidden_split):
    return hidden_split - 1 if STAGGER_LEADERS and choice == 1 else 0


# Choice k's reader: (RISC, NoC). op_design: reader 0 on RISCV_0 / NOC_0, reader 1 on RISCV_1 / NOC_1.
READER_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
)

BF16_TILE_BYTES = TILE * TILE * 2
FP32_TILE_BYTES = TILE * TILE * 4
INDEX_TILE_ROW_BYTES = TILE * 4  # 32 uint32 ids of one tile row of one choice


def default_compute_kernel_config():
    """MATH_FIDELITY (HiFi3) + fp32 DEST accumulation: the precision contract (weights at TF32 in srcB, one rounding
    at the pack)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=MATH_FIDELITY,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        dst_full_sync_en=DST_FULL_SYNC,
    )


# ---------------------------------------------------------------------------
# Prepared bias
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedCombineBias:
    """(1, 1, BIAS_COPIES, 768) float32 ROW_MAJOR interleaved in DRAM: row p (page p, on DRAM bank p) is the bias."""

    rows: ttnn.Tensor


def prepare_combine_bias(bias, device):
    """Load time: widen the (1, 1, 1, 768) bf16 bias to fp32 (exact) and replicate it once per DRAM bank."""
    import torch

    host = ttnn.to_torch(bias).reshape(-1)[:HIDDEN].to(torch.float32)
    rows = host.reshape(1, 1, 1, HIDDEN).repeat(1, 1, BIAS_COPIES, 1).contiguous()
    tensor = ttnn.from_torch(
        rows, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return PreparedCombineBias(rows=tensor)


# ---------------------------------------------------------------------------
# Plan (the work split; computed once per shape and cached)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    num_tokens: int
    batch: int
    token_tiles: int  # T' / 32, flat (batch folded in)
    seq_tiles: int  # St = ceil(S / 32), per batch
    shard_row_tiles: int  # r
    shard_col_tiles: int  # W
    hidden_split: int  # G = 24 / W
    rows_depth_tile_rows: int
    cores: ttnn.CoreRangeSet
    core_list: tuple
    # per core i: (first flat tile row tr0, core_row_tiles, hidden slice g, bias copy)
    per_core: tuple


def _refuse(message):
    raise UnsupportedAxisValue(f"nomic_moe_combine: {message}")


def _shard_geometry(memory_config):
    """(r, W, cores CoreRangeSet, row-major core list) of an `owner_computes`-shaped ND-sharded memory config."""
    if memory_config is None or memory_config.buffer_type != ttnn.BufferType.L1:
        _refuse("the output must be ND-sharded in L1 (norm layout)")
    spec = getattr(memory_config, "nd_shard_spec", None)
    if spec is None:
        _refuse("the output memory config has no nd_shard_spec")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        _refuse("the output's shards must be in ROW_MAJOR orientation")
    if spec.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        _refuse("the output's shards must be distributed ROUND_ROBIN_1D")
    shape = list(spec.shard_shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[2] % TILE or shape[3] % TILE:
        _refuse(f"output shard shape {shape} is not (1, 1, 32r, 32W)")
    r, w = shape[2] // TILE, shape[3] // TILE
    if HIDDEN_TILES_TOTAL % w or w > MAX_SHARD_COL_TILES:
        _refuse(f"output shard width {w} tiles must divide {HIDDEN_TILES_TOTAL} and be <= {MAX_SHARD_COL_TILES}")
    max_rows = NOC_MAX_TRID - BATCH_DEPTH - SIDE_TRIDS
    if r > max_rows:
        _refuse(f"output shard height {r} tiles is over {max_rows} (one transaction id per tile row's weight tile)")
    cores = spec.grid
    core_list = tuple(ttnn.corerange_to_cores(cores, None, True))
    return r, w, cores, core_list


def _rows_depth(r):
    return r if ROWS_DEPTH_TILE_ROWS is None else max(1, min(r, ROWS_DEPTH_TILE_ROWS))


def cb_bytes(r, w, rows_depth=None, fused_tilize=None):
    """L1 the op's own CBs take per core, beyond the io tensors (l1_ledger.md § Total). Sums exactly the CBs the
    descriptor allocates (circular_buffer_specs is the one source of their sizes)."""
    from .nomic_moe_combine_program_descriptor import circular_buffer_specs

    depth = r if rows_depth is None else rows_depth
    fused = FUSED_TILIZE if fused_tilize is None else fused_tilize
    return sum(pages * page_size for _, pages, _, page_size in circular_buffer_specs(r, w, depth, fused))


_PLAN_CACHE = {}


def _plan(num_tokens, batch, memory_config) -> Plan:
    key = (num_tokens, batch, repr(memory_config))
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    seqlen = num_tokens // batch
    seq_tiles = math.ceil(seqlen / TILE)
    token_tiles = math.ceil(num_tokens / TILE)
    assert batch * seq_tiles == token_tiles, "B > 1 implies S % 32 == 0 (validated)"
    r, w, cores, core_list = _shard_geometry(memory_config)
    g_split = HIDDEN_TILES_TOTAL // w
    shard_blocks = math.ceil(seq_tiles / r)  # nSB: shard row groups per batch
    num_shards = batch * shard_blocks * g_split
    if num_shards != len(core_list):
        _refuse(f"{num_shards} output shards on {len(core_list)} cores: exactly one shard per core is required")
    # The kernels find their shard as core i = y * grid_x + x (core tables, Refinement 4): the shard order must be the
    # grid's first cores in row-major order (the norm layout's num_cores_to_corerangeset(..., row_wise=True)).
    if any((core.x, core.y) != (i % GRID[0], i // GRID[0]) for i, core in enumerate(core_list)):
        _refuse("the output's shard grid must be the device grid's first cores in row-major order")
    per_core = []
    for i in range(len(core_list)):
        g, rg = i % g_split, i // g_split
        b, sb = rg // shard_blocks, rg % shard_blocks
        tr0 = b * seq_tiles + sb * r
        rows = min(r, seq_tiles - sb * r)
        bias_copy = BIAS_COPY_OFFSET + i % (BIAS_COPIES - BIAS_COPY_OFFSET)
        per_core.append((tr0, rows, g, bias_copy))
    plan = Plan(
        num_tokens=num_tokens,
        batch=batch,
        token_tiles=token_tiles,
        seq_tiles=seq_tiles,
        shard_row_tiles=r,
        shard_col_tiles=w,
        hidden_split=g_split,
        rows_depth_tile_rows=_rows_depth(r),
        cores=cores,
        core_list=core_list,
        per_core=tuple(per_core),
    )
    _PLAN_CACHE[key] = plan
    return plan


def circular_buffer_bytes(num_tokens, batch, memory_config):
    """Per-core L1 of the op's CBs beyond the io tensors for this plan (l1_ledger.md § Total)."""
    r, w, _, _ = _shard_geometry(memory_config)
    return cb_bytes(r, w, _rows_depth(r))


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs[0] = output shape (B, 1, S, 768), inputs[1] = expert_rows shape (1, 1, R, 768))
# ---------------------------------------------------------------------------


def tag_window(inputs, axes):
    shape = inputs[0]
    tokens = shape[0] * shape[-2]
    return "aligned" if tokens % TILE == 0 else "ragged"


def tag_batch(inputs, axes):
    return "single" if inputs[0][0] == 1 else "multi"


INPUT_TAGGERS = {
    "window": tag_window,
    "batch": tag_batch,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
# rows_buffer "dram" (regime owner_computes_dram_rows) and output_layout l1/dram_interleaved (regime
# interleaved_output) are deferred (op_design.md § Regimes).

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "rows_buffer": ["l1"],
    "output_layout": ["norm_layout"],
    "window": ["aligned", "ragged"],
    "batch": ["single", "multi"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _output_layout(memory_config, shape, device):
    try:
        from ttnn.operations.nomic_residual_layer_norm import input_memory_config as norm_layout

        if memory_config == norm_layout(tuple(shape), device):
            return "norm_layout"
    except Exception:  # noqa: BLE001  (no norm plan for this shape: not the norm layout)
        pass
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if memory_config.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    return "other"


def _is_interleaved(tensor):
    return tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@dataclass
class _CallState:
    """Everything a call derives from its key (validate()'s derived values + the program), cached per key."""

    plan: Plan
    output_shape: ttnn.Shape
    padded_tokens: int
    compute: ttnn.ComputeConfigDescriptor
    descriptor: object = None  # ttnn.ProgramDescriptor, built on the key's first call (needs the tensors)


# Call-state cache (Refinement 4). Key: (T, B, R, rows dtype, rows buffer type, output memory config, compute config,
# device) -- everything the plan, the derived checks and the program descriptor depend on beyond buffer addresses.
# Knobs are read when an entry is built: after patching one, reset this cache (and _PLAN_CACHE), as the perf sweep does.
_CALL_CACHE = {}


def _compute_key(compute):
    if compute is None:
        return None
    return (
        compute.math_fidelity,
        compute.fp32_dest_acc_en,
        compute.math_approx_mode,
        compute.dst_full_sync_en,
        compute.bfp8_pack_precise,
        tuple(compute.unpack_to_dest_mode),
    )


def _derive(expert_rows, rows_shape, num_tokens, batch, config, compute_kernel_config, device):
    """The key's checks that do not depend on a call's tensors beyond the key (axes, domain, grid, compute config,
    plan, CB budget). Run once per key."""
    shape = (batch, 1, num_tokens // batch, HIDDEN)
    axes = {
        "dtype": expert_rows.dtype,
        "rows_buffer": "l1" if expert_rows.memory_config().buffer_type == ttnn.BufferType.L1 else "dram",
        "output_layout": _output_layout(config, shape, device),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape, rows_shape), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_moe_combine: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_moe_combine: unsupported combination (refinement candidate): {exc}")

    # Domain (prompt § Validation).
    if not (TOKENS_MIN <= num_tokens <= TOKENS_MAX):
        _refuse(f"num_tokens {num_tokens} outside [{TOKENS_MIN}, {TOKENS_MAX}]")
    if batch > 1 and (num_tokens // batch) % TILE:
        _refuse(f"batch {batch} > 1 needs S = {num_tokens // batch} to be a multiple of {TILE}")
    if len(rows_shape) != 4 or rows_shape[:2] != (1, 1) or rows_shape[3] != HIDDEN:
        _refuse(f"expert_rows shape {rows_shape} is not (1, 1, R, {HIDDEN})")
    if not (ROWS_MIN <= rows_shape[2] <= ROWS_MAX):
        _refuse(f"expert_rows R = {rows_shape[2]} outside [{ROWS_MIN}, {ROWS_MAX}]")

    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != GRID:
        _refuse(f"the device's compute grid {grid.x}x{grid.y} is not {GRID[0]}x{GRID[1]}")

    compute = compute_kernel_config or default_compute_kernel_config()
    if not getattr(compute, "fp32_dest_acc_en", False) or compute.math_fidelity not in PRECISION_FIDELITIES:
        _refuse(
            f"compute_kernel_config must have fp32_dest_acc_en=True and math_fidelity in {PRECISION_FIDELITIES} "
            "(precision contract)"
        )

    # owner_computes predicate: geometry, one shard per core, CB budget.
    plan = _plan(num_tokens, batch, config)
    used = cb_bytes(plan.shard_row_tiles, plan.shard_col_tiles, plan.rows_depth_tile_rows)
    if used > CB_L1_BUDGET:
        _refuse(f"circular buffers need {used} B a core, over the {CB_L1_BUDGET} B budget")
    return _CallState(
        plan=plan,
        output_shape=ttnn.Shape(list(shape)),
        padded_tokens=math.ceil(num_tokens / TILE) * TILE,
        compute=compute,
    )


def _check_tensors(expert_rows, combine_index, combine_weights, bias, padded_tokens):
    """The checks on the call's tensors themselves (layouts, dtypes, shapes, buffer types). Run on every call."""
    if expert_rows.layout != ttnn.ROW_MAJOR_LAYOUT or not _is_interleaved(expert_rows):
        _refuse("expert_rows must be ROW_MAJOR interleaved")

    if (
        tuple(combine_index.shape) != (TOP_K, padded_tokens)
        or combine_index.dtype != ttnn.uint32
        or combine_index.layout != ttnn.ROW_MAJOR_LAYOUT
        or not _is_interleaved(combine_index)
        or combine_index.memory_config().buffer_type != ttnn.BufferType.DRAM
    ):
        _refuse(f"combine_index must be ({TOP_K}, {padded_tokens}) uint32 ROW_MAJOR interleaved in DRAM")
    if (
        tuple(combine_weights.shape) != (1, TOP_K, padded_tokens, 1)
        or combine_weights.dtype != ttnn.float32
        or combine_weights.layout != ttnn.TILE_LAYOUT
        or not _is_interleaved(combine_weights)
        or combine_weights.memory_config().buffer_type != ttnn.BufferType.DRAM
    ):
        _refuse(f"combine_weights must be (1, {TOP_K}, {padded_tokens}, 1) float32 TILE interleaved in DRAM")

    if not isinstance(bias, PreparedCombineBias):
        _refuse("bias must be what prepare_combine_bias returned")
    rows = bias.rows
    if (
        tuple(rows.shape) != (1, 1, BIAS_COPIES, HIDDEN)
        or rows.dtype != ttnn.float32
        or rows.layout != ttnn.ROW_MAJOR_LAYOUT
        or not _is_interleaved(rows)
        or rows.memory_config().buffer_type != ttnn.BufferType.DRAM
    ):
        _refuse("bias must be what prepare_combine_bias returned (fp32 rows in DRAM)")


def _validate(
    expert_rows,
    combine_index,
    combine_weights,
    bias,
    *,
    num_tokens,
    batch=1,
    memory_config=None,
    compute_kernel_config=None,
):
    """validate() returning the key's cached _CallState. The derived checks run once per key; the tensor checks run on
    every call."""
    config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    if not isinstance(num_tokens, int) or not isinstance(batch, int) or batch < 1 or num_tokens < 1:
        _refuse(f"num_tokens={num_tokens!r} and batch={batch!r} must be positive integers")
    if num_tokens % batch:
        _refuse(f"num_tokens {num_tokens} is not a multiple of batch {batch}")
    rows_shape = tuple(expert_rows.shape)
    device = expert_rows.device()
    key = (
        num_tokens,
        batch,
        rows_shape,
        expert_rows.dtype,
        expert_rows.memory_config().buffer_type,
        repr(config),
        _compute_key(compute_kernel_config),
        device.id(),
    )
    state = _CALL_CACHE.get(key)
    if state is None:
        state = _derive(expert_rows, rows_shape, num_tokens, batch, config, compute_kernel_config, device)
        _check_tensors(expert_rows, combine_index, combine_weights, bias, state.padded_tokens)
        _CALL_CACHE[key] = state
    else:
        _check_tensors(expert_rows, combine_index, combine_weights, bias, state.padded_tokens)
    return state


def validate(
    expert_rows,
    combine_index,
    combine_weights,
    bias,
    *,
    num_tokens,
    batch=1,
    memory_config=None,
    compute_kernel_config=None,
):
    return _validate(
        expert_rows,
        combine_index,
        combine_weights,
        bias,
        num_tokens=num_tokens,
        batch=batch,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    ).plan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def nomic_moe_combine(
    expert_rows,
    combine_index,
    combine_weights,
    bias,
    *,
    num_tokens,
    batch=1,
    memory_config=None,
    compute_kernel_config=None,
):
    state = _validate(
        expert_rows,
        combine_index,
        combine_weights,
        bias,
        num_tokens=num_tokens,
        batch=batch,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    from .nomic_moe_combine_program_descriptor import bind_call, create_program_descriptor

    device = expert_rows.device()
    output = ttnn.allocate_tensor_on_device(
        state.output_shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG,
    )
    if state.descriptor is None:
        state.descriptor = create_program_descriptor(
            state.plan, expert_rows, combine_index, combine_weights, bias.rows, output, state.compute
        )
    # Every call rebinds the addresses (a cached descriptor never carries a previous call's).
    bind_call(state.descriptor, state.plan, expert_rows, combine_index, combine_weights, bias.rows, output)
    return ttnn.generic_op([expert_rows, combine_index, combine_weights, bias.rows, output], state.descriptor)
