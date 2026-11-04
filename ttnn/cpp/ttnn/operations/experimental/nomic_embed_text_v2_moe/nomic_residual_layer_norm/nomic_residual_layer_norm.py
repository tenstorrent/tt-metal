# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_residual_layer_norm: LayerNorm(x + residual) * gamma + beta over the hidden axis (768).

One ttnn.generic_op program (see op_design.md, regime `split3_exchange`), on either of two placements:
  * "op_layout" (the op's own input_memory_config): x / residual / output are ND-sharded in L1 over groups of
    G consecutive row-major cores (G = 3 or 2, picked per shape by measurement: _choose_hidden_split): a group owns
    `rows_per_shard` tile rows of one batch, and core c of the group owns hidden slice c (768 / G channels).
  * "model_layout" (the model's split layout, Refinement 1): BLOCK_SHARDED L1, COL_MAJOR, MODEL_HIDDEN_SPLIT core
    rows x C core columns, shard (32 * bh, 768 / MODEL_HIDDEN_SPLIT). A group is one grid column (bh tile rows of
    M), member g its grid row (hidden slice g). Same algorithm, plan read off the layout.
  * every core sums its slice of each row (S, Q in fp32), packs the per-row statistics into one tile,
    all-gathers the packed (S, Q) inside its group (one rendezvous per row block), combines them in a fixed order
    and finalizes mean / rstd itself, then normalizes its own slice into the output shard. The output is a new
    tensor in the input's layout, or x itself (output_tensor=x, in place).

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; INVALID lives in
eval/golden_tests/nomic_residual_layer_norm/feature_spec.py.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, SupportRefusal, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every dependent quantity derives from these)
# ---------------------------------------------------------------------------

HIDDEN = 768
TILE = 32
HIDDEN_TILES_TOTAL = HIDDEN // TILE  # 24
M_MIN, M_MAX = 32, 128
# op_layout: cores a row is cut across (block_extent[hidden] = HIDDEN_TILES_TOTAL / G). Perf lamp L1. The plan picks
# G per shape from these candidates (_choose_hidden_split; a 1-tuple forces one, the sweep driver does that).
HIDDEN_SPLIT_CANDIDATES = (3, 2)
# Measured device-kernel us of an op_layout plan by (G, rows_per_shard), residual, HiFi4 (Refinement 2 sweep,
# test_nomic_residual_layer_norm_split_sweep.py; Blackhole p150, 1350 MHz; median over the golden M classes that land
# on the pair). The plan takes the candidate with the lowest entry; a pair not in the table ranks last, ties go to
# candidate order. Why G = 2 wins where it does: a 2-member exchange and 2-contributor combine are cheaper fixed
# costs, which dominate 1-row shards and beat G = 3's 3-row plans; G = 3 wins where it has the same row count with
# fewer tiles a row (2 rows) or keeps one block where G = 2 would need three (M > 110: 4 rows vs 3 rows at
# 12 tiles, which only fits L1 as 1-row blocks).
#   G = 2 (12 tiles a row):  1 row 7.2-9.5 (M 32-48) | 2 rows 11.6-11.8 (M 64-96) | 3 rows 18.4 (M 128)
#   G = 3 ( 8 tiles a row):  1 row 7.9-9.2 (M 32-36) | 2 rows 10.1-10.3 (M 39-64) | 3 rows 14.7-14.8 (M 80-96) |
#                            4 rows 18.2 (M 128; 16.9 as 2 pipelined 2-row blocks, PIPELINE_BLOCK_ROWS)
SPLIT_COST_US = {
    (2, 1): 7.6,
    (2, 2): 11.7,
    (2, 3): 18.4,
    (3, 1): 8.2,
    (3, 2): 10.2,
    (3, 3): 14.8,
    (3, 4): 16.9,  # Refinement 3: 2 pipelined row blocks (was 18.2 as one block)
}
# op_layout: largest per-core tile-row count the L1 budget admits (l1_ledger: 240 KB at 4 rows).
MAX_ROWS_PER_SHARD = 4
# The model's split layout (models/experimental/nomic_embed_text_v2_moe/tt/common.py, sharded_norm_split /
# sharded_norm_configs), replicated here so the op does not import models/: MODEL_HIDDEN_SPLIT core rows carry the
# hidden axis (SHARDED_NORM_COLUMNS), M spreads over the grid's columns at bh = ceil(M / grid.x) tile rows a column.
MODEL_HIDDEN_SPLIT = 8
# Largest row block (block_extent[rows]) a shard is processed in. Perf lamp L2. The plan takes the largest divisor of
# rows_per_shard that is <= MAX_BLOCK_ROWS and whose CBs fit CB_L1_BUDGET, so every shape stays plannable when the
# knob is turned. Blocks are uniform: the ring CBs (cb_y, cb_row_partials, ...) are re-walked by several passes a
# block with linear tile indices, which is only wrap-safe when every block fills them exactly. Default: no cap
# beyond L1, so one block = the whole shard wherever it fits (every op_layout plan), else the coarsest that fits
# (model_layout: 4 rows at G = 8, 3 hidden tiles).
MAX_BLOCK_ROWS = M_MAX
# Row-block software pipelining (perf lamp L2, Refinement 3): a plan that the rule above leaves as ONE block is split
# into PIPELINE_BLOCK_ROWS-row blocks when that divides the shard into at least two, so the lookahead schedule
# (descriptor LOOKAHEAD) overlaps block b's exchange with block b+1's phase A and block b+1's with block b's normalize.
# Measured (device us, residual): op 8x512 / 7x512 (4-row shards) 18.1 -> 16.9; model 4x288 (bh 4) 15.2 -> 14.8.
# Smaller blocks lose wherever they appear: 1-row blocks (op 8x512 20.2, op 6x512 G = 3 16.5 vs 14.8, model 2x512 17.2
# vs 13.3) and plans L1 already splits (model 8x512, 3 blocks of 4 -> 6 of 2: 38.9 vs 34.3), so the split only applies
# to single-block plans. None turns it off.
PIPELINE_BLOCK_ROWS = 2
# Per-core L1 the op's own CBs may take beyond the io shards (l1_ledger.md § Total).
CB_L1_BUDGET = 256 * 1024
# Rows one packed statistics ("pair") tile holds: row j's S in column slot j, its Q in slot 2 + j of the four
# column slots {0, 1, 16, 17} (mechanism cap). S and Q of a row in one tile let the finalize extract every row's
# mean / rstd in one DEST window.
STAT_SLOTS = 2

PLACEMENTS = ("op_layout", "model_layout")


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


@dataclass(frozen=True)
class Plan:
    placement: str
    batch: int
    seq_tiles: int  # St = ceil(S / 32), per batch
    rows_per_shard: int
    block_rows: int
    num_row_blocks: int
    stat_slots: int
    stat_tiles: int
    hidden_split: int
    hidden_tiles: int
    num_row_groups: int
    num_cores: int
    cores: ttnn.CoreRangeSet
    # Ordered logical cores: group g is core_list[g*G : (g+1)*G], member c of a group owns hidden slice c.
    # op_layout: row-major (core i owns ND shard i). model_layout: column-major (a group is one grid column).
    core_list: tuple
    column_major: bool


def _check_shape(shape):
    if len(shape) != 4 or shape[1] != 1 or shape[-1] != HIDDEN:
        raise UnsupportedAxisValue(f"nomic_residual_layer_norm: shape {shape} is not (B, 1, S, {HIDDEN})")
    batch, seq_tiles = shape[0], math.ceil(shape[2] / TILE)
    m = batch * seq_tiles
    if not (M_MIN <= m <= M_MAX):
        raise UnsupportedAxisValue(f"nomic_residual_layer_norm: M = {m} tile rows outside [{M_MIN}, {M_MAX}]")
    return batch, seq_tiles, m


def _model_split(m, grid):
    """(core columns C, tile rows a column bh) of the model's split layout, or None where the model has none."""
    if m > M_MAX or MODEL_HIDDEN_SPLIT > grid.y or HIDDEN_TILES_TOTAL % MODEL_HIDDEN_SPLIT:
        return None
    block_h = math.ceil(m / grid.x)
    return math.ceil(m / block_h), block_h


def _block_rows(rows_per_shard, hidden_tiles, hidden_split):
    """Largest divisor of rows_per_shard <= MAX_BLOCK_ROWS whose CBs fit CB_L1_BUDGET (uniform blocks); a single-block
    result is split into PIPELINE_BLOCK_ROWS-row blocks where that gives at least two (see the knob)."""
    from .nomic_residual_layer_norm_program_descriptor import cb_bytes

    def fits(d):
        blocks = rows_per_shard // d
        return cb_bytes(d, hidden_tiles, math.ceil(d / STAT_SLOTS), hidden_split, blocks) <= CB_L1_BUDGET

    for d in range(min(MAX_BLOCK_ROWS, rows_per_shard), 0, -1):
        if rows_per_shard % d == 0 and fits(d):
            p = PIPELINE_BLOCK_ROWS
            if (
                d == rows_per_shard
                and p
                and p <= MAX_BLOCK_ROWS
                and rows_per_shard % p == 0
                and rows_per_shard // p >= 2
            ):
                if fits(p):
                    return p
            return d
    raise UnsupportedAxisValue("nomic_residual_layer_norm: no row block fits the CB L1 budget")


def _op_layout_rows(batch, seq_tiles, grid, hidden_split):
    """Fewest tile rows a core (<= MAX_ROWS_PER_SHARD) whose row groups of hidden_split cores fit the grid."""
    max_groups = (grid.x * grid.y) // hidden_split
    for r in range(1, MAX_ROWS_PER_SHARD + 1):
        if batch * math.ceil(seq_tiles / r) <= max_groups:
            return r
    return None


def _choose_hidden_split(batch, seq_tiles, grid):
    """(G, rows_per_shard) of the op layout: the plannable candidate with the lowest measured cost (SPLIT_COST_US).

    rows_per_shard is None when no candidate plans."""
    best = None
    for order, g in enumerate(HIDDEN_SPLIT_CANDIDATES):
        if HIDDEN_TILES_TOTAL % g or (HIDDEN_TILES_TOTAL // g) % 2:
            continue  # the hidden split must divide 24 tiles into an even width (pairwise-halves row sum)
        rows = _op_layout_rows(batch, seq_tiles, grid, g)
        if rows is None:
            continue
        key = (SPLIT_COST_US.get((g, rows), math.inf), order)
        if best is None or key < best[0]:
            best = (key, g, rows)
    if best is None:
        return HIDDEN_SPLIT_CANDIDATES[0], None
    return best[1], best[2]


def _plan(shape, device, placement="op_layout") -> Plan:
    """The single source of the work split. Raises SupportRefusal when no plan exists for the placement."""
    shape = tuple(shape)
    batch, seq_tiles, m = _check_shape(shape)
    grid = device.compute_with_storage_grid_size()
    if placement == "model_layout":
        split = _model_split(m, grid)
        if split is None:
            raise UnsupportedAxisValue(f"nomic_residual_layer_norm: no model split layout for shape {shape}")
        columns, rows_per_shard = split
        hidden_split = MODEL_HIDDEN_SPLIT
        num_row_groups = columns
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, hidden_split - 1))})
        core_list = tuple(ttnn.CoreCoord(x, y) for x in range(columns) for y in range(hidden_split))
        column_major = True
    else:
        hidden_split, rows_per_shard = _choose_hidden_split(batch, seq_tiles, grid)
        if rows_per_shard is None:
            raise UnsupportedAxisValue(
                f"nomic_residual_layer_norm: no plan with <= {MAX_ROWS_PER_SHARD} rows a core for shape {shape}"
            )
        num_row_groups = batch * math.ceil(seq_tiles / rows_per_shard)
        cores = ttnn.num_cores_to_corerangeset(hidden_split * num_row_groups, grid, True)
        core_list = tuple(ttnn.corerange_to_cores(cores, None, True))
        column_major = False
    assert HIDDEN_TILES_TOTAL % hidden_split == 0, "the hidden split must divide the hidden tile count"
    hidden_tiles = HIDDEN_TILES_TOTAL // hidden_split
    block_rows = _block_rows(rows_per_shard, hidden_tiles, hidden_split)
    num_cores = hidden_split * num_row_groups
    assert len(core_list) == num_cores
    return Plan(
        placement=placement,
        batch=batch,
        seq_tiles=seq_tiles,
        rows_per_shard=rows_per_shard,
        block_rows=block_rows,
        num_row_blocks=rows_per_shard // block_rows,
        stat_slots=STAT_SLOTS,
        stat_tiles=math.ceil(block_rows / STAT_SLOTS),
        hidden_split=hidden_split,
        hidden_tiles=hidden_tiles,
        num_row_groups=num_row_groups,
        num_cores=num_cores,
        cores=cores,
        core_list=core_list,
        column_major=column_major,
    )


def input_memory_config(input_shape, device):
    """ND-sharded L1: shard [1, 1, rows_per_shard*32, hidden_tiles*32] on the plan's cores, row-major, round-robin."""
    plan = _plan(input_shape, device)
    spec = ttnn.NdShardSpec(
        ttnn.Shape([1, 1, plan.rows_per_shard * TILE, plan.hidden_tiles * TILE]),
        plan.cores,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.L1, spec)


def model_memory_config(input_shape, device):
    """The model's split layout ("model_layout"): BLOCK_SHARDED L1, COL_MAJOR, 8 core rows x C core columns, shard
    (32 * bh, 96). Equal to sharded_norm_configs(*sharded_norm_split(M, 24, grid), 24)[0] of the model."""
    _, _, m = _check_shape(tuple(input_shape))
    split = _model_split(m, device.compute_with_storage_grid_size())
    if split is None:
        raise UnsupportedAxisValue(f"nomic_residual_layer_norm: no model split layout for shape {tuple(input_shape)}")
    columns, block_h = split
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, MODEL_HIDDEN_SPLIT - 1))}
    )
    shard = ttnn.ShardSpec(
        grid, (block_h * TILE, (HIDDEN_TILES_TOTAL // MODEL_HIDDEN_SPLIT) * TILE), ttnn.ShardOrientation.COL_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def output_memory_config(input_shape, device, input_memory_config):
    """The output layout equals the input layout (either placement)."""
    return input_memory_config


def circular_buffer_bytes(input_shape, residual=None, device=None, placement="op_layout"):
    """Closed form of l1_ledger.md (§ Total) for the plan of input_shape on `placement` (bytes per core beyond the
    io tensors)."""
    if device is None:
        # The plan depends on the device only through its grid; default to the 11 x 10 Blackhole grid.
        class _Device:
            @staticmethod
            def compute_with_storage_grid_size():
                return ttnn.CoreCoord(11, 10)

        device = _Device()
    plan = _plan(input_shape, device, placement)
    from .nomic_residual_layer_norm_program_descriptor import cb_bytes_per_core

    return cb_bytes_per_core(plan)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_seq_alignment(inputs, axes):
    shape = inputs[0]
    return "tile_aligned" if shape[-2] % TILE == 0 else "seq_padded"


INPUT_TAGGERS = {
    "seq_alignment": tag_seq_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "placement": ["op_layout", "model_layout"],
    "residual_mode": ["residual", "no_residual"],
    "output_mode": ["new", "in_place"],
    "seq_alignment": ["tile_aligned", "seq_padded"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _placement(input_tensor):
    shape, device, config = tuple(input_tensor.shape), input_tensor.device(), input_tensor.memory_config()
    for placement, layout_of in (("op_layout", input_memory_config), ("model_layout", model_memory_config)):
        try:
            if config == layout_of(shape, device):
                return placement
        except SupportRefusal:
            pass
    return "other"


def validate(input_tensor, gamma, beta, *, residual=None, output_tensor=None, compute_kernel_config=None):
    config = compute_kernel_config or default_compute_kernel_config()
    in_place = output_tensor is not None and output_tensor.buffer_address() == input_tensor.buffer_address()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(getattr(config, "fp32_dest_acc_en", False)),
        "layout": input_tensor.layout,
        "placement": _placement(input_tensor),
        "residual_mode": "residual" if residual is not None else "no_residual",
        "output_mode": "in_place" if in_place else ("new" if output_tensor is None else "provided"),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_residual_layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_residual_layer_norm: unsupported combination (refinement candidate): {exc}")

    # Shape gate: M in range and a plan exists for the placement (raises SupportRefusal otherwise).
    _plan(tuple(input_tensor.shape), input_tensor.device(), axes["placement"])

    # Operand contract. The prompt's Validation section requires a SupportRefusal (not a ValueError) for each of
    # these, so a caller can fall back to its old path on any refused call.
    if residual is not None:
        if tuple(residual.shape) != tuple(input_tensor.shape) or residual.dtype != input_tensor.dtype:
            raise UnsupportedAxisValue("nomic_residual_layer_norm: residual must match x in shape and dtype")
        if residual.memory_config() != input_tensor.memory_config() or residual.layout != input_tensor.layout:
            raise UnsupportedAxisValue("nomic_residual_layer_norm: residual must share x's layout and memory config")
    for name, tensor, dtype in (("gamma", gamma, ttnn.float32), ("beta", beta, ttnn.bfloat16)):
        if tuple(tensor.shape)[-1] != HIDDEN or tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype != dtype:
            raise UnsupportedAxisValue(f"nomic_residual_layer_norm: {name} must be (1, 1, 1, {HIDDEN}) {dtype} TILE")
        if tensor.memory_config().is_sharded():
            raise UnsupportedAxisValue(f"nomic_residual_layer_norm: {name} must be interleaved")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def nomic_residual_layer_norm(
    input_tensor,
    gamma,
    beta,
    *,
    residual=None,
    epsilon=1e-5,
    output_tensor=None,
    compute_kernel_config=None,
):
    validate(
        input_tensor,
        gamma,
        beta,
        residual=residual,
        output_tensor=output_tensor,
        compute_kernel_config=compute_kernel_config,
    )
    from .nomic_residual_layer_norm_program_descriptor import create_program_descriptor

    device = input_tensor.device()
    plan = _plan(tuple(input_tensor.shape), device, _placement(input_tensor))
    config = compute_kernel_config or default_compute_kernel_config()

    if output_tensor is not None:
        # In place (validate() admits output_tensor only when it is x): cb_output is backed on x's own shard. Each
        # block's x tiles are consumed (residual_add_block, or normalize stage 1 without a residual) before stage 4
        # packs the block's first output tile over them.
        output = input_tensor
    else:
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            output_memory_config(tuple(input_tensor.shape), device, input_tensor.memory_config()),
        )
    descriptor = create_program_descriptor(
        plan, input_tensor, residual, gamma, beta, output, _fp32_bits(epsilon), config
    )
    io = [input_tensor] + ([residual] if residual is not None else []) + [gamma, beta, output]
    return ttnn.generic_op(io, descriptor)
