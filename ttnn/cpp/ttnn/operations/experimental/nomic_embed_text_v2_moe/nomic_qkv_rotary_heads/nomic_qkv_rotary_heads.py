# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_qkv_rotary_heads: the front of every nomic-embed-text-v2-moe attention layer in one generic_op dispatch.

qkv = x @ W^T + b (fp32), split into q, k, v (12 heads of 64 each), q *= 1/8 (folded into the weights), NeoX
rotate-half rotary (base 10000) on q and k, written as three interleaved (B, 12, S, 64) bfloat16 TILE tensors.
Padding rows (s >= S) of q, k and v are exactly zero (selected, not multiplied).

Regime `mcast2d_fused_rotary` (op_design.md): M (tile rows) over grid columns, N lanes of LANE_N_TILES over grid
rows. x is gathered from the norm's ND shards by each column's top core and multicast down the column; W (with the
bias as K tile 24) is read once per row by the row's right core, multicast west and stays resident after M block 0.
K runs in K_BLOCK_TILES blocks with packer L1 accumulation; the last K block reloads the fp32 partials, adds the
bias tile, and (q/k rows) rotates each head in fp32 DEST before the single bf16 pack. The writer scatters every
output tile straight into its q / k / v page.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; INVALID lives in
eval/golden_tests/nomic_qkv_rotary_heads/feature_spec.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Problem constants and blocking knobs (single source of truth; every dependent quantity derives from these)
# ---------------------------------------------------------------------------

TILE = 32
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
ROTARY_BASE = 10000.0
Q_SCALE = HEAD_DIM**-0.5  # 1/8, exact in bf16

HIDDEN_TILES = HIDDEN // TILE  # 24 K tiles of x
K_TILES = HIDDEN_TILES + 1  # + the bias K tile (ones column of x x bias row of W)
BIAS_K_TILE = HIDDEN_TILES  # K index of the bias tile
HEAD_TILES = HEAD_DIM // TILE  # 2 tiles a head (the rotary pair)
TENSOR_N_TILES = HEADS * HEAD_TILES  # 24 N tiles of each of q, k, v
NUM_TENSORS = 3
N_TILES = NUM_TENSORS * TENSOR_N_TILES  # 72
NUM_ROTARY_TENSORS = 2  # q and k

# N block = one grid row's lane (knob). The rotary couples a head's two tiles, so lanes are whole heads.
LANE_N_TILES = 8
NUM_LANES = N_TILES // LANE_N_TILES  # 9 grid rows
# K block (knob). It must divide K_TILES so the bias tile lands at a fixed K position (mechanism cap).
K_BLOCK_TILES = 5
NUM_K_BLOCKS = K_TILES // K_BLOCK_TILES
# M block cap (L1 budget; l1_ledger.md): block_m_tiles = ceil(core_m_tiles / ceil(core_m_tiles / cap)).
BLOCK_M_TILES_MAX = 3
# Buffer-depth knobs (op_design.md § Buffer-depth knobs).
X_BLOCK_DEPTH = 2  # cb_x_block slots of one (M block x K block)
ROTARY_TABLE_DEPTH = 1  # cb_rotary_table M blocks
# The in0 K block of each M block after whose multicast a q/k reader requests the block's C / S tiles. They are only
# needed by the last K block, and cb_rotary_table is single-slot, so the request waits for the previous M block's last
# K block to finish. Issued right after K block 0 (0), that wait plus the table read sit in front of K block 1's
# gather + multicast on every reader of the column; after K block n-2 the previous block is long done (Refinement 4:
# 8x512 83.07 -> 82.41 us, 2x512 27.37 -> 26.84).
TABLE_LOAD_K_BLOCK = max(0, NUM_K_BLOCKS - 2)
TABLE_TILES_PER_ROW = 2  # one C and one S fp32 tile per position tile row (RotaryTables page order)
OUT_DEPTH_BLOCKS = 1  # cb_heads_out M blocks
# Output drain step (tiles, whole heads, divides LANE_N_TILES): the writer waits, scatters and pops this many
# cb_heads_out pages at a time, and the q/k finish call pushes per head when it is below LANE_N_TILES. LANE_N_TILES =
# one row at a time (the previous behaviour). One head lets a row's first heads leave while its last heads compute, so
# only the last head's pages trail the compute (perf lamp L7, the last-block tail).
OUT_DRAIN_TILES = HEAD_TILES
# in1 stream of M block 0 (Refinement 3). K rows per in1 multicast chunk (divides K_BLOCK_TILES). A sender retires
# its DRAM reads chunk by chunk (one transaction id each) and multicasts each chunk as soon as it lands; the K block's
# last chunk carries the ready signal. K_BLOCK_TILES = one multicast per K block (read, then multicast). Measured at
# 1 K row: block 0 reaches compute ~0.5 us earlier and compute ends ~0.6-1.1 us earlier at 2x512 / 4x512 / 8x512, but
# the output-write tail (v rows' scatter) still sets the op's end, so it is parked at the per-block default until
# that tail shortens.
W_MCAST_CHUNK_K_ROWS = K_BLOCK_TILES
# Read-ahead: chunks of DRAM reads a sender keeps in flight (1 .. NUM_K_BLOCKS * K_BLOCK_TILES / W_MCAST_CHUNK_K_ROWS,
# below the 15 transaction ids). The weight stream is DRAM-bandwidth-bound grid-wide, so more in flight delays the
# first K block that compute is waiting on (measured: 2 K blocks in flight cost ~3 us at 2x512).
WEIGHT_READ_AHEAD_CHUNKS = K_BLOCK_TILES // W_MCAST_CHUNK_K_ROWS
# Which core of a row reads the lane's weights from DRAM and multicasts them (perf lamp L5; catalog noc_placement,
# mcast_topology). The senders of the 9 rows read DRAM at the same time, so their placement is a NoC-placement knob:
#   W_SENDER_RIGHT     the row's right core (outside the receivers): the 9 readers form a column.
#   W_SENDER_DIAGONAL  row y's core in column y mod cols (inside the row's receiver group): a diagonal.
#   W_SENDER_ROTATING  K block b's sender is column (b + W_SENDER_STAGGER * y) mod cols: each row's reads spread over
#                      NUM_K_BLOCKS cores.
W_SENDER_RIGHT, W_SENDER_DIAGONAL, W_SENDER_ROTATING = 0, 1, 2
# Measured (3-run medians, 2x512 / 4x512 / 8x512): diagonal no faster than right (+0.3-1.3 us); rotating flat at a
# read-ahead of one block and up to +7 us deeper (the stream is DRAM-bound grid-wide: concurrent blocks delay block 0).
W_SENDER = W_SENDER_RIGHT
W_SENDER_STAGGER = 1  # rotating only: per-row offset of the rotation
# Half-sync fp32 DEST holds 4 tiles: a q/k subblock is one head (1 x HEAD_TILES) and C / S take the other two.
DEST_TILES = 4
QK_SUBBLOCK_W = HEAD_TILES
# The widest h = 1 subblock that fits DEST and divides the lane: v cores' subblock in every K block, and the q/k
# subblock of the K blocks before the last one (no rotary there, so C / S need no DEST slots; perf lamp L1: fewer
# operand reloads per tile product). The q/k last K block stays QK_SUBBLOCK_W (one head). Overriding the q/k
# accumulation width to QK_SUBBLOCK_W (QKV_QK_ACCUM_SUBBLOCK_W) restores the single-call K loop.
WIDE_SUBBLOCK_W = max(w for w in range(1, DEST_TILES + 1) if LANE_N_TILES % w == 0)
# Where the q/k C / S tiles enter DEST: 1 = in matmul_block's post-reload slot, on the fp32 partials reload's datacopy
# config (no SrcA reconfig / copy_init per head); 0 = after the head's FMAs, with their own reconfig + copy_init.
CS_ON_RELOAD = 1
# Where the q/k rotary runs: 1 = packer thread (overlaps the next head's FMAs), 0 = math thread (lamp L2 fallback).
ROTARY_ON_PACK = 1
# matmul_block output / partials layout: 1 = TileRowMajor (per tile-row spills, reloads and pushes; measured faster),
# 0 = SubblockMajor. Both give the writer the same tile-row-major cb_heads_out because out_subblock_h = 1.
OUT_LAYOUT_ROW_MAJOR = 1
# Per-core L1 the op's circular buffers may take (the task's budget at M = 128).
CB_L1_BUDGET = 640 * 1024
# The device grid this op is planned for (Blackhole p300c compute grid).
GRID_X, GRID_Y = 11, 10
M_MIN, M_MAX = 32, 128

BF16_TILE_BYTES = TILE * TILE * 2
FP32_TILE_BYTES = TILE * TILE * 4
# Weight DRAM page = WEIGHT_PAGE_TILES consecutive N tiles of one K row (catalog read_coalescing: one page per
# (K row, lane) is one 16 KB read instead of four 4 KB head pages). HEAD_TILES is the old head-page layout.
WEIGHT_PAGE_TILES = LANE_N_TILES
WEIGHT_PAGES_PER_K_ROW = N_TILES // WEIGHT_PAGE_TILES
WEIGHT_PAGE_ELEMS = WEIGHT_PAGE_TILES * TILE * TILE
WEIGHT_LAYOUT_TAG = f"n_pages_{WEIGHT_PAGE_TILES}_v1"

assert K_TILES % K_BLOCK_TILES == 0, "K_BLOCK_TILES must divide K_TILES (the bias tile sits at a fixed K index)"
assert LANE_N_TILES % HEAD_TILES == 0, "a lane must hold whole heads (rotary pairs)"
assert N_TILES % LANE_N_TILES == 0
assert TENSOR_N_TILES % LANE_N_TILES == 0, "a lane must not straddle q / k / v"
assert 1 <= WEIGHT_READ_AHEAD_CHUNKS < 15, "one NoC transaction id per in-flight chunk"
assert K_BLOCK_TILES % W_MCAST_CHUNK_K_ROWS == 0, "an in1 multicast chunk is whole K rows of one K block"
assert LANE_N_TILES % WEIGHT_PAGE_TILES == 0, "a lane must be whole weight pages"
assert LANE_N_TILES % OUT_DRAIN_TILES == 0 and OUT_DRAIN_TILES % HEAD_TILES == 0, "a drain step is whole heads of a row"
assert 0 <= TABLE_LOAD_K_BLOCK < NUM_K_BLOCKS, "the C / S request rides one of the M block's in0 K blocks"
assert QK_SUBBLOCK_W + 2 <= DEST_TILES, "the q/k subblock must leave two DEST slots for C and S"


# ---------------------------------------------------------------------------
# Device-resident inputs this op defines (host preparation, outside the dispatch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QkvWeights:
    """Wt = W^T with q's columns (and b_q) scaled by 1/8, plus the bias as K row 768 (rows 769..799 zero).

    ROW_MAJOR bf16 (1, 1, K_TILES * WEIGHT_PAGES_PER_K_ROW, WEIGHT_PAGE_ELEMS), DRAM interleaved. Page
    k * WEIGHT_PAGES_PER_K_ROW + p holds tiles (k, p * WEIGHT_PAGE_TILES) .. (k, (p + 1) * WEIGHT_PAGE_TILES - 1), in
    N order, each in standard 32x32 face order."""

    tensor: ttnn.Tensor
    layout_tag: str = WEIGHT_LAYOUT_TAG


@dataclass(frozen=True)
class RotaryTables:
    """fp32 TILE (1, 1, 32 * ceil(seqlen / 32), 64), L1 interleaved. Page 2 * s_t = cos tile, 2 * s_t + 1 = sin tile of
    position tile s_t; column i of either is frequency i. Rows s >= seqlen are 0."""

    tensor: ttnn.Tensor
    seqlen: int
    layout_tag: str = "cos_sin_tiles_v1"


def _to_face_order(tiles):
    """(..., 32, 32) -> (..., 1024) in the hardware tile order: face 0 (rows 0-15, cols 0-15), face 1 (rows 0-15,
    cols 16-31), face 2, face 3; each face row-major."""
    shape = tiles.shape[:-2]
    t = tiles.reshape(*shape, 2, 16, 2, 16)  # (face_row, r, face_col, c)
    t = t.transpose(-3, -2)  # (face_row, face_col, r, c)
    return t.reshape(*shape, TILE * TILE)


def prepare_qkv_weights(weight, bias, device):
    """weight (2304, 768), bias (2304,) as in the HF checkpoint -> QkvWeights (see the class docstring)."""
    import torch

    weight = torch.as_tensor(weight).to(torch.float32)
    bias = torch.as_tensor(bias).to(torch.float32)
    assert tuple(weight.shape) == (N_TILES * TILE, HIDDEN) and tuple(bias.shape) == (N_TILES * TILE,)
    wt = torch.zeros(K_TILES * TILE, N_TILES * TILE, dtype=torch.float32)
    wt[:HIDDEN, :] = weight.T
    wt[BIAS_K_TILE * TILE, :] = bias  # row 0 of the bias K tile
    wt[:, : TENSOR_N_TILES * TILE] *= Q_SCALE  # q columns and b_q (exact: power of two)
    wt = wt.to(torch.bfloat16)
    tiles = wt.reshape(K_TILES, TILE, N_TILES, TILE).permute(0, 2, 1, 3)  # (k, n, 32, 32)
    faces = _to_face_order(tiles)  # (k, n, 1024)
    # n = p * WEIGHT_PAGE_TILES + j -> page (k, p), j-th tile
    pages = faces.reshape(K_TILES, WEIGHT_PAGES_PER_K_ROW, WEIGHT_PAGE_ELEMS)
    host = pages.reshape(1, 1, K_TILES * WEIGHT_PAGES_PER_K_ROW, WEIGHT_PAGE_ELEMS).contiguous()
    tensor = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return QkvWeights(tensor=tensor)


def prepare_rotary_tables(seqlen, device):
    """cos / sin of positions 0 .. seqlen - 1 (float64 on the host, stored fp32), see RotaryTables."""
    import torch

    seq_tiles = math.ceil(seqlen / TILE)
    inv_freq = 1.0 / (ROTARY_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.arange(seqlen, dtype=torch.float64)[:, None] * inv_freq[None, :]
    table = torch.zeros(seq_tiles * TILE, HEAD_DIM, dtype=torch.float64)
    table[:seqlen, :TILE] = angles.cos()
    table[:seqlen, TILE:] = angles.sin()
    tensor = ttnn.from_torch(
        table.to(torch.float32).reshape(1, 1, seq_tiles * TILE, HEAD_DIM),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return RotaryTables(tensor=tensor, seqlen=int(seqlen))


def head_dim_order():
    """The reference order: rotate-half is a tile-pair swap at equal lanes."""
    return tuple(range(HEAD_DIM))


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _norm_layout(shape, device):
    from ttnn.operations.nomic_residual_layer_norm import (
        input_memory_config as norm_input_memory_config,
        output_memory_config as norm_output_memory_config,
    )

    return norm_output_memory_config(tuple(shape), device, norm_input_memory_config(tuple(shape), device))


# The model's split layout (models/experimental/nomic_embed_text_v2_moe/tt/common.py: sharded_norm_split,
# sharded_norm_configs): legacy BLOCK_SHARDED L1, COL_MAJOR, MODEL_SPLIT_ROWS core rows (K parts) x C core columns,
# shard (32 * bh, HIDDEN / MODEL_SPLIT_ROWS) with bh = ceil(M / grid.x), C = ceil(M / bh). Rebuilt here (no import of
# the model package); test_nomic_qkv_rotary_heads_model_layout.py pins equality with the model's builder.
MODEL_SPLIT_ROWS = 8
MODEL_SPLIT_MAX_M = 128


def model_layout_memory_config(input_shape, device):
    """The model's split layout for x of `input_shape`, or None where the model runs the norm interleaved."""
    _, _, _, m = _check_shape(tuple(input_shape))
    grid = device.compute_with_storage_grid_size()
    if m > MODEL_SPLIT_MAX_M or MODEL_SPLIT_ROWS > grid.y or HIDDEN_TILES % MODEL_SPLIT_ROWS:
        return None
    block_h = math.ceil(m / grid.x)
    columns = math.ceil(m / block_h)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, MODEL_SPLIT_ROWS - 1))})
    shard = ttnn.ShardSpec(
        cores, (block_h * TILE, (HIDDEN_TILES // MODEL_SPLIT_ROWS) * TILE), ttnn.ShardOrientation.COL_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


def input_memory_config(input_shape, device):
    """op_layout == the norm's own layout (op_design.md, "op_layout decision")."""
    return _norm_layout(tuple(input_shape), device)


# ---------------------------------------------------------------------------
# Plan (the single source of the work split)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    batch: int
    seqlen: int
    seq_tiles: int  # S_t
    m_tiles: int  # M = B * S_t
    core_m_tiles: int  # rows a grid column owns
    num_columns: int
    num_m_blocks: int  # M blocks of a full column
    block_m_tiles: int
    num_lanes: int


def _check_shape(shape):
    if len(shape) != 4 or shape[1] != 1 or shape[-1] != HIDDEN:
        raise UnsupportedAxisValue(f"nomic_qkv_rotary_heads: shape {tuple(shape)} is not (B, 1, S, {HIDDEN})")
    batch, seqlen = int(shape[0]), int(shape[2])
    seq_tiles = math.ceil(seqlen / TILE)
    m = batch * seq_tiles
    if not (M_MIN <= m <= M_MAX):
        raise UnsupportedAxisValue(f"nomic_qkv_rotary_heads: M = {m} tile rows outside [{M_MIN}, {M_MAX}]")
    return batch, seqlen, seq_tiles, m


def _plan(shape, grid_x=GRID_X) -> Plan:
    batch, seqlen, seq_tiles, m = _check_shape(tuple(shape))
    core_m_tiles = math.ceil(m / grid_x)
    num_columns = math.ceil(m / core_m_tiles)
    num_m_blocks = math.ceil(core_m_tiles / BLOCK_M_TILES_MAX)
    block_m_tiles = math.ceil(core_m_tiles / num_m_blocks)
    return Plan(
        batch=batch,
        seqlen=seqlen,
        seq_tiles=seq_tiles,
        m_tiles=m,
        core_m_tiles=core_m_tiles,
        num_columns=num_columns,
        num_m_blocks=num_m_blocks,
        block_m_tiles=block_m_tiles,
        num_lanes=NUM_LANES,
    )


def _cb_bytes(block_m_tiles):
    """Closed form of l1_ledger.md (§ Total)."""
    bf16_pages = (
        X_BLOCK_DEPTH * block_m_tiles * K_BLOCK_TILES
        + K_TILES * LANE_N_TILES
        + OUT_DEPTH_BLOCKS * block_m_tiles * LANE_N_TILES
    )
    fp32_pages = ROTARY_TABLE_DEPTH * TABLE_TILES_PER_ROW * block_m_tiles + block_m_tiles * LANE_N_TILES
    return BF16_TILE_BYTES * bf16_pages + FP32_TILE_BYTES * fp32_pages


def circular_buffer_bytes(input_shape):
    """L1 a core's circular buffers take beyond the io tensors (bytes)."""
    return _cb_bytes(_plan(tuple(input_shape)).block_m_tiles)


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
    # model_layout is the model's split layout (BLOCK_SHARDED COL_MAJOR), read through the same gather with a
    # block-shard x source table (Refinement 1). op_layout is NOT a distinct placement of this op: input_memory_config
    # returns the norm's layout (op_design.md, "op_layout decision"), so _placement() tags such a tensor "ln_layout"
    # and can never produce "op_layout". Listing it here declared a value validate() never yields (its golden cells
    # all skip as "covered by placement ln_layout"); those cells are served under ln_layout.
    "placement": ["ln_layout", "model_layout"],
    "output_buffer": ["l1", "dram"],
    "output_mode": ["new", "preallocated"],
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
    try:
        if config == _norm_layout(shape, device):
            return "ln_layout"
    except Exception:  # noqa: BLE001 - the norm refuses (SupportRefusal or otherwise) shapes outside its domain
        pass
    model = model_layout_memory_config(shape, device)
    if model is not None and config == model:
        return "model_layout"
    return "other"


def _buffer_name(memory_config):
    if memory_config.is_sharded():
        return "sharded"
    if memory_config.buffer_type == ttnn.BufferType.L1:
        return "l1"
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        return "dram"
    return "other"


def validate(
    input_tensor,
    weights,
    tables,
    *,
    memory_config=None,
    output_tensors=None,
    compute_kernel_config=None,
):
    config = compute_kernel_config or default_compute_kernel_config()
    memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
    shape = tuple(input_tensor.shape)
    # Shape gate first: the placement check asks the norm for its layout, which only exists inside the M range.
    _check_shape(shape)
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(getattr(config, "fp32_dest_acc_en", False)),
        "layout": input_tensor.layout,
        "placement": _placement(input_tensor),
        "output_buffer": _buffer_name(memory_config),
        "output_mode": "preallocated" if output_tensors is not None else "new",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_qkv_rotary_heads: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_qkv_rotary_heads: unsupported combination (refinement candidate): {exc}")

    # Grid: the plan maps lanes to grid rows and M to grid columns of the 11 x 10 Blackhole grid.
    grid = input_tensor.device().compute_with_storage_grid_size()
    if (grid.x, grid.y) != (GRID_X, GRID_Y):
        raise UnsupportedAxisValue(f"nomic_qkv_rotary_heads: grid {grid.x}x{grid.y} is not {GRID_X}x{GRID_Y}")
    plan = _plan(shape, grid.x)
    if plan.num_lanes > grid.y:
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: not enough grid rows for the lanes")
    if _cb_bytes(plan.block_m_tiles) > CB_L1_BUDGET:
        raise UnsupportedAxisValue(
            f"nomic_qkv_rotary_heads: plan needs {_cb_bytes(plan.block_m_tiles)} B of CBs a core (> {CB_L1_BUDGET})"
        )
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.L1:
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: x must be in L1")

    # Prepared objects.
    if not isinstance(weights, QkvWeights) or weights.layout_tag != QkvWeights.layout_tag:
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: weights must come from prepare_qkv_weights")
    w = weights.tensor
    if (
        tuple(w.shape) != (1, 1, K_TILES * WEIGHT_PAGES_PER_K_ROW, WEIGHT_PAGE_ELEMS)
        or w.dtype != ttnn.bfloat16
        or w.layout != ttnn.ROW_MAJOR_LAYOUT
        or w.memory_config().is_sharded()
    ):
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: weights tensor is not in prepare_qkv_weights' layout")
    if not isinstance(tables, RotaryTables) or tables.layout_tag != RotaryTables.layout_tag:
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: tables must come from prepare_rotary_tables")
    if tables.seqlen < shape[2]:
        raise UnsupportedAxisValue(
            f"nomic_qkv_rotary_heads: tables cover {tables.seqlen} positions, the sequence has {shape[2]}"
        )
    t = tables.tensor
    if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or t.memory_config().is_sharded():
        raise UnsupportedAxisValue("nomic_qkv_rotary_heads: tables tensor is not in prepare_rotary_tables' layout")

    # Outputs.
    if output_tensors is not None:
        expected = (shape[0], HEADS, shape[2], HEAD_DIM)
        if len(output_tensors) != NUM_TENSORS:
            raise UnsupportedAxisValue("nomic_qkv_rotary_heads: output_tensors must be [q, k, v]")
        for out in output_tensors:
            if (
                tuple(out.shape) != expected
                or out.dtype != ttnn.bfloat16
                or out.layout != ttnn.TILE_LAYOUT
                or out.memory_config() != memory_config
            ):
                raise UnsupportedAxisValue(
                    f"nomic_qkv_rotary_heads: each output tensor must be {expected} bfloat16 TILE in {memory_config}"
                )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def nomic_qkv_rotary_heads(
    input_tensor,
    weights,
    tables,
    *,
    memory_config=None,
    output_tensors=None,
    compute_kernel_config=None,
):
    validate(
        input_tensor,
        weights,
        tables,
        memory_config=memory_config,
        output_tensors=output_tensors,
        compute_kernel_config=compute_kernel_config,
    )
    from .nomic_qkv_rotary_heads_program_descriptor import create_program_descriptor

    device = input_tensor.device()
    memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
    config = compute_kernel_config or default_compute_kernel_config()
    shape = tuple(input_tensor.shape)
    plan = _plan(shape, device.compute_with_storage_grid_size().x)

    if output_tensors is not None:
        query, key, value = output_tensors
    else:
        heads_shape = ttnn.Shape((shape[0], HEADS, shape[2], HEAD_DIM))
        # v, k, q in that order (v lands highest in L1, as the model allocates them).
        value, key, query = (
            ttnn.allocate_tensor_on_device(heads_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)
            for _ in range(NUM_TENSORS)
        )

    descriptor = create_program_descriptor(
        plan, input_tensor, weights.tensor, tables.tensor, (query, key, value), config
    )
    ttnn.generic_op([input_tensor, weights.tensor, tables.tensor, query, key, value], descriptor)
    return query, key, value
