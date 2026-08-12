# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, args).

This file owns the Blocking Model (op_design.md §1). Every knob is defined
EXACTLY ONCE here and every dependent quantity (CB page counts, loop trip
counts, grid sizing, kernel args) is computed *from* it — never from a whole-op
dimension and never as a magic literal:

  knob                 symbol             defined as
  -------------------  -----------------  ------------------------------------
  read byte target     TARGET_READ_BYTES  module constant (1024, measured — see below)
  max block width      WT_BLOCK_MAX       max(2, TARGET_READ_BYTES // (32*elem))
  block width (tiles)  WT_BLOCK           min(Wt, WT_BLOCK_MAX)
  tail block width     WT_TAIL            Wt - (n_wchunks-1)*WT_BLOCK
  column-blocks/row    n_wchunks          ceil(Wt / WT_BLOCK)
  grid cores           grid_cores         1 if not use_multicore else grid.x*y
  CB depth             CB_DEPTH           2 if use_double_buffer and it fits L1
  cast flag            NEEDS_CAST         out_dtype != in_dtype

One block = one output tile-row x WT_BLOCK output tile-columns. Blocks are
linearized `b = wchunk * nt_h + r` and that linear space is spread across the
grid by `split_work_to_cores(grid, nt_h * n_wchunks, row_wise=True)`, which
subsumes both distribution regimes with no gate expression: when
`Wt <= WT_BLOCK_MAX` (`n_wchunks == 1`) the block index *is* the tile-row index,
so it degenerates to the pure height split; when `nt_h == 1` (wide-short) it
degenerates to the pure width split and still fills the grid.
"""

from pathlib import Path

import math

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names are the primary identifier) -------------------
CB_INPUT_STICKS = 0  # reader -> compute : row-major sticks, tile-sized pages
CB_OUTPUT_TILES = 16  # compute -> writer : tiled output pages

# --- Knob: read transaction byte target ------------------------------------
# Expressed in BYTES so every dtype lands on the same measured sweet spot
# (bf16 -> WT_BLOCK 16, fp32 -> 8, uint8 -> 32); sweep this ONE line to move the
# transaction size, and `_bench_tilize.py`'s `lever_b6_read_*` arms re-measure it.
#
# 1024, not the 512 B "one-packet" value: 512 B is NOC_MAX_BURST_SIZE on
# **Wormhole**; on Blackhole (this box) NOC_MAX_BURST_SIZE is 256 words x 64 B =
# 16 KB, so the one-packet argument does not bind at 512 B and the sweep decides.
# Measured (grid 11x10, bf16, DEVICE KERNEL DURATION, phase-0 bench):
#   [1,1,2048,2048]  128 B 152.6us | 256 B 86.3us | 512 B 55.9us | 1024 B 44.5us
#                    | 2048 B 44.7us | 4096 B 47.0us
#   [1,1,32,16384]   512 B 7.69us | 1024 B 7.41us | 2048 B 7.87us | 4096 B 9.37us
#   [1,1,2048,64] and [1,1,32,64] are unaffected (WT_BLOCK = min(Wt, ...) clamps).
# 1024 B is the joint optimum and regresses no benched regime.
TARGET_READ_BYTES = 1024

TILE_WIDTH = 32  # a tile is ALWAYS 32 wide; only its height varies

# --- Lever knobs (the perf-gate counterfactual surface) --------------------
# Every performance lever this op lands is a NAMED knob here, so its off-arm is
# re-runnable from the bench (`levers=dict(<knob>=0)`) instead of being an ad-hoc
# kernel edit. The defaults below ARE the production path — `dict(DEFAULT_LEVERS)`
# reproduces the shipped kernel byte-for-byte, so an unmeasured knob costs
# nothing. `stub_*` are the /perf-measure ablation arms (keep the sync
# scaffolding, drop one payload) and are never on in production.
DEFAULT_LEVERS = {
    "multicore": 1,  # A0: full grid vs a single core (also the user-facing kwarg)
    "width_split": 1,  # A0/A1: 2-D linearization (b = wchunk*nt_h + r) vs height-only
    "row_wise": 1,  # A1: spread cores across the DRAM-facing (row) axis
    "target_read_bytes": TARGET_READ_BYTES,  # B6/B7: read transaction size -> WT_BLOCK
    "coalesce_writes": 1,  # B5: whole-tile-page writes vs per-face writes
    "barrier_per_block": 1,  # B7: one barrier per block vs one per transaction
    "noc_split": 1,  # B9: reader NoC0 / writer NoC1 vs swapped
    "double_buffer": 1,  # C16: CB depth 2 vs 1  (also the user-facing kwarg)
    "stub_read": 0,  # ablation: drop the NoC read payload
    "stub_compute": 0,  # ablation: drop the tilize math
    "stub_write": 0,  # ablation: drop the NoC write payload
}


def resolve_levers(levers=None) -> dict:
    """Merge caller overrides onto DEFAULT_LEVERS, rejecting unknown knobs."""
    resolved = dict(DEFAULT_LEVERS)
    if levers:
        unknown = set(levers) - set(DEFAULT_LEVERS)
        if unknown:
            raise ValueError(f"tilize: unknown lever knob(s) {sorted(unknown)}")
        resolved.update(levers)
    return resolved


# Fallback per-core CB budget (bytes) used only when the device cannot be
# queried for its real CB limit. Depth-2 auto-falls back to depth-1 rather than
# OOMing (master.md C16 / the ttnn.concat precedent).
_CB_BUDGET_FALLBACK_BYTES = 400 * 1024


def _round_up(value: int, multiple: int) -> int:
    """`tt::round_up`. Local because this build does not export ttnn.round_up."""
    return ((value + multiple - 1) // multiple) * multiple


def _div_up(a: int, b: int) -> int:
    """`tt::div_up`. Local because this build does not export ttnn.div_up."""
    return (a + b - 1) // b


def wt_block_max(elem_size: int, target_read_bytes: int = TARGET_READ_BYTES) -> int:
    """Max tiles per compute block for this element size.

    `max(2, ...)` keeps row_bytes >= 64 B (2 x the 32 B DRAM read-alignment
    unit) even for 1-byte dtypes.
    """
    return max(2, target_read_bytes // (TILE_WIDTH * elem_size))


def _cb_budget_bytes(device) -> int:
    """Per-core L1 bytes this op may spend on CBs.

    Queried from the device when the build exposes a device-info binding; the
    module constant is the fallback. Only ever used to decide whether depth-2
    fits (never to size a CB), so a conservative value degrades to depth-1
    rather than to a wrong CB.
    """
    info = getattr(ttnn, "get_device_info", None)
    if info is not None:
        try:
            return int(info(device).cb_limit)
        except Exception:  # pragma: no cover - device info is best-effort
            pass
    return _CB_BUDGET_FALLBACK_BYTES


def tile_geometry(shape, tile_height: int):
    """Alignment-aware tile geometry (op_design.md §5.1).

    `ceil` and per-image from the start: in TILE layout each image is tile-padded
    independently, so nt_h = nimg * ceil(H/tile_h), NOT floor(nimg*H/tile_h).
    """
    shape = list(shape)
    H = shape[-2] if len(shape) >= 2 else 1
    W = shape[-1] if len(shape) >= 1 else 1
    Hp = _round_up(H, tile_height)
    Wp = _round_up(W, TILE_WIDTH)
    nimg = math.prod(shape[:-2]) if len(shape) > 2 else 1
    nt_h = nimg * (Hp // tile_height)
    Wt = Wp // TILE_WIDTH
    return nt_h, Wt, Hp, Wp


def blocking(shape, tile_height: int, elem_size: int, target_read_bytes: int = TARGET_READ_BYTES):
    """The whole Blocking Model for one call, derived from the knobs above."""
    nt_h, Wt, _, _ = tile_geometry(shape, tile_height)
    wt_block = min(Wt, wt_block_max(elem_size, target_read_bytes))
    n_wchunks = _div_up(Wt, wt_block)
    wt_tail = Wt - (n_wchunks - 1) * wt_block
    total_blocks = nt_h * n_wchunks
    tail_block_start = (n_wchunks - 1) * nt_h
    return {
        "nt_h": nt_h,
        "Wt": Wt,
        "wt_block": wt_block,
        "wt_tail": wt_tail,
        "n_wchunks": n_wchunks,
        "total_blocks": total_blocks,
        "tail_block_start": tail_block_start,
    }


def plan_cores(device, total_blocks: int, *, use_multicore: bool, row_wise: bool = True, max_cores=None):
    """Core assignment: `grid_cores` is a PARAMETER whose trivial value is 1.

    Returns (cores, all_cores, per_core_blocks) where per_core_blocks[i] is the
    block count for cores[i] and the list is in the SAME order the split produced
    — `row_wise` MUST match on both calls or the runtime-arg assignment silently
    mismatches the split's group order (master.md A1 / op_design risk 18).
    """
    grid = device.compute_with_storage_grid_size()
    split_grid = grid if use_multicore else ttnn.CoreCoord(1, 1)

    if max_cores is not None:
        # Counterfactual arm only (the height-only-split off-arm caps the core
        # count at nt_h). The production path always goes through
        # ttnn.split_work_to_cores below.
        num_cores = max(1, min(max_cores, total_blocks, split_grid.x * split_grid.y))
        cores = ttnn.grid_to_cores(num_cores, split_grid.x, split_grid.y, row_wise)
        base, rem = divmod(total_blocks, num_cores)
        per_core_blocks = [base + (1 if k < rem else 0) for k in range(num_cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        return cores, all_cores, per_core_blocks

    (
        num_cores,
        all_cores,
        core_group_1,
        _core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(split_grid, total_blocks, row_wise)

    cores = ttnn.grid_to_cores(num_cores, split_grid.x, split_grid.y, row_wise)
    per_core_blocks = [blocks_per_core_g1 if core_group_1.contains(core) else blocks_per_core_g2 for core in cores]
    return cores, all_cores, per_core_blocks


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    use_multicore: bool = True,
    use_double_buffer: bool = True,
    tile_height: int = 32,
    levers=None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    lv = resolve_levers(levers)

    # ---------- 1. geometry + the block knobs ----------
    elem_size = input_tensor.element_size()
    blk = blocking(list(input_tensor.shape), tile_height, elem_size, lv["target_read_bytes"])
    nt_h = blk["nt_h"]
    Wt = blk["Wt"]
    wt_block = blk["wt_block"]
    wt_tail = blk["wt_tail"]
    n_wchunks = blk["n_wchunks"]
    total_blocks = blk["total_blocks"]
    tail_block_start = blk["tail_block_start"]

    # One tile-column of one stick. The reader derives its per-block transfer
    # size as `w * tile_row_bytes`, so the block width is never restated.
    tile_row_bytes = TILE_WIDTH * elem_size

    in_tile_bytes = tile_height * TILE_WIDTH * elem_size  # RM input is never block-float
    out_tile_bytes = output_tensor.buffer_page_size()

    needs_cast = int(output_tensor.dtype != input_tensor.dtype)

    # ---------- 2. CB depth knob (a distinct knob from block factor) ----------
    depth2_bytes = 2 * wt_block * (in_tile_bytes + out_tile_bytes)
    depth2_fits_l1 = depth2_bytes <= _cb_budget_bytes(device)
    want_depth2 = use_double_buffer and bool(lv["double_buffer"])
    cb_depth = 2 if (want_depth2 and depth2_fits_l1) else 1
    cb_pages = cb_depth * wt_block  # >= wt_block >= wt_tail: no reader deadlock

    # ---------- 3. core assignment ----------
    # width_split == 0 is the height-only-split off-arm: cap the core count at
    # nt_h, which is byte-identical to the default when n_wchunks == 1 and
    # collapses a wide-short tensor onto min(nt_h, grid) cores — the exact
    # regression the 2-D linearization exists to prevent.
    cores, all_cores, per_core_blocks = plan_cores(
        device,
        total_blocks,
        use_multicore=use_multicore and bool(lv["multicore"]),
        row_wise=bool(lv["row_wise"]),
        max_cores=None if lv["width_split"] else nt_h,
    )

    # ---------- 4. circular buffers ----------
    tile_descriptor = ttnn.TileDescriptor(tile_height, TILE_WIDTH)

    cb_input_sticks = ttnn.CBDescriptor(
        total_size=cb_pages * in_tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_STICKS,
                data_format=input_tensor.dtype,
                page_size=in_tile_bytes,
                tile=tile_descriptor,
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=cb_pages * out_tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=out_tile_bytes,
                tile=tile_descriptor,
            )
        ],
    )

    # ---------- 5. kernels ----------
    # CT args: scalar args first, TensorAccessorArgs appended LAST (master.md
    # D18). RT args carry only buffer addresses + the per-core block range
    # (D19), so a second call with the same spec hits the program cache.
    reader_ct_args = [
        CB_INPUT_STICKS,
        nt_h,
        n_wchunks,
        tile_height,
        tile_row_bytes,
        wt_block,
        wt_tail,
        lv["barrier_per_block"],
        lv["stub_read"],
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        nt_h,
        n_wchunks,
        Wt,
        wt_block,
        wt_tail,
        out_tile_bytes,
        lv["coalesce_writes"],
        lv["stub_write"],
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [
        CB_INPUT_STICKS,
        CB_OUTPUT_TILES,
        wt_block,
        wt_tail,
        needs_cast,
        lv["stub_compute"],
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    block_start = 0
    for core, num_blocks in zip(cores, per_core_blocks):
        # A core's contiguous [b0, b0+nb) crosses the full/tail column-block
        # boundary at most once, because the tail column-block occupies the
        # contiguous suffix [tail_block_start, total_blocks) of the linear space.
        n_full = min(max(tail_block_start - block_start, 0), num_blocks)
        n_tail = num_blocks - n_full

        reader_rt[core.x][core.y] = [in_addr, block_start, num_blocks]
        writer_rt[core.x][core.y] = [out_addr, block_start, num_blocks]
        compute_rt[core.x][core.y] = [n_full, n_tail]
        block_start += num_blocks

    # B9 off-arm: swap the two configs so the read stream lands on the writer's
    # RISC/NoC and vice versa.
    reader_config = ttnn.ReaderConfigDescriptor() if lv["noc_split"] else ttnn.WriterConfigDescriptor()
    writer_config = ttnn.WriterConfigDescriptor() if lv["noc_split"] else ttnn.ReaderConfigDescriptor()

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=reader_config,  # NCRISC / NoC0 by default (master.md B9)
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=writer_config,  # BRISC / NoC1 by default (master.md B9)
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_sticks, cb_output_tiles],
    )
