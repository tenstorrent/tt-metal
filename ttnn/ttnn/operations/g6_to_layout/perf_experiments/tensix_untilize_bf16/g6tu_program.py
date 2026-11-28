# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-1 experiment tensix_untilize_bf16: the bf16 TILE -> ROW_MAJOR conversion through the Tensix
(reader: whole bf16 tiles -> cb_tile_window; compute: unpack A -> MOVA2D (Src zero flag pinned to
preserve) -> pack-untilize -> cb_row_block; writer: R1/R3's one write per live row segment).

Reuses the op's geometry and tile split (g6_to_layout_program_descriptor) read-only; the blocking and
every CB size are this experiment's knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_TILE_WINDOW = 0
CB_ROW_BLOCK = 1
CB_NAN_MARKS = 3  # unused: the writer's patch is off
TILE_BYTES = 2048
MAX_TRID_OUTSTANDING = 255
MAX_READS_PER_TILE = 2  # live-prefix reads: at most two per tile under one block trid


@dataclass(frozen=True)
class Knobs:
    max_block_cols: int = 0  # 0: no cap below the core's column span (R3's blocking)
    # Cap on block_cols_tiles when every image is shorter than one tile-row (H < 32): a row then
    # costs one write per block whatever its width, so narrower blocks are free on the write side and
    # let the untilize overlap the reads (c294: 20.7 -> 18.8 us at 8; c2036, H >= 32, keeps 1x24:
    # 8 costs it +2.5%). 0: no cap.
    short_max_block_cols: int = 8
    blocks_per_core: int = 16  # pd.PIPELINE_BLOCKS_PER_CORE
    read_ahead: int = 1  # blocks on the wire in the reader (1 = R3)
    window_blocks: int = 2  # nominal blocks in cb_tile_window (R3 TILE_BLOCK_DEPTH)
    row_depth: int = 2  # nominal blocks in cb_row_block
    # 1: also pin the Src zero flag to preserve after untilize_init (raw MATH call). Studied for the
    # bf16 CB format, where it changes nothing measurable (509/65536 patterns altered either way: the
    # unpack/pack float conversion, not MOVA2D's flush). With UInt16 CBs hw_configure already selects
    # preserve (requires_disabled_src_zero_flag(UInt16)), so the recommended variant leaves it off.
    preserve_zero_flag: int = 0
    # Data format the CBs declare to the unpacker / packer: "bf16" (Float16_b: unpack/pack convert
    # floats, -0.0/subnormals -> +0 and NaN -> inf) or "uint16" (the same bytes as Integer 16: no float
    # conversion anywhere on the Tensix path).
    cb_format: str = "uint16"
    live_reads: int = 1  # 1: R1's live face-row prefix reads; 0: whole tiles (R3)
    diag_mode: int = 0  # >0: kernels/g6tu_diag_compute.cpp in that mode (precision census only)


@dataclass
class Plan:
    geometry: object
    all_cores: object
    core_args: list
    num_cores: int
    block_rows_tiles: int
    block_cols_tiles: int
    knobs: Knobs
    cb_bytes: dict = field(default_factory=dict)
    out_l1_share: int = 0

    @property
    def block_tiles(self):
        return self.block_rows_tiles * self.block_cols_tiles


def compute_plan(shape, grid, *, output_in_l1, num_l1_banks, out_alignment, knobs: Knobs):
    g = pd.tensor_geometry(shape)
    gx, gy = grid
    out_l1_share = 0
    if output_in_l1:
        slot = pd.out_slot_bytes(g, out_alignment)
        banks_per_core = max(1, pd._div_up(num_l1_banks, gx * gy))
        out_l1_share = pd._div_up(g.out_pages, num_l1_banks) * slot * banks_per_core
    per_tile = (knobs.window_blocks + knobs.row_depth) * TILE_BYTES
    max_block_tiles = min((pd.L1_BUDGET_BYTES - out_l1_share) // per_tile, MAX_TRID_OUTSTANDING // MAX_READS_PER_TILE)
    assert max_block_tiles >= 1
    num_cores, all_cores, core_args, core_tiles_max = pd._tile_split(gx, gy, g)
    cols = g.tensor_cols_tiles
    cap = knobs.max_block_cols if knobs.max_block_cols > 0 else cols
    if g.height < pd.TILE_HW and knobs.short_max_block_cols > 0:
        cap = min(cap, knobs.short_max_block_cols)
    block_cols = min(cols, core_tiles_max, max_block_tiles, cap)
    if block_cols < cols:
        block_rows = 1
    else:
        rows = pd._div_up(core_tiles_max // cols, knobs.blocks_per_core)
        block_rows = max(1, min(rows, max_block_tiles // block_cols))
    plan = Plan(g, all_cores, core_args, num_cores, block_rows, block_cols, knobs, out_l1_share=out_l1_share)
    plan.cb_bytes = {
        CB_TILE_WINDOW: knobs.window_blocks * plan.block_tiles * TILE_BYTES,
        CB_ROW_BLOCK: knobs.row_depth * plan.block_tiles * TILE_BYTES,
    }
    return plan


def plan_for(input_tensor, output_memory_config, knobs):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    output_in_l1 = output_memory_config.buffer_type == ttnn.BufferType.L1
    return compute_plan(
        tuple(input_tensor.shape),
        (int(grid.x), int(grid.y)),
        output_in_l1=output_in_l1,
        num_l1_banks=ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks if output_in_l1 else 1,
        out_alignment=ttnn.get_l1_alignment() if output_in_l1 else ttnn.get_dram_alignment(),
        knobs=knobs,
    )


def _cb(index, total, page, cores, fmt):
    return ttnn.CBDescriptor(
        total_size=total,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def build_descriptor(plan: Plan, input_tensor, output_tensor):
    g = plan.geometry
    k = plan.knobs
    cores = plan.all_cores
    geometry_ct = [
        g.tensor_cols_tiles,
        g.image_rows_tiles,
        g.height,
        g.width,
        plan.block_rows_tiles,
        plan.block_cols_tiles,
        pd.TILE_HW,  # tile_row_stage_rows: the packer emits whole 32-row tile-rows
    ]
    rt = ttnn.RuntimeArgs()
    for (x, y), args in plan.core_args:
        rt[x][y] = list(args)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6tu_tile_reader.cpp"),
        core_ranges=cores,
        compile_time_args=geometry_ct
        + [CB_TILE_WINDOW, k.read_ahead, k.window_blocks, k.live_reads]
        + ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
        runtime_args=rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_staged_writer.cpp"),
        core_ranges=cores,
        compile_time_args=geometry_ct
        + [CB_ROW_BLOCK, 1, 0, 0, CB_NAN_MARKS]
        + ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
        runtime_args=rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    if k.diag_mode:
        compute_source, compute_extra = "g6tu_diag_compute.cpp", [CB_TILE_WINDOW, CB_ROW_BLOCK, k.diag_mode]
    else:
        compute_source, compute_extra = "g6tu_untilize_compute.cpp", [
            CB_TILE_WINDOW,
            CB_ROW_BLOCK,
            k.preserve_zero_flag,
        ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / compute_source),
        core_ranges=cores,
        compile_time_args=geometry_ct + compute_extra,
        runtime_args=rt,
        # The op's fixed config (no compute_kernel_config in its signature): unchanged.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            math_approx_mode=False,
        ),
    )
    cb_fmt = {"bf16": ttnn.bfloat16, "uint16": ttnn.uint16}[k.cb_format]
    cbs = [
        _cb(CB_TILE_WINDOW, plan.cb_bytes[CB_TILE_WINDOW], TILE_BYTES, cores, cb_fmt),
        _cb(CB_ROW_BLOCK, plan.cb_bytes[CB_ROW_BLOCK], TILE_BYTES, cores, cb_fmt),
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run(input_tensor, output_memory_config, knobs, output_tensor=None, plan=None):
    """One launch of the candidate; allocates the output unless one is passed."""
    if plan is None:
        plan = plan_for(input_tensor, output_memory_config, knobs)
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            input_tensor.device(),
            output_memory_config,
        )
    return ttnn.generic_op([input_tensor, output_tensor], build_descriptor(plan, input_tensor, output_tensor))


def describe(plan):
    return (
        f"cores={plan.num_cores} block={plan.block_rows_tiles}x{plan.block_cols_tiles} "
        f"l1={sum(plan.cb_bytes.values()) + plan.out_l1_share} knobs={plan.knobs}"
    )
