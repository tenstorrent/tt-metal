# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""writer_starvation — isolated bake-off for the tilize PRODUCER -> WRITER handoff.

The writer sits idle for 5952 of its 9710 ns BRISC span on the flagged profile
`[1,1,32,16384]`, because two handoffs are coarse:

  (a) compute -> writer: `compute_kernel_lib::tilize` pushes the whole
      `block_width_tiles`-wide tile-row at once, so the first NoC write cannot
      be issued until the LAST tile of the row is packed.
  (b) reader -> compute: the reader pushes only after all `tile_h` sticks of the
      tile-row have landed behind one barrier.

Variants (see each kernel's head for the mechanism and the raw-LLK gap):

  baseline             the op's current pipeline, reconstructed.
  raw_full             CONTROL. Identical decomposition to `baseline` but with
                       the tilize LLK sequence written out by hand at
                       `push_tiles == block_width_tiles`. Isolates "raw LLK vs
                       helper" from "finer push", so any fine-push delta is
                       attributable.
  fine_half            (a) at `push_tiles = block_width_tiles / 2`.
  fine_min             (a) at the finest legal push (`unit_dim` = 2 tiles on WH,
                       1 tile at `block_width_tiles == 1`).
  split_read           (b) the tile-row's stick reads cut into two half-barriers.
  combo                (c) fine_half + split_read together.
  baseline_no_row_rot  CONTROL for the fine path's forced ascending row order.
                       Inert wherever a batch is one tile-row (the whole focus
                       regime); priced explicitly where it is not.

MEASURED VERDICT (Wormhole B0 n150, 8x8 = 64/64 cores, 1 GHz, 12 DRAM banks;
`DEVICE KERNEL DURATION [ns]`, medians; raw logs in `results/`).

  (a) fine push            NULL.  focus 12288 vs baseline 12521 ns (5 reps) and
                                  12396 vs 12624 (7 reps) — the same +-2% the
                                  `raw_full` and `baseline_no_row_rot` CONTROLS
                                  show against the same baseline.
  (b) split read barrier   REGRESSION. -19.0% at [1,1,16384,32] (23233 vs 19527),
                                  -9.5% at [1,1,2048,64] (5389 vs 4921), flat on
                                  the wide shapes. Splitting one barrier into two
                                  serializes the drain of the first half against
                                  the issue of the second; on a narrow stick the
                                  read is issue/latency bound and pays for it.
  (c) combo                REGRESSION, inherited from (b) (-5.0% at [1,1,2048,2048]).

WHY (a) IS BOUNDED SO LOW — the decisive zone number, focus shape:
  `writer_wait_first` (entry to the batch's FIRST NoC write)
      baseline (whole 8-tile row)  6091 ns
      push=4                       6025 ns   (-66 ns)
      push=2                       6002 ns   (-89 ns)
  `writer_wait_rest`               37 ns (push=4) / 106 ns total (push=2)
So the row's TRAILING tiles are already packed by the time the writer has
finished issuing the first group: the finer push does exactly what it is
supposed to, and the whole thing it can buy is 66-89 ns of a 12.4 us wall
(0.5-0.7%), an order of magnitude under this box's ~+-2.5% run-to-run band.
The reason is visible in the same table: `writer_wait_first` (6091) is
essentially `reader_read_block` (5300) — the writer is starved by the READ, not
by the compute -> writer push quantum. Only ~790 ns of its 6091 ns wait is
downstream of the reader's push at all, and the pack of 8 tiles is a small part
of that.

`raw_full` (the hand-written LLK sequence at push == block_width_tiles) ties the
helper to within noise on every shape measured (0.982x - 1.007x), so none of the
above is an artifact of bypassing `compute_kernel_lib::tilize`.

GEOMETRY is READ (never mutated) from the real op's own `derive_plan`, so every
variant runs on EXACTLY the op's block plan for a given shape. Precision is the
op's own too: `fp32_dest_acc_en` comes off the plan and is identical in every
variant, as are the dtypes and the (default) fidelity.
"""

from pathlib import Path

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as tilize_pd

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INPUT_ROWS = 0
CB_OUTPUT_TILES = 1

VARIANTS = (
    "baseline",
    "raw_full",
    "fine_half",
    "fine_min",
    "split_read",
    "combo",
    "baseline_no_row_rot",
)


class Inexpressible(Exception):
    """This variant has no distinct realization at this geometry."""


def derive_geometry(input_tensor, output_tensor, device):
    grid = device.compute_with_storage_grid_size()
    plan = tilize_pd.derive_plan(input_tensor, output_tensor, low_l1=False, grid=grid)
    if plan.tail_group is not None:
        raise Inexpressible("ragged column tail (two core ranges) is out of this bench's scope")
    if plan.input_native or plan.output_native or plan.pad_active or plan.is_retile:
        raise Inexpressible("only the plain interleaved->interleaved path is in scope")
    if plan.input_pages_per_row > 1:
        raise Inexpressible("the strided reader branch is out of this bench's scope")
    return plan


def _unit_dim(plan) -> int:
    """The helper's own Wormhole `unit_dim` (api/compute/tilize.h)."""
    return 1 if plan.block_width_tiles == 1 else 2


def _dest_capacity(plan) -> int:
    """Tiles per dest section — the helper's own `dest_size`."""
    return 4 if plan.fp32_dest_acc_en else 8


def push_tiles_for(plan, variant: str) -> int:
    """Output push quantum, in tiles. Must divide block_width_tiles, be a
    multiple of unit_dim and fit one dest section."""
    bw = plan.block_width_tiles
    unit = _unit_dim(plan)
    cap = _dest_capacity(plan)

    if variant == "raw_full":
        push = min(bw, cap)
        # The helper's own decomposition of a row wider than one dest section is
        # several dest sections; reproduce the largest legal one.
        while bw % push or push % unit:
            push -= 1
        return push

    if variant in ("fine_half", "combo"):
        if bw == 1:
            return 1
        push = max(unit, bw // 2)
        push = min(push, cap)
        while bw % push or push % unit:
            push -= 1
        return push

    if variant == "fine_min":
        return unit

    raise ValueError(variant)


def variant_spec(plan, variant: str):
    """(read_barriers, compute_kernel, push_tiles, wait_tiles, row_rotate)."""
    bw = plan.block_width_tiles
    if variant == "baseline":
        return 1, "compute_baseline.cpp", 0, 0, 1
    if variant == "baseline_no_row_rot":
        return 1, "compute_baseline.cpp", 0, 0, 0
    if variant == "split_read":
        return 2, "compute_baseline.cpp", 0, 0, 1

    push = push_tiles_for(plan, "raw_full" if variant == "raw_full" else variant)
    if variant == "raw_full":
        return 1, "compute_fine_push.cpp", push, 0, 1
    if push >= bw:
        raise Inexpressible(f"{variant}: push_tiles would be the whole row at block_width_tiles={bw}")
    if variant == "fine_half":
        return 1, "compute_fine_push.cpp", push, push, 0
    if variant == "fine_min":
        return 1, "compute_fine_push.cpp", push, push, 0
    if variant == "combo":
        return 2, "compute_fine_push.cpp", push, push, 0
    raise ValueError(variant)


def _compute_config(plan):
    # The user's precision contract, taken straight off the op's own plan and
    # held IDENTICAL across every variant.
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=bool(plan.fp32_dest_acc_en),
        dst_full_sync_en=False,
    )


def _cb(index, dtype, page_bytes, pages, cores, tile_desc):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes, tile=tile_desc)
        ],
    )


def build_descriptor(input_tensor, output_tensor, plan, variant: str, zones: bool = False):
    if variant not in VARIANTS:
        raise ValueError(f"writer_starvation bench: unknown variant {variant!r}")
    read_barriers, compute_src, push_tiles, wait_tiles, row_rotate = variant_spec(plan, variant)

    tile_desc = ttnn.TileDescriptor(plan.tile_h, 32)
    cores = plan.all_cores
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    in_accessor_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    defines = [("TILIZE_BENCH_ZONES", "1")] if zones else []

    cbs = [
        _cb(
            CB_INPUT_ROWS,
            input_tensor.dtype,
            plan.in_page_bytes,
            plan.input_depth_rows * plan.block_width_tiles,
            cores,
            tile_desc,
        ),
        _cb(
            CB_OUTPUT_TILES,
            output_tensor.dtype,
            plan.out_page_bytes,
            plan.output_depth_batches * plan.write_rows_per_barrier * plan.block_width_tiles,
            cores,
            tile_desc,
        ),
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, start_block_id, num_blocks, block_stride in plan.assignment:
        reader_rt[core.x][core.y] = [in_addr, start_block_id, num_blocks, block_stride]
        writer_rt[core.x][core.y] = [out_addr, start_block_id, num_blocks, block_stride]
        compute_rt[core.x][core.y] = [start_block_id, num_blocks, block_stride]

    reader_ct = [
        CB_INPUT_ROWS,
        plan.block_width_tiles,
        plan.tile_h,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.block_row_bytes,
        read_barriers,
    ] + in_accessor_args

    writer_ct = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.tensor_col_tiles,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.write_rows_per_barrier,
        plan.out_page_bytes,
        wait_tiles,
        row_rotate,
    ] + out_accessor_args

    compute_ct = [
        CB_INPUT_ROWS,
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
    ]
    if compute_src == "compute_fine_push.cpp":
        compute_ct.append(push_tiles)

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
            defines=defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
            defines=defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / compute_src),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=_compute_config(plan),
            defines=defines,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
