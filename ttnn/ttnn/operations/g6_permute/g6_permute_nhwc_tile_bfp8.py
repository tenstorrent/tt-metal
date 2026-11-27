# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime nhwc_tile_bfp8_transpose: plan (group split, depth knob) and ProgramDescriptor.

TILE bfloat8_b x [N, H, W, C], dims (0, 3, 1, 2): out[n, c, h, w] = x[n, h, w, c].
Work unit = group (n, ht, wt, ct): input tiles (n, h, wt, ct) for the 32 h of tile row ht become
output tiles (n, c, ht, wt) for the 32 c of tile column ct. Compute transposes each input tile in
DEST and packs it untilized in bfloat16, so input tile h becomes row h of a [32, 32 * 32] row-major
block whose 32-column chunk c is output tile c. Tilizing that block is the row exchange; the packer
writes it in bfloat8_b and gives each 16-value group the largest exponent of its values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn

from .g6_permute_program_descriptor import (
    KERNEL_DIR,
    L1_OP_BUDGET_BYTES,
    READER_KERNEL_INDEX,
    TILE_ROWS,
    WRITER_KERNEL_INDEX,
    _div_up,
    out_l1_bytes_per_core,
    split_units,
    worker_grid,
)

REGIME = "nhwc_tile_bfp8_transpose"
DIMS = (0, 3, 1, 2)

COMPUTE_KERNEL_INDEX = 2

CB_INPUT_TILES = 0
CB_ROW_BLOCK = 1
CB_OUTPUT_TILES = 2

# A group is 32 input tiles in and 32 output tiles out: the tilize width of the row exchange.
GROUP_TILES = TILE_ROWS
# Buffer-depth knob of cb_input_tiles and cb_output_tiles, in groups; the allocated depth is
# min(GROUP_DEPTH, max_groups_per_core), lowered to 1 when two groups do not fit the L1 budget.
GROUP_DEPTH = 2
# Push granularity of the reader inside a group, in input tiles (a divisor of GROUP_TILES): compute
# starts transposing the first chunk while the rest is read. GROUP_TILES pushes the group at once.
READ_CHUNK_TILES = 8


@dataclass
class NhwcPlan:
    regime: str
    num_cores: int
    tensor_groups: int
    groups_per_core_g1: int
    groups_per_core_g2: int
    group_depth: int
    l1_bytes_per_core: int
    out_l1_bytes_per_core: int
    core_roles: dict
    shape: tuple = ()
    tensor_ht: int = 0
    tensor_wt: int = 0
    tensor_ct: int = 0
    in_page_bytes: int = 0
    row_page_bytes: int = 0
    read_chunk_tiles: int = GROUP_TILES
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_group, num_groups)]


def knob_key():
    """Host knobs that reach this regime's CT args and CB sizes; part of the program cache key."""
    return (GROUP_DEPTH, READ_CHUNK_TILES)


def _footprint(depth, in_page_bytes, row_page_bytes):
    return GROUP_TILES * (2 * depth * in_page_bytes + row_page_bytes)


def build_plan(input_tensor, canonical_dims) -> NhwcPlan:
    assert tuple(canonical_dims) == DIMS, f"regime {REGIME} serves dims {DIMS}"
    assert input_tensor.dtype == ttnn.bfloat8_b and input_tensor.layout == ttnn.TILE_LAYOUT
    assert GROUP_DEPTH >= 1
    assert GROUP_TILES % READ_CHUNK_TILES == 0, f"READ_CHUNK_TILES {READ_CHUNK_TILES} must divide {GROUP_TILES}"
    N, H, W, C = (int(d) for d in input_tensor.shape)
    tensor_ht = _div_up(H, TILE_ROWS)
    tensor_wt = _div_up(W, TILE_ROWS)
    tensor_ct = _div_up(C, TILE_ROWS)
    assert N * H * tensor_wt * tensor_ct == input_tensor.buffer_num_pages()
    tensor_groups = N * tensor_ht * tensor_wt * tensor_ct

    in_page_bytes = input_tensor.buffer_page_size()
    # The row block holds bfloat16: a bfloat8_b value is exact in bfloat16, and its exponents are
    # recomputed when the output groups are packed.
    row_page_bytes = TILE_ROWS * TILE_ROWS * 2

    out_l1 = out_l1_bytes_per_core(
        input_tensor.memory_config().buffer_type,
        N * C * tensor_ht * tensor_wt,
        input_tensor.buffer_aligned_page_size(),
        input_tensor.device(),
    )
    budget = L1_OP_BUDGET_BYTES - out_l1

    num_cores, all_cores, groups_g1, groups_g2, assignment = split_units(
        worker_grid(input_tensor.device()), tensor_groups
    )
    depth = min(GROUP_DEPTH, groups_g1)
    while depth > 1 and _footprint(depth, in_page_bytes, row_page_bytes) > budget:
        depth -= 1
    footprint = _footprint(depth, in_page_bytes, row_page_bytes)
    if footprint > budget:
        raise RuntimeError(
            f"g6_permute: shape {[N, H, W, C]} leaves no room for one 32-tile group within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B, one group {footprint} B)"
        )

    core_roles = {
        f"{c.x},{c.y}": f"{REGIME} reader+compute+writer groups (n, ht, wt, ct) [{s}, {s + n})"
        for c, s, n in assignment
    }

    return NhwcPlan(
        regime=REGIME,
        num_cores=num_cores,
        tensor_groups=tensor_groups,
        groups_per_core_g1=groups_g1,
        groups_per_core_g2=groups_g2,
        group_depth=depth,
        l1_bytes_per_core=footprint,
        out_l1_bytes_per_core=out_l1,
        core_roles=core_roles,
        shape=(N, H, W, C),
        tensor_ht=tensor_ht,
        tensor_wt=tensor_wt,
        tensor_ct=tensor_ct,
        in_page_bytes=in_page_bytes,
        row_page_bytes=row_page_bytes,
        read_chunk_tiles=READ_CHUNK_TILES,
        all_cores=all_cores,
        assignment=assignment,
    )


def _start_coords(plan: NhwcPlan, group):
    """(n, ht, wt, ct) of group `group`; the kernels advance them without dividing."""
    rest, ct = divmod(group, plan.tensor_ct)
    rest, wt = divmod(rest, plan.tensor_wt)
    n, ht = divmod(rest, plan.tensor_ht)
    return n, ht, wt, ct


def create_program_descriptor(plan: NhwcPlan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    N, H, W, C = plan.shape
    out_page_bytes = output_tensor.buffer_page_size()
    assert out_page_bytes == plan.in_page_bytes

    def cb(index, dtype, pages, page_bytes):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    cbs = [
        cb(CB_INPUT_TILES, input_tensor.dtype, plan.group_depth * GROUP_TILES, plan.in_page_bytes),
        cb(CB_ROW_BLOCK, ttnn.bfloat16, GROUP_TILES, plan.row_page_bytes),
        cb(CB_OUTPUT_TILES, output_tensor.dtype, plan.group_depth * GROUP_TILES, out_page_bytes),
    ]

    geometry_ct_args = [H, W, C, plan.tensor_ht, plan.tensor_wt, plan.tensor_ct]
    reader_ct_args = [
        CB_INPUT_TILES,
        *geometry_ct_args,
        plan.in_page_bytes,
        plan.read_chunk_tiles,
        *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
    ]
    writer_ct_args = [
        CB_OUTPUT_TILES,
        *geometry_ct_args,
        out_page_bytes,
        *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
    ]
    compute_ct_args = [CB_INPUT_TILES, CB_ROW_BLOCK, CB_OUTPUT_TILES]

    dataflow_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        dataflow_rt_args[core.x][core.y] = [count, *_start_coords(plan, start)]
        compute_rt_args[core.x][core.y] = [count]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_bfp8_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=dataflow_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_bfp8_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=dataflow_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    # Values only move: no arithmetic reaches the math engine, so fidelity and approximation are moot.
    # 16-bit half-sync DEST holds 8 tiles per transpose batch and is what fast tilize needs.
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_bfp8_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False, math_approx_mode=False),
    )
    kernels = [None, None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    kernels[COMPUTE_KERNEL_INDEX] = compute
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
