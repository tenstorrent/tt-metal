# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program: `select_plan` (the single source of every block knob) and the ProgramDescriptor.

Grid rectangle x in [0, num_row_groups), y in [0, num_col_groups). Row y is one column group: it owns
x pages [y * core_src_col_tiles, ...). Core (0, y) is the group's sender (reads + expands + multicasts),
cores (1.., y) are receivers; every core writes out tile rows [x * core_out_row_tiles, ...) of the
group's columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME = "column_group_broadcast"

# Per-core L1 left to the op by the model (BB-05).
L1_OP_BUDGET = 1347776

CB_EXPANDED_TILES = 0

MCAST_PREFIX = "group_mcast"

# Rows of each upper face the RISC-V copies from row 0 before NoC self-reads fan them out.
DEFAULT_SEED_ROWS = 2


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Plan:
    regime: str
    page_bytes: int
    tensor_w_tiles: int
    tensor_src_col_tiles: int
    tensor_out_row_tiles: int
    core_src_col_tiles: int
    num_col_groups: int
    core_out_row_tiles: int
    num_row_groups: int
    src_block_depth: int
    block_src_col_tiles: int
    block_out_row_tiles: int
    seed_rows: int
    tile_h: int
    tile_w: int
    face_h: int
    face_w: int

    @property
    def num_cores(self):
        return self.num_col_groups * self.num_row_groups

    @property
    def cb_pages(self):
        return self.src_block_depth * self.block_src_col_tiles

    @property
    def l1_bytes_per_core(self):
        return self.cb_pages * self.page_bytes

    def src_range(self, y):
        start = y * self.core_src_col_tiles
        return start, min(self.tensor_src_col_tiles, start + self.core_src_col_tiles) - start

    def row_range(self, x):
        start = x * self.core_out_row_tiles
        return start, min(self.tensor_out_row_tiles, start + self.core_out_row_tiles) - start

    def core_roles(self):
        return {
            (x, y): ("sender" if x == 0 else "receiver")
            for y in range(self.num_col_groups)
            for x in range(self.num_row_groups)
        }


def select_plan(input_tensor, out_shape, *, seed_rows=DEFAULT_SEED_ROWS) -> Plan:
    shape = list(input_tensor.shape)
    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in input_tensor.tile.face_shape)
    page_bytes = input_tensor.buffer_page_size()

    tensor_w_tiles = _div_up(shape[3], tile_w)
    tensor_src_col_tiles = shape[0] * shape[1] * tensor_w_tiles
    tensor_out_row_tiles = _div_up(out_shape[2], tile_h)

    grid = input_tensor.device().compute_with_storage_grid_size()
    max_col_groups, max_row_groups = int(grid.y), int(grid.x)

    core_src_col_tiles = _div_up(tensor_src_col_tiles, min(tensor_src_col_tiles, max_col_groups))
    num_col_groups = _div_up(tensor_src_col_tiles, core_src_col_tiles)
    core_out_row_tiles = _div_up(tensor_out_row_tiles, min(tensor_out_row_tiles, max_row_groups))
    num_row_groups = _div_up(tensor_out_row_tiles, core_out_row_tiles)

    src_block_depth = 1 if core_src_col_tiles * page_bytes <= L1_OP_BUDGET else 2
    block_src_col_tiles = min(core_src_col_tiles, L1_OP_BUDGET // (src_block_depth * page_bytes))
    block_out_row_tiles = core_out_row_tiles

    plan = Plan(
        regime=REGIME,
        page_bytes=page_bytes,
        tensor_w_tiles=tensor_w_tiles,
        tensor_src_col_tiles=tensor_src_col_tiles,
        tensor_out_row_tiles=tensor_out_row_tiles,
        core_src_col_tiles=core_src_col_tiles,
        num_col_groups=num_col_groups,
        core_out_row_tiles=core_out_row_tiles,
        num_row_groups=num_row_groups,
        src_block_depth=src_block_depth,
        block_src_col_tiles=block_src_col_tiles,
        block_out_row_tiles=block_out_row_tiles,
        seed_rows=seed_rows,
        tile_h=tile_h,
        tile_w=tile_w,
        face_h=face_h,
        face_w=face_w,
    )

    assert seed_rows in (1, 2, 4, 8) and face_h % seed_rows == 0, f"seed_rows={seed_rows} must divide {face_h}"
    assert num_col_groups <= max_col_groups and num_row_groups <= max_row_groups, "rectangle off the grid"
    assert block_src_col_tiles >= 1, "a single page exceeds the op's L1 budget"
    assert plan.l1_bytes_per_core <= L1_OP_BUDGET
    return plan


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


class ProgramArtifacts:
    """Built once per cache key; `bind` patches the two buffer addresses for a call."""

    def __init__(self, plan, descriptor, input_address_kernels, output_address_kernels):
        self.plan = plan
        self._descriptor = descriptor
        self._input_address_kernels = input_address_kernels
        self._output_address_kernels = output_address_kernels
        self._kernels = [*input_address_kernels, *output_address_kernels]

    def bind(self, input_tensor, output_tensor):
        for kernel in self._input_address_kernels:
            kernel.common_runtime_args = [input_tensor.buffer_address()]
        for kernel in self._output_address_kernels:
            kernel.common_runtime_args = [output_tensor.buffer_address()]
        self._descriptor.kernels = self._kernels
        return self._descriptor


def create_program_artifacts(input_tensor, output_tensor) -> ProgramArtifacts:
    plan = select_plan(input_tensor, list(output_tensor.shape))
    device = input_tensor.device()
    page_bytes = plan.page_bytes
    if input_tensor.buffer_aligned_page_size() != page_bytes:
        raise ValueError("g6_expand: x pages must be aligned to the DRAM alignment")

    ncg, nrg = plan.num_col_groups, plan.num_row_groups
    rectangle = _rect(0, 0, nrg - 1, ncg - 1)
    senders = _rect(0, 0, 0, ncg - 1)

    cb_expanded_tiles = ttnn.CBDescriptor(
        total_size=plan.cb_pages * page_bytes,
        core_ranges=rectangle,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXPANDED_TILES,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    element_bytes = page_bytes // (plan.tile_h * plan.tile_w)
    face_row_bytes = plan.face_w * element_bytes
    reader_ct_args = [
        CB_EXPANDED_TILES,
        page_bytes,
        plan.block_src_col_tiles,
        0,  # is_sender, set per role below
        plan.seed_rows,
        plan.face_h,
        face_row_bytes,
        plan.face_h * face_row_bytes,
        plan.tile_w // plan.face_w,
        plan.tile_h // plan.face_h,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    IS_SENDER_CT_INDEX = 3

    def reader_kernel(core_ranges, is_sender, cores):
        ct_args = list(reader_ct_args)
        ct_args[IS_SENDER_CT_INDEX] = int(is_sender)
        rt_args = ttnn.RuntimeArgs()
        for x, y in cores:
            rt_args[x][y] = list(plan.src_range(y))
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=ct_args,
            runtime_args=rt_args,
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )

    sender_reader = reader_kernel(senders, True, [(0, y) for y in range(ncg)])
    readers = [sender_reader]

    writer_ct_args = [
        CB_EXPANDED_TILES,
        page_bytes,
        plan.block_src_col_tiles,
        plan.block_out_row_tiles,
        plan.tensor_w_tiles,
        plan.tensor_out_row_tiles,
    ]
    if output_tensor.buffer_page_size() != page_bytes:
        raise ValueError("g6_expand: out and x pages must have the same size")
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    for y in range(ncg):
        for x in range(nrg):
            writer_rt_args[x][y] = [*plan.src_range(y), *plan.row_range(x)]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=rectangle,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_expanded_tiles])
    if nrg > 1:
        receivers = _rect(1, 0, nrg - 1, ncg - 1)
        receiver_reader = reader_kernel(receivers, False, [(x, y) for y in range(ncg) for x in range(1, nrg)])
        readers.append(receiver_reader)
        group_mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True, data_ready=ttnn.McastDataReady.Flag),
            rectangle,
            nrg,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        group_mcast.attach(descriptor, MCAST_PREFIX, readers, 0)
    else:
        ttnn.attach_absent_mcast(sender_reader, MCAST_PREFIX)

    descriptor.kernels = [*readers, writer]
    return ProgramArtifacts(plan, descriptor, input_address_kernels=readers, output_address_kernels=[writer])
