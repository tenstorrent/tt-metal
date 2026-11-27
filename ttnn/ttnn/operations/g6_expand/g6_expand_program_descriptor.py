# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand ProgramDescriptor: one CB, two reader roles (sender / receiver), one writer, one Mcast channel.

Core rectangle x in [0, num_row_groups), y in [0, num_col_groups). Row y is column group y: it owns x
pages [y * core_src_col_tiles, ...). Core (x, y) writes out tile rows [x * core_out_row_tiles, ...)
of those columns. The sender of each group is (0, y).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Expanded tiles: x pages land here on the sender, are expanded in place, and are multicast to the
# same slot address on every receiver of the group.
CB_EXPANDED_TILES = 0

MCAST_PREFIX = "x_mcast"


def _range_set(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


class G6ExpandProgram:
    """Cached program artifacts; bind() patches only the two buffer addresses (common runtime args)."""

    def __init__(self, kernels, reader_kernels, writer_kernel, cbs, semaphores, plan, mcast):
        self.kernels = kernels
        self.reader_kernels = reader_kernels
        self.writer_kernel = writer_kernel
        self.cbs = cbs
        self.semaphores = semaphores
        self.plan = plan
        self._mcast = mcast

    def bind(self, input_addr, output_addr):
        for kernel in self.reader_kernels:
            kernel.common_runtime_args = [input_addr]
        self.writer_kernel.common_runtime_args = [output_addr]
        return ttnn.ProgramDescriptor(kernels=self.kernels, semaphores=self.semaphores, cbs=self.cbs)


def _core_src_range(plan, y):
    start = y * plan["core_src_col_tiles"]
    count = min(plan["tensor_src_col_tiles"], start + plan["core_src_col_tiles"]) - start
    return start, count


def _core_row_range(plan, x):
    start = x * plan["core_out_row_tiles"]
    count = min(plan["tensor_out_row_tiles"], start + plan["core_out_row_tiles"]) - start
    return start, count


def create_program_descriptor(input_tensor, output_tensor, plan):
    device = input_tensor.device()
    num_row_groups = plan["num_row_groups"]
    num_col_groups = plan["num_col_groups"]
    page_bytes = plan["page_bytes"]
    block_src_col_tiles = plan["block_src_col_tiles"]

    all_cores = _range_set(0, 0, num_row_groups - 1, num_col_groups - 1)
    sender_cores = _range_set(0, 0, 0, num_col_groups - 1)
    has_receivers = num_row_groups > 1

    cb_expanded_tiles = ttnn.CBDescriptor(
        total_size=plan["src_block_depth"] * block_src_col_tiles * page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXPANDED_TILES,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    input_accessor_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()

    def reader_ct(is_sender):
        return [
            CB_EXPANDED_TILES,
            int(is_sender),
            block_src_col_tiles,
            page_bytes,
            plan["face_row_bytes"],
            plan["face_bytes"],
            plan["face_rows"],
            plan["faces_per_tile_row"],
            plan["face_rows_per_tile"],
            plan["seed_rows"],
            *input_accessor_ct,
        ]

    def reader_kernel(core_ranges, is_sender, xs):
        rt = ttnn.RuntimeArgs()
        for y in range(num_col_groups):
            src_start, src_count = _core_src_range(plan, y)
            for x in xs:
                rt[x][y] = [src_start, src_count]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct(is_sender),
            runtime_args=rt,
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )

    sender_reader = reader_kernel(sender_cores, True, [0])
    reader_kernels = [sender_reader]
    if has_receivers:
        receiver_cores = _range_set(1, 0, num_row_groups - 1, num_col_groups - 1)
        reader_kernels.append(reader_kernel(receiver_cores, False, range(1, num_row_groups)))

    writer_rt = ttnn.RuntimeArgs()
    for y in range(num_col_groups):
        src_start, src_count = _core_src_range(plan, y)
        for x in range(num_row_groups):
            row_start, row_count = _core_row_range(plan, x)
            writer_rt[x][y] = [src_start, src_count, row_start, row_count]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=[
            CB_EXPANDED_TILES,
            block_src_col_tiles,
            plan["block_out_row_tiles"],
            page_bytes,
            plan["tensor_w_tiles"],
            plan["tensor_out_row_tiles"],
            *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
        ],
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_expanded_tiles])
    mcast = None
    if has_receivers:
        # One 1-D multicast per grid row: row-major groups of num_row_groups cores, sender = first core (x = 0).
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True, data_ready=ttnn.McastDataReady.Flag),
            all_cores,
            num_row_groups,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, reader_kernels, 0)
    else:
        ttnn.attach_absent_mcast(sender_reader, MCAST_PREFIX)

    kernels = [*reader_kernels, writer_kernel]
    return G6ExpandProgram(
        kernels=kernels,
        reader_kernels=reader_kernels,
        writer_kernel=writer_kernel,
        cbs=[cb_expanded_tiles],
        semaphores=list(descriptor.semaphores),
        plan=plan,
        mcast=mcast,
    )
