# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program: plan (select_plan) and ProgramDescriptor.

Regime column_group_broadcast (op_design.md, Blocking Model). The core rectangle is
x in [0, num_row_groups), y in [0, num_col_groups). Grid row y is a column group that owns x
pages [y * core_src_col_tiles, ...); grid column x owns out tile rows [x * core_out_row_tiles, ...)
of every image. The group's sender (0, y) reads and expands the pages and multicasts them along
its grid row; every core writes its own out rows of those columns.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME = "column_group_broadcast"

# BB-05: L1 the model leaves to the op on every core.
L1_OP_BUDGET = 1347776

# Rows of each upper face the sender's RISC-V seeds before the NoC fans them out (lamp L3).
SEED_ROWS = 2
SEED_ROWS_ALLOWED = (1, 2, 4, 8)

CB_EXPANDED_TILES = 0
MCAST_PREFIX = "expand_mcast"


def _div_up(a, b):
    return -(-a // b)


def select_plan(input_tensor, out_shape, *, max_row_groups=None, max_col_groups=None, seed_rows=SEED_ROWS):
    """The single source of every block knob, core assignment and buffer depth."""
    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in input_tensor.tile.face_shape)
    page_bytes = int(input_tensor.buffer_page_size())
    element_bytes = int(input_tensor.element_size())

    d0, d1, _, width = (int(v) for v in input_tensor.shape)
    h_out = int(out_shape[2])
    tensor_w_tiles = _div_up(width, tile_w)
    tensor_src_col_tiles = d0 * d1 * tensor_w_tiles
    tensor_out_row_tiles = _div_up(h_out, tile_h)

    grid = input_tensor.device().compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    max_col_groups = grid_y if max_col_groups is None else min(int(max_col_groups), grid_y)
    max_row_groups = grid_x if max_row_groups is None else min(int(max_row_groups), grid_x)

    S, Hc = tensor_src_col_tiles, tensor_out_row_tiles
    core_src_col_tiles = _div_up(S, min(S, max_col_groups))
    num_col_groups = _div_up(S, core_src_col_tiles)
    core_out_row_tiles = _div_up(Hc, min(Hc, max_row_groups))
    num_row_groups = _div_up(Hc, core_out_row_tiles)

    src_block_depth = 1 if core_src_col_tiles * page_bytes <= L1_OP_BUDGET else 2
    block_src_col_tiles = min(core_src_col_tiles, L1_OP_BUDGET // (src_block_depth * page_bytes))
    block_out_row_tiles = core_out_row_tiles

    assert seed_rows in SEED_ROWS_ALLOWED and face_h % seed_rows == 0, f"seed_rows={seed_rows}"
    assert block_src_col_tiles >= 1, "a page does not fit the op's L1 budget"

    cores = {}
    for y in range(num_col_groups):
        src_start = y * core_src_col_tiles
        src_count = min(S, src_start + core_src_col_tiles) - src_start
        for x in range(num_row_groups):
            row_start = x * core_out_row_tiles
            row_count = min(Hc, row_start + core_out_row_tiles) - row_start
            cores[(x, y)] = {
                "role": "sender" if x == 0 else "receiver",
                "src_start": src_start,
                "src_count": src_count,
                "row_start": row_start,
                "row_count": row_count,
            }

    return {
        "page_bytes": page_bytes,
        "face_row_bytes": face_w * element_bytes,
        "face_rows": face_h,
        "faces_per_band": tile_w // face_w,
        "bands_per_tile": tile_h // face_h,
        "tensor_w_tiles": tensor_w_tiles,
        "tensor_src_col_tiles": S,
        "tensor_out_row_tiles": Hc,
        "num_col_groups": num_col_groups,
        "num_row_groups": num_row_groups,
        "core_src_col_tiles": core_src_col_tiles,
        "core_out_row_tiles": core_out_row_tiles,
        "block_src_col_tiles": block_src_col_tiles,
        "block_out_row_tiles": block_out_row_tiles,
        "src_block_depth": src_block_depth,
        "seed_rows": seed_rows,
        "num_cores": num_col_groups * num_row_groups,
        "l1_bytes_per_core": src_block_depth * block_src_col_tiles * page_bytes,
        "cores": cores,
        "core_roles": {xy: c["role"] for xy, c in cores.items()},
    }


def _core_range_set(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _reader_kernel(core_ranges, is_sender, plan, x_ct_args, cores):
    runtime_args = ttnn.RuntimeArgs()
    for x, y in cores:
        c = plan["cores"][(x, y)]
        runtime_args[x][y] = [c["src_start"], c["src_count"]]
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=[
            1 if is_sender else 0,
            CB_EXPANDED_TILES,
            plan["page_bytes"],
            plan["block_src_col_tiles"],
            plan["seed_rows"],
            plan["face_row_bytes"],
            plan["face_rows"],
            plan["faces_per_band"],
            plan["bands_per_tile"],
            *x_ct_args,
        ],
        runtime_args=runtime_args,
        # Only the sender reads x; its address is the one common arg, patched per call.
        common_runtime_args=[0] if is_sender else [],
        config=ttnn.ReaderConfigDescriptor(),
    )


def create_program_descriptor(input_tensor, output_tensor, plan):
    """Build the descriptor once per cache key; bind_buffer_addresses() patches each call."""
    device = input_tensor.device()
    num_row_groups = plan["num_row_groups"]
    num_col_groups = plan["num_col_groups"]
    rectangle = _core_range_set(0, 0, num_row_groups - 1, num_col_groups - 1)
    senders = _core_range_set(0, 0, 0, num_col_groups - 1)
    sender_cores = [(0, y) for y in range(num_col_groups)]
    receiver_cores = [(x, y) for y in range(num_col_groups) for x in range(1, num_row_groups)]

    # One CB over the whole rectangle: the same slot address on every core is the multicast
    # landing contract.
    cb_expanded_tiles = ttnn.CBDescriptor(
        total_size=plan["src_block_depth"] * plan["block_src_col_tiles"] * plan["page_bytes"],
        core_ranges=rectangle,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXPANDED_TILES,
                data_format=input_tensor.dtype,
                page_size=plan["page_bytes"],
            )
        ],
    )

    x_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    readers = [_reader_kernel(senders, True, plan, x_ct_args, sender_cores)]
    if num_row_groups > 1:
        receivers = _core_range_set(1, 0, num_row_groups - 1, num_col_groups - 1)
        readers.append(_reader_kernel(receivers, False, plan, x_ct_args, receiver_cores))

    writer_args = ttnn.RuntimeArgs()
    for (x, y), c in plan["cores"].items():
        writer_args[x][y] = [c["src_start"], c["src_count"], c["row_start"], c["row_count"]]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=rectangle,
        compile_time_args=[
            CB_EXPANDED_TILES,
            plan["page_bytes"],
            plan["block_src_col_tiles"],
            plan["block_out_row_tiles"],
            plan["tensor_w_tiles"],
            plan["tensor_out_row_tiles"],
            *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
        ],
        runtime_args=writer_args,
        common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_expanded_tiles])
    if num_row_groups > 1:
        # Receiver groups are the grid rows of the rectangle (row-major, num_row_groups cores
        # each); the first core of each row, (0, y), is its fixed sender.
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True, data_ready=ttnn.McastDataReady.Flag),
            rectangle,
            num_row_groups,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, readers, 0)
    else:
        ttnn.attach_absent_mcast(readers[0], MCAST_PREFIX)

    descriptor.kernels = [*readers, writer]
    descriptor.custom_program_hash = ttnn.compute_program_descriptor_hash(descriptor)
    return {
        "descriptor": descriptor,
        "x_address_kernel": 0,
        "out_address_kernel": len(readers),
    }


def bind_buffer_addresses(program, input_tensor, output_tensor):
    """Patch the two per-call buffer addresses into the cached descriptor."""
    descriptor = program["descriptor"]
    kernels = descriptor.kernels
    kernels[program["x_address_kernel"]].common_runtime_args = [input_tensor.buffer_address()]
    kernels[program["out_address_kernel"]].common_runtime_args = [output_tensor.buffer_address()]
    descriptor.kernels = kernels
