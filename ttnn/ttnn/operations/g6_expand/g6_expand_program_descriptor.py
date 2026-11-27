# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program: `select_plan` (the single source of every knob) and the ProgramDescriptor.

Regime `column_group_broadcast` (op_design.md, Blocking Model). The cores form the rectangle
x in [0, num_row_groups), y in [0, num_col_groups). Column group y (a grid row) owns x pages
[y * core_src_col_tiles, ...); its sender (0, y) reads them, expands them in place and multicasts
them along the row. Core (x, y) writes out tile rows [x * core_out_row_tiles, ...) of those columns.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME = "column_group_broadcast"

# Model-wide L1 bound per core left to the op (BB-05).
L1_OP_BUDGET = 1347776

# Face geometry `expand_rows_block` is built for; validate() refuses anything else.
TILE_SHAPE = (32, 32)
TILE_FACE_SHAPE = (16, 16)

CB_EXPANDED_TILES = 0
MCAST_PREFIX = "group"
MCAST_FIRST_SEMAPHORE_ID = 0

SEED_ROWS_ALLOWED = (1, 2, 4, 8)


@dataclasses.dataclass(frozen=True)
class PlanKnobs:
    """Tunables of the plan. None means: derive from the device grid / L1 budget in select_plan."""

    max_row_groups: int | None = None  # cores per column group along grid x (lamp L1); None -> grid.x
    max_col_groups: int | None = None  # column groups along grid y; None -> grid.y
    src_block_depth: int | None = None  # cb_expanded_tiles depth in blocks; None -> 1, or 2 past the L1 clamp
    seed_rows: int = 2  # rows of each upper face seeded by RISC-V stores before the NoC fan-out (lamp L3)


DEFAULT_KNOBS = PlanKnobs()


def select_plan(input_tensor, out_shape, knobs: PlanKnobs = DEFAULT_KNOBS) -> dict:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in input_tensor.tile.face_shape)
    page_bytes = int(input_tensor.buffer_page_size())
    element_bytes = int(input_tensor.element_size())

    num_images = int(out_shape[0]) * int(out_shape[1])
    tensor_w_tiles = math.ceil(int(out_shape[3]) / tile_w)
    tensor_src_col_tiles = num_images * tensor_w_tiles
    tensor_out_row_tiles = math.ceil(int(out_shape[2]) / tile_h)

    max_col_groups = min(int(grid.y), knobs.max_col_groups or int(grid.y))
    max_row_groups = min(int(grid.x), knobs.max_row_groups or int(grid.x))

    core_src_col_tiles = math.ceil(tensor_src_col_tiles / min(tensor_src_col_tiles, max_col_groups))
    num_col_groups = math.ceil(tensor_src_col_tiles / core_src_col_tiles)
    core_out_row_tiles = math.ceil(tensor_out_row_tiles / min(tensor_out_row_tiles, max_row_groups))
    num_row_groups = math.ceil(tensor_out_row_tiles / core_out_row_tiles)

    if knobs.src_block_depth is not None:
        src_block_depth = knobs.src_block_depth
    else:
        src_block_depth = 1 if core_src_col_tiles * page_bytes <= L1_OP_BUDGET else 2
    block_src_col_tiles = min(core_src_col_tiles, L1_OP_BUDGET // (src_block_depth * page_bytes))
    block_out_row_tiles = core_out_row_tiles
    num_src_blocks_per_core = math.ceil(core_src_col_tiles / block_src_col_tiles)

    seed_rows = knobs.seed_rows
    assert seed_rows in SEED_ROWS_ALLOWED and face_h % seed_rows == 0, f"seed_rows {seed_rows} must divide {face_h}"
    assert src_block_depth >= 1 and block_src_col_tiles >= 1
    assert num_col_groups <= grid.y and num_row_groups <= grid.x

    return {
        "regime": REGIME,
        "grid": (int(grid.x), int(grid.y)),
        "page_bytes": page_bytes,
        "face_row_bytes": face_w * element_bytes,
        "face_bytes": face_h * face_w * element_bytes,
        "faces_per_band": tile_w // face_w,
        "face_height": face_h,
        "num_bands": tile_h // face_h,
        "tensor_w_tiles": tensor_w_tiles,
        "tensor_src_col_tiles": tensor_src_col_tiles,
        "tensor_out_row_tiles": tensor_out_row_tiles,
        "core_src_col_tiles": core_src_col_tiles,
        "num_col_groups": num_col_groups,
        "core_out_row_tiles": core_out_row_tiles,
        "num_row_groups": num_row_groups,
        "src_block_depth": src_block_depth,
        "block_src_col_tiles": block_src_col_tiles,
        "block_out_row_tiles": block_out_row_tiles,
        "num_src_blocks_per_core": num_src_blocks_per_core,
        "seed_rows": seed_rows,
        "num_cores": num_col_groups * num_row_groups,
        "l1_bytes_per_core": src_block_depth * block_src_col_tiles * page_bytes,
    }


def plan_report(plan: dict) -> dict:
    """BB-08 report: the plan's knobs and the role of every core (the BB-21 role dump)."""
    roles = {
        (x, y): ("sender" if x == 0 else "receiver")
        for y in range(plan["num_col_groups"])
        for x in range(plan["num_row_groups"])
    }
    keys = (
        "regime",
        "num_col_groups",
        "num_row_groups",
        "core_src_col_tiles",
        "core_out_row_tiles",
        "block_src_col_tiles",
        "block_out_row_tiles",
        "src_block_depth",
        "seed_rows",
        "num_cores",
        "l1_bytes_per_core",
    )
    return {**{k: plan[k] for k in keys}, "core_roles": roles}


def _core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _core_share(index, per_core, total):
    start = index * per_core
    return start, min(total, start + per_core) - start


def create_program_descriptor(input_tensor, output_tensor, plan):
    """Returns (descriptor, address_slots); address_slots lists (kernel index, tensor slot) of the
    common runtime arg 0 that carries a tensor's base address (0 = input, 1 = output)."""
    device = input_tensor.device()
    page_bytes = plan["page_bytes"]
    assert output_tensor.buffer_page_size() == page_bytes

    num_row_groups = plan["num_row_groups"]
    num_col_groups = plan["num_col_groups"]
    rectangle = _core_range(0, 0, num_row_groups - 1, num_col_groups - 1)
    sender_cores = _core_range(0, 0, 0, num_col_groups - 1)
    has_receivers = num_row_groups > 1

    # One CB over the whole rectangle, so its slots sit at the same L1 address on every core: the
    # sender multicasts from its slot into the same slot of each receiver.
    cb_expanded_tiles = ttnn.CBDescriptor(
        total_size=plan["src_block_depth"] * plan["block_src_col_tiles"] * page_bytes,
        core_ranges=rectangle,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXPANDED_TILES,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    reader_ct_args = [
        CB_EXPANDED_TILES,
        page_bytes,
        plan["block_src_col_tiles"],
        plan["face_row_bytes"],
        plan["face_bytes"],
        plan["faces_per_band"],
        plan["face_height"],
        plan["num_bands"],
        plan["seed_rows"],
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    src_total = plan["tensor_src_col_tiles"]
    row_total = plan["tensor_out_row_tiles"]
    core_src = plan["core_src_col_tiles"]
    core_rows = plan["core_out_row_tiles"]

    def reader_kernel(is_sender, x_range, core_ranges):
        rt_args = ttnn.RuntimeArgs()
        for y in range(num_col_groups):
            src_start, src_count = _core_share(y, core_src, src_total)
            for x in x_range:
                rt_args[x][y] = [src_start, src_count]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=[int(is_sender), *reader_ct_args],
            runtime_args=rt_args,
            common_runtime_args=[input_tensor.buffer_address()] if is_sender else [],
            config=ttnn.ReaderConfigDescriptor(),
        )

    sender_kernel = reader_kernel(True, range(0, 1), sender_cores)
    receiver_kernel = (
        reader_kernel(False, range(1, num_row_groups), _core_range(1, 0, num_row_groups - 1, num_col_groups - 1))
        if has_receivers
        else None
    )

    writer_ct_args = [
        CB_EXPANDED_TILES,
        page_bytes,
        plan["block_src_col_tiles"],
        plan["block_out_row_tiles"],
        plan["tensor_w_tiles"],
        plan["tensor_out_row_tiles"],
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    for y in range(num_col_groups):
        src_start, src_count = _core_share(y, core_src, src_total)
        for x in range(num_row_groups):
            row_start, row_count = _core_share(x, core_rows, row_total)
            writer_rt_args[x][y] = [src_start, src_count, row_start, row_count]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=rectangle,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_expanded_tiles])
    if has_receivers:
        # Receiver groups are consecutive runs of num_row_groups cores in row-major order, i.e. one
        # grid row each; sender_index 0 makes (0, y) the sender of row y.
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True, data_ready=ttnn.McastDataReady.Flag),
            rectangle,
            num_row_groups,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, [sender_kernel, receiver_kernel], MCAST_FIRST_SEMAPHORE_ID)
        descriptor.kernels = [sender_kernel, receiver_kernel, writer_kernel]
    else:
        ttnn.attach_absent_mcast(sender_kernel, MCAST_PREFIX)
        descriptor.kernels = [sender_kernel, writer_kernel]

    address_slots = [(0, 0), (len(descriptor.kernels) - 1, 1)]
    return descriptor, address_slots


# BB-07: one plan + descriptor per key; a hit only re-binds the two tensor addresses.
_ARTIFACTS = {}


def _artifact_key(input_tensor, out_shape, knobs):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        tuple(int(d) for d in input_tensor.padded_shape),
        tuple(int(d) for d in out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        int(device.id()),
        int(grid.x),
        int(grid.y),
        knobs,
    )


def get_program_artifacts(input_tensor, output_tensor, out_shape, knobs: PlanKnobs = DEFAULT_KNOBS):
    """(plan, descriptor, address_slots) for this call, built once per key."""
    key = _artifact_key(input_tensor, out_shape, knobs)
    artifacts = _ARTIFACTS.get(key)
    if artifacts is None:
        plan = select_plan(input_tensor, out_shape, knobs)
        descriptor, address_slots = create_program_descriptor(input_tensor, output_tensor, plan)
        artifacts = (plan, descriptor, address_slots)
        _ARTIFACTS[key] = artifacts
    return artifacts


def bind_tensor_addresses(descriptor, address_slots, input_tensor, output_tensor):
    """generic_op re-applies the descriptor's runtime args on every dispatch, so the cached
    descriptor must carry this call's buffer addresses."""
    addresses = (input_tensor.buffer_address(), output_tensor.buffer_address())
    kernels = descriptor.kernels
    for kernel_index, tensor_slot in address_slots:
        kernels[kernel_index].common_runtime_args[0] = addresses[tensor_slot]
