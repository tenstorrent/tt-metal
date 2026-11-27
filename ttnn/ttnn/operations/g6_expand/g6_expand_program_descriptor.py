# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program descriptor: one CB, an injector reader, a receiver reader, one writer.

Role placement (BB-17): group g sits in grid column g // b, rows [(g mod b)*rs, (g mod b)*rs + rs).
The injector reader runs on the top core of every strip, the receiver reader on the rest of the
strip, the writer on every used core. Each role covers at most one rectangle per strip band.

Program cache (BB-07): a built descriptor is kept per call signature. On a hit only the two buffer
addresses change; they are common runtime args and are patched in place, so the device-side
structural hash (sources, CT args, core ranges, CBs, semaphores) is unchanged.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_ROW_BCAST_TILES = 0  # x pages, row 0 replicated into all rows in place; written out from here
MCAST_PREFIX = "bcast"

_DESCRIPTOR_CACHE = {}


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def _merge_vertical(rects):
    """Merge (x0, y0, x1, y1) rectangles that share an x-extent and touch vertically."""
    merged = []
    for r in sorted(rects, key=lambda r: (r[0], r[2], r[1])):
        if merged and merged[-1][0] == r[0] and merged[-1][2] == r[2] and merged[-1][3] + 1 == r[1]:
            last = merged.pop()
            r = (last[0], last[1], r[2], r[3])
        merged.append(r)
    return merged


def _crs(rects):
    return ttnn.CoreRangeSet([_rect(*r) for r in _merge_vertical(rects)])


def _role_rects(plan):
    """Per-role rectangles (x0, y0, x1, y1) derived from (b, rs, G)."""
    b, rs, G = plan.strips_per_column, plan.row_splits, plan.num_groups
    full_cols, partial = divmod(G, b)
    injector, receiver, used = [], [], []
    for strip in range(b):
        cols = full_cols + (1 if strip < partial else 0)
        if cols == 0:
            continue
        top = strip * rs
        injector.append((0, top, cols - 1, top))
        if rs > 1:
            receiver.append((0, top + 1, cols - 1, top + rs - 1))
        used.append((0, top, cols - 1, top + rs - 1))
    return injector, receiver, used


def _runtime_args(plan, role, width):
    rt = ttnn.RuntimeArgs()
    for (x, y), work in plan.core_work.items():
        if role is None or plan.core_roles[(x, y)] == role:
            rt[x][y] = list(work[:width])
    return rt


def _build(input_tensor, output_tensor, plan):
    device = input_tensor.device()
    page_bytes = plan.page_bytes
    injector_rects, receiver_rects, used_rects = _role_rects(plan)
    used_cores = _crs(used_rects)

    cb_row_bcast_tiles = ttnn.CBDescriptor(
        total_size=plan.t_block_depth * plan.block_pages * page_bytes,
        core_ranges=used_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ROW_BCAST_TILES,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    x_accessor_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    def reader(is_injector, rects):
        role = "injector" if is_injector else "receiver"
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=_crs(rects),
            compile_time_args=[
                CB_ROW_BCAST_TILES,
                page_bytes,
                plan.block_pages,
                int(is_injector),
                plan.replicate_seed_rows,
                *x_accessor_ct,
            ],
            # (group_q_start, group_pages)
            runtime_args=_runtime_args(plan, role, 2),
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )

    injector_reader = reader(True, injector_rects)
    readers = [injector_reader]
    if receiver_rects:
        readers.append(reader(False, receiver_rects))

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=used_cores,
        compile_time_args=[
            CB_ROW_BCAST_TILES,
            page_bytes,
            plan.block_pages,
            plan.tensor_cols_tiles,
            plan.tensor_rows_tiles,
            *out_accessor_ct,
        ],
        # (group_q_start, group_pages, core_r_start, core_rows_tiles)
        runtime_args=_runtime_args(plan, None, 4),
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_row_bcast_tiles])
    if plan.row_splits > 1:
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(handshake=True),
            used_cores,
            plan.row_splits,
            ttnn.McastFixedSenderConfig(),
            ttnn.McastCoreOrder.ColumnMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, readers, 0)
    else:
        ttnn.attach_absent_mcast(injector_reader, MCAST_PREFIX)
    kernels = [*readers, writer]
    descriptor.kernels = kernels
    return descriptor, readers, writer, kernels


def _cache_key(input_tensor, output_tensor, plan):
    device = input_tensor.device()
    return (
        device.id() if hasattr(device, "id") else id(device),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        tuple(output_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        repr(input_tensor.memory_config()),
        plan.grid,
        plan.knobs(),
    )


def get_program_descriptor(input_tensor, output_tensor, plan):
    key = _cache_key(input_tensor, output_tensor, plan)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        entry = _build(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = entry
        return entry[0]
    descriptor, readers, writer, kernels = entry
    for kernel in readers:
        kernel.common_runtime_args = [input_tensor.buffer_address()]
    writer.common_runtime_args = [output_tensor.buffer_address()]
    descriptor.kernels = kernels
    return descriptor
