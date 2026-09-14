# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_x_first bake-off: isolated rms_norm reader start-up ordering.

Builds a two-kernel program (reader variant + consumer/dump writer, no compute) on EXACTLY the
op's core geometry (derive_blocking is imported read-only from the op), with the op's CB slots and
sizes, and returns the dumped x / gamma / scaler so the host can gate each variant byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from ttnn.operations.rms_norm.rms_norm_program_descriptor import (
    CB_GAMMA_STICKS,
    CB_GAMMA_TILES,
    CB_SCALER,
    CB_X_STICKS,
    CB_X_TILES,
    DEPTH_X,
    DEPTH_X_STICKS_BLOCKS,
    GAMMA_MODE_NONE,
    GAMMA_MODE_RM,
    GAMMA_MODE_TILE,
    T_SCALER,
    TILE,
    derive_blocking,
)

KERNEL_DIR = Path(__file__).parent / "kernels"

ORDER_NAMES = {
    0: "baseline",
    1: "xfirst_one_barrier",
    2: "xfirst_split_barriers",
    3: "xfirst_trid_barriers",
    4: "scaler_x_gamma",
    5: "scaler_xg_one_barrier",
    6: "scaler_xg_trid",
    7: "x_gamma_scaler_last",
    8: "x_scaler_gamma",
}


def _cores(roles) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(r.x, r.y), ttnn.CoreCoord(r.x, r.y)) for r in roles])


def _cb(index, core_ranges, page_size, num_pages, dtype):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def build_program(x, gamma, out_x, out_gamma, out_scaler, order: int):
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    l1_free = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    blocking = derive_blocking(x, gamma, grid.x, grid.y, l1_free, fp32_dest_acc_en=False)
    assert not blocking.regime.startswith("R3"), "interleaved bench only"

    input_rm = x.layout == ttnn.ROW_MAJOR_LAYOUT
    if gamma is None:
        gamma_mode = GAMMA_MODE_NONE
    elif gamma.layout == ttnn.ROW_MAJOR_LAYOUT:
        gamma_mode = GAMMA_MODE_RM
    else:
        gamma_mode = GAMMA_MODE_TILE

    t_in = ttnn.tile_size(x.dtype)
    e_in = x.element_size() if input_rm else 0
    t_g = ttnn.tile_size(gamma.dtype) if gamma is not None else 0
    e_g = gamma.element_size() if gamma_mode == GAMMA_MODE_RM else 0
    B = blocking.block_rows
    Wt = blocking.tensor_w_tiles

    active = [r for r in blocking.all_roles if r.is_active]
    active_cores = _cores(active)

    cbs = [_cb(CB_SCALER, active_cores, T_SCALER, 1, ttnn.bfloat16)]
    w_groups = {}
    for r in active:
        w_groups.setdefault(r.core_w_tiles, []).append(r)
    for wc, wroles in sorted(w_groups.items(), reverse=True):
        cr = _cores(wroles)
        if input_rm:
            cbs.append(_cb(CB_X_STICKS, cr, t_in, DEPTH_X_STICKS_BLOCKS * B * wc, x.dtype))
        else:
            cbs.append(_cb(CB_X_TILES, cr, t_in, DEPTH_X * B * wc, x.dtype))
        if gamma_mode == GAMMA_MODE_TILE:
            cbs.append(_cb(CB_GAMMA_TILES, cr, t_g, wc, gamma.dtype))
        elif gamma_mode == GAMMA_MODE_RM:
            cbs.append(_cb(CB_GAMMA_STICKS, cr, t_g, wc, gamma.dtype))

    named_common = [
        ("CB_X_TILES", CB_X_TILES),
        ("CB_X_STICKS", CB_X_STICKS),
        ("CB_SCALER", CB_SCALER),
        ("CB_GAMMA_TILES", CB_GAMMA_TILES),
        ("CB_GAMMA_STICKS", CB_GAMMA_STICKS),
        ("INPUT_RM", 1 if input_rm else 0),
        ("GAMMA_MODE", gamma_mode),
        ("IN_PAGE_BYTES", x.buffer_page_size()),
        ("IN_TILE_BYTES", t_in),
        ("IN_ELEM_BYTES", e_in),
        ("GAMMA_TILE_BYTES", t_g),
        ("GAMMA_ELEM_BYTES", e_g),
    ]

    def blocks_of(role):
        num_blocks = -(-role.core_row_tiles // B)
        return num_blocks, role.core_row_tiles - (num_blocks - 1) * B

    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    reader_ct += list(
        (ttnn.TensorAccessorArgs(gamma) if gamma is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for idx, r in enumerate(active):
        num_blocks, last_rows = blocks_of(r)
        reader_rt[r.x][r.y] = [
            x.buffer_address(),
            gamma.buffer_address() if gamma is not None else 0,
            r.row_tile_start,
            num_blocks,
            B,
            last_rows,
            r.w_tile_start,
            r.core_w_tiles,
            Wt,
            blocking.tensor_row_tiles,
        ]
        writer_rt[r.x][r.y] = [
            out_x.buffer_address(),
            out_gamma.buffer_address() if out_gamma is not None else 0,
            out_scaler.buffer_address(),
            r.row_tile_start,
            num_blocks,
            B,
            last_rows,
            r.w_tile_start,
            r.core_w_tiles,
            Wt,
            idx,
        ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_bench.cpp"),
        core_ranges=active_cores,
        compile_time_args=reader_ct,
        named_compile_time_args=named_common
        + [("GAMMA_PAGE_BYTES", gamma.buffer_page_size() if gamma is not None else 0), ("READER_ORDER", order)],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_ct = list(ttnn.TensorAccessorArgs(out_x).get_compile_time_args())
    writer_ct += list(
        (
            ttnn.TensorAccessorArgs(out_gamma) if out_gamma is not None else ttnn.TensorAccessorArgs()
        ).get_compile_time_args()
    )
    writer_ct += list(ttnn.TensorAccessorArgs(out_scaler).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_bench.cpp"),
        core_ranges=active_cores,
        compile_time_args=writer_ct,
        named_compile_time_args=named_common
        + [
            ("OUT_GAMMA_PAGE_BYTES", out_gamma.buffer_page_size() if out_gamma is not None else 0),
            ("SCALER_TILE_BYTES", T_SCALER),
        ],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=cbs)
    return program, blocking, active


def make_inputs(device, shape, x_layout, gamma_layout, seed=0):
    """x (shape) and gamma (1,1,1,W) bf16 on DRAM interleaved; None gamma_layout -> no gamma."""
    import torch

    torch.manual_seed(seed)
    W = shape[-1]
    x_t = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, layout=x_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gamma = None
    g_t = None
    if gamma_layout is not None:
        g_t = torch.randn((1, 1, 1, W), dtype=torch.bfloat16)
        gamma = ttnn.from_torch(g_t, layout=gamma_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x, x_t, gamma, g_t


def run_variant(device, x, gamma, order: int, num_active: int):
    """Runs one reader ordering; returns (out_x, out_gamma, out_scaler) torch tensors (sentinel-filled
    before the run so an unwritten page is visible)."""
    import torch

    shape = list(x.shape)
    W = shape[-1]
    sentinel = -777.0
    out_x = ttnn.from_torch(
        torch.full(shape, sentinel, dtype=torch.bfloat16),
        layout=x.layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_gamma = None
    if gamma is not None:
        if gamma.layout == ttnn.ROW_MAJOR_LAYOUT:
            g_shape = (1, 1, TILE, W)  # the whole 32-row stick block the reader publishes
        else:
            g_shape = (1, 1, 1, W)
        out_gamma = ttnn.from_torch(
            torch.full(g_shape, sentinel, dtype=torch.bfloat16),
            layout=gamma.layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    out_scaler = ttnn.from_torch(
        torch.full((1, 1, TILE * num_active, TILE), sentinel, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    program, blocking, active = build_program(x, gamma, out_x, out_gamma, out_scaler, order)
    assert len(active) == num_active
    io = (
        [x]
        + ([gamma] if gamma is not None else [])
        + [out_x]
        + ([out_gamma] if out_gamma is not None else [])
        + [out_scaler]
    )
    ttnn.generic_op(io, program)
    ox = ttnn.to_torch(out_x)
    og = ttnn.to_torch(out_gamma) if out_gamma is not None else None
    osc = ttnn.to_torch(out_scaler)
    return ox, og, osc


def num_active_cores(device, x, gamma):
    grid = device.compute_with_storage_grid_size()
    l1_free = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    blocking = derive_blocking(x, gamma, grid.x, grid.y, l1_free, fp32_dest_acc_en=False)
    active = [r for r in blocking.all_roles if r.is_active]
    return len(active), blocking
