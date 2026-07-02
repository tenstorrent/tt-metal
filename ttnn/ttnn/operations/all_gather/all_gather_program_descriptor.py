# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — MeshProgramDescriptor assembly (ring store-and-forward on a line).

Builds one ``ttnn.ProgramDescriptor`` per participating device on the 1-D line
and parks them in a ``ttnn.MeshProgramDescriptor``. Every device runs the same
two-worker-core ring role:

  * forward core  (0, 0): flow rightward, fabric connection -> chip i+1
  * backward core (0, 1): flow leftward,  fabric connection -> chip i-1

Each core runs a reader (NCRISC) + writer (BRISC). The op is pure byte movement
(no tilize/untilize, no compute) and is format-agnostic: it copies physical
pages (``buffer_page_size`` bytes, ``buffer_num_pages`` pages) verbatim.

Primary path: ``gather_dim=0`` (page-contiguous concat) — device c's shard maps
to output pages ``[c*P, (c+1)*P)`` with ``P = input.buffer_num_pages()``, so the
neighbour's destination is the local output page index routed one fabric hop.

Cross-device sync uses ONE op-internal ``GlobalSemaphore`` (parked on the
descriptor): per ``(device, core)`` it is the store-and-forward counting
semaphore. The upstream writer atomic-increments it once per landed block (the
SENDING half, owned by the CCL helper); the local reader waits on the cumulative
count (the WAITING half, a plain ``noc_semaphore_wait_min`` the op owns) before
reading each block back, and resets it at the end (cache-reuse re-arm).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

# Topology lives on the C++ module; reference it directly to stay safe at
# eager-import time (the top-level ttnn.Topology alias binds later).
from ttnn._ttnn.operations.ccl import Topology as _Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_RELAY_PAGES = 16  # reader -> writer: seed pages + store-and-forward read-backs
_CB_SELF_COPY = 24  # forward reader scratch for the local self-copy (input -> own output block)

_LINK_IDX = 0  # single-link transfer


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _gather_page_params(input_tensor, gd_neg: int):
    """Whole-page concat-by-gather_dim parameters (Refinement 2).

    Returns ``(dim_j, inner_stride)`` in PAGE units for the reader/writer remap
    ``out_page(c,p) = high*(N*dim_j*inner) + (c*dim_j+mid)*inner + low``. The page
    grid is layout-specific:

      * TILE : [B, C, Ht, Wt]   (page == one 32x32 tile)
      * RM   : [B, C, H]        (page == one W-row; W is INSIDE the page)

    ``dim_j`` = the gathered axis's size in that grid; ``inner_stride`` = product
    of grid dims inner to it. For gather_dim=0 this yields (B_pages, P/B_pages),
    i.e. the page-contiguous ``c*P+p``. validate() has already refused the two
    structural gaps (RM+gd=-1 sub-page write; TILE+gd=-2 non-tile-aligned H), so
    this only ever runs for whole-page-copyable cells.
    """
    shape = list(input_tensor.shape)
    while len(shape) < 4:
        shape = [1] + shape
    B, C, H, W = shape[-4], shape[-3], shape[-2], shape[-1]
    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    if is_tile:
        grid = [B, C, _round_up(H, 32) // 32, _round_up(W, 32) // 32]
    else:
        grid = [B, C, H]  # page == W-row
    page_axis = 4 + gd_neg  # 0=B 1=C 2=H 3=W
    dim_j = grid[page_axis]
    inner = 1
    for a in range(page_axis + 1, len(grid)):
        inner *= grid[a]
    return dim_j, inner


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as conn_arg_idx; its leading has_forward
    flag also equals the send direction, which the kernel peeks for is_forward.
    ``setup_fabric_connection`` also mutates ``program`` (appends SemaphoreDescriptors).
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def _build_device_program(
    mesh_device,
    i,
    num_devices,
    input_tensor,
    output_tensor,
    sem_addr,
    topology,
    fwd_noc,
    bwd_noc,
    page_size,
    aligned_page_size,
    pages_per_shard,
    l1_alignment,
    data_format,
    input_ta,
    output_ta,
    dim_j,
    inner_stride,
):
    """Build the ProgramDescriptor for device ``i`` on the line."""
    num_targets_fwd = num_devices - 1 - i  # devices reachable rightward (i+1..N-1)
    num_targets_bwd = i  # devices reachable leftward (0..i-1)

    coord_i = ttnn.MeshCoordinate(0, i)
    fabric_id_i = mesh_device.get_fabric_node_id(coord_i)

    forward_core = ttnn.CoreCoord(0, 0)
    backward_core = ttnn.CoreCoord(0, 1)
    fwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, forward_core)])
    bwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(backward_core, backward_core)])
    both_set = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, backward_core)])

    # ----- circular buffers (on both worker cores) -----
    cb_relay = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming chunk
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RELAY_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_self = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # forward reader's self-copy scratch
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SELF_COPY, data_format=data_format, page_size=aligned_page_size)
        ],
    )

    # ----- readers (NCRISC) -----
    def reader_ct(direction):
        return (
            [
                _CB_RELAY_PAGES,
                direction,
                i,
                num_devices,
                num_targets_fwd,
                num_targets_bwd,
                _CB_SELF_COPY,
            ]
            + input_ta
            + output_ta
        )

    reader_rt_vals = [
        input_tensor.buffer_address(),
        output_tensor.buffer_address(),
        pages_per_shard,
        page_size,
        sem_addr,
        dim_j,  # gathered-axis page size (concat addressing)
        inner_stride,  # pages inner to the gathered axis
    ]

    fwd_reader_rt = ttnn.RuntimeArgs()
    fwd_reader_rt[forward_core.x][forward_core.y] = list(reader_rt_vals)
    bwd_reader_rt = ttnn.RuntimeArgs()
    bwd_reader_rt[backward_core.x][backward_core.y] = list(reader_rt_vals)

    fwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
        core_ranges=fwd_set,
        compile_time_args=reader_ct(0),
        runtime_args=fwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    bwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
        core_ranges=bwd_set,
        compile_time_args=reader_ct(1),
        runtime_args=bwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- writers (BRISC) -----
    def writer_ct(direction):
        return [
            _CB_RELAY_PAGES,
            direction,
            i,
            num_devices,
            num_targets_fwd,
            num_targets_bwd,
            l1_alignment,
        ] + output_ta

    # Routes (ccl_dm_route owns the fwd/bwd sign reversal + ring short-way).
    fwd_writer_rt = ttnn.RuntimeArgs()
    bwd_writer_rt = ttnn.RuntimeArgs()
    fwd_route = None
    bwd_route = None
    if num_targets_fwd > 0:
        coord_next = ttnn.MeshCoordinate(0, i + 1)
        fwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_next, topology)
        fwd_writer_rt[forward_core.x][forward_core.y] = [
            output_tensor.buffer_address(),
            pages_per_shard,
            page_size,
            fwd_route.num_hops,
            sem_addr,
            fwd_noc.x,  # neighbour forward core noc x (counting-sem target)
            fwd_noc.y,
            dim_j,  # gathered-axis page size (concat addressing)
            inner_stride,  # pages inner to the gathered axis
        ]
    else:
        fwd_writer_rt[forward_core.x][forward_core.y] = []  # early-return writer reads nothing

    if num_targets_bwd > 0:
        coord_prev = ttnn.MeshCoordinate(0, i - 1)
        bwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_prev, topology)
        bwd_writer_rt[backward_core.x][backward_core.y] = [
            output_tensor.buffer_address(),
            pages_per_shard,
            page_size,
            bwd_route.num_hops,
            sem_addr,
            bwd_noc.x,  # neighbour backward core noc x (counting-sem target)
            bwd_noc.y,
            dim_j,  # gathered-axis page size (concat addressing)
            inner_stride,  # pages inner to the gathered axis
        ]
    else:
        bwd_writer_rt[backward_core.x][backward_core.y] = []

    fwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
        core_ranges=fwd_set,
        compile_time_args=writer_ct(0),
        runtime_args=fwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    bwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
        core_ranges=bwd_set,
        compile_time_args=writer_ct(1),
        runtime_args=bwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer],
        semaphores=[],
        cbs=[cb_relay, cb_self],
    )

    # Fabric connection args live on the writers (kernel idx 1 = fwd, idx 3 = bwd).
    if num_targets_fwd > 0:
        fabric_id_next = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, i + 1))
        ref = program.kernels[1].runtime_args[forward_core.x][forward_core.y]
        _append_fabric_rt_args(ref, fabric_id_i, fabric_id_next, program, forward_core, fwd_route.is_forward)
    if num_targets_bwd > 0:
        fabric_id_prev = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, i - 1))
        ref = program.kernels[3].runtime_args[backward_core.x][backward_core.y]
        _append_fabric_rt_args(ref, fabric_id_i, fabric_id_prev, program, backward_core, bwd_route.is_forward)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    topology: ttnn.Topology,
    sem_addr: int,
    gather_dim: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    num_devices = _num_line_devices(mesh_device)

    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype
    page_size = input_tensor.buffer_page_size()
    pages_per_shard = input_tensor.buffer_num_pages()
    aligned_page_size = _round_up(page_size, l1_alignment)

    # Concat-by-gather_dim page-remap parameters (same for every device: they
    # depend only on the shard shape + layout + gather_dim). gather_dim is the
    # canonical NEGATIVE index (-4..-1) from validate().
    dim_j, inner_stride = _gather_page_params(input_tensor, gather_dim)

    # NoC coords of the two worker cores (uniform across the mesh) — the
    # counting-sem targets the SAME logical core on the neighbour device.
    forward_core = ttnn.CoreCoord(0, 0)
    backward_core = ttnn.CoreCoord(0, 1)
    fwd_noc = mesh_device.worker_core_from_logical_core(forward_core)
    bwd_noc = mesh_device.worker_core_from_logical_core(backward_core)

    input_ta = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        coord_i = ttnn.MeshCoordinate(0, i)
        program = _build_device_program(
            mesh_device,
            i,
            num_devices,
            input_tensor,
            output_tensor,
            sem_addr,
            topology,
            fwd_noc,
            bwd_noc,
            page_size,
            aligned_page_size,
            pages_per_shard,
            l1_alignment,
            data_format,
            input_ta,
            output_ta,
            dim_j,
            inner_stride,
        )
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program

    return mesh_pd


def _num_line_devices(mesh_device) -> int:
    """Number of devices on the 1-D line (mesh view is (1, N))."""
    shape = tuple(mesh_device.shape)
    n = 1
    for d in shape:
        n *= d
    return n
