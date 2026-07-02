# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — MeshProgramDescriptor assembly (bidirectional store-and-forward ring).

Builds one ``ProgramDescriptor`` per device on the 1-D line. Every device runs two
worker cores:

    forward core  = CoreCoord(0, 0), direction 0 -> neighbour i+1 (rightward)
    backward core = CoreCoord(0, 1), direction 1 -> neighbour i-1 (leftward)

Each worker core has a reader (NCRISC) and a writer (BRISC). There is no compute
kernel — this is pure data movement.

Distribution / addressing (gather_dim = 0, page-contiguous concat):
    pages_per_shard = input.buffer_num_pages()
    out_page(chip c, local page p) = c * pages_per_shard + p
The output is identical on every device, so a block's canonical page range is the
same address on every device; a fabric write of that range routed +1 hop lands in
the neighbour's identical range.

1-D routing (``ccl_dm_route``) owns the fwd/bwd sign reversal; the fabric-connection
RT-arg block is laid out exactly as ``append_ccl_fabric_rt_args``
(``[has_forward][fwd?][has_backward][bwd?]``), which the kernel-side
``FabricStreamSender`` consumes. Two op-internal ``GlobalSemaphore``s (barrier +
counting) are parked on the descriptor so the framework keeps their L1 alive across
program-cache hits.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB index (semantic). One instance per worker core (same index on both).
_CB_RELAY_PAGES = 16
# Constant streaming double-buffer depth (independent of shard size -> bounded L1).
_CB_RELAY_NUM_PAGES = 4

_LINK_IDX = 0  # single-link per direction

# Worker cores (uniform across every device on the line).
_FWD_CORE = ttnn.CoreCoord(0, 0)
_BWD_CORE = ttnn.CoreCoord(0, 1)


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as conn_arg_idx; its leading has_forward flag
    also equals the send direction, which the kernel peeks for is_forward.
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_devices: int,
    topology: ttnn.Topology,
    barrier_sem_addr: int,
    counting_sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype

    # ----- per-device page metrics (identical on every device) -----
    pages_per_shard = input_tensor.buffer_num_pages()
    page_size = input_tensor.buffer_page_size()
    aligned_page_size = output_tensor.buffer_aligned_page_size()

    # ----- virtual (NOC) coords of the two worker cores (uniform mesh-wide) -----
    fwd_vc = mesh_device.worker_core_from_logical_core(_FWD_CORE)
    bwd_vc = mesh_device.worker_core_from_logical_core(_BWD_CORE)

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(_FWD_CORE, _FWD_CORE), ttnn.CoreRange(_BWD_CORE, _BWD_CORE)])

    mesh_pd = ttnn.MeshProgramDescriptor()

    for i in range(num_devices):
        coord = ttnn.MeshCoordinate(0, i)
        num_fwd = num_devices - 1 - i
        num_bwd = i

        # cb_relay_pages: one program-local instance per worker core (same index).
        cb_relay = ttnn.CBDescriptor(
            total_size=_CB_RELAY_NUM_PAGES * aligned_page_size,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=_CB_RELAY_PAGES, data_format=data_format, page_size=aligned_page_size
                )
            ],
        )

        # ---- common scalar arg blocks ----
        def reader_ct(direction):
            ct = [
                direction,
                num_devices,
                i,
                pages_per_shard,
                page_size,
                aligned_page_size,
                num_fwd,
                num_bwd,
                _CB_RELAY_PAGES,
            ]
            ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
            ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
            return ct

        def writer_ct(direction):
            ct = [
                direction,
                num_devices,
                i,
                pages_per_shard,
                page_size,
                aligned_page_size,
                num_fwd,
                num_bwd,
                _CB_RELAY_PAGES,
                l1_alignment,
            ]
            ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
            return ct

        def reader_rt(core):
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                counting_sem_addr,
            ]
            return rt

        def writer_rt(core):
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                output_tensor.buffer_address(),
                barrier_sem_addr,
                counting_sem_addr,
                fwd_vc.x,
                fwd_vc.y,
                bwd_vc.x,
                bwd_vc.y,
            ]
            return rt

        # ---- kernels (order matters: writers are indices 1 and 3) ----
        fwd_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(_FWD_CORE, _FWD_CORE)]),
            compile_time_args=reader_ct(0),
            runtime_args=reader_rt(_FWD_CORE),
            config=ttnn.ReaderConfigDescriptor(),
        )
        fwd_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(_FWD_CORE, _FWD_CORE)]),
            compile_time_args=writer_ct(0),
            runtime_args=writer_rt(_FWD_CORE),
            config=ttnn.WriterConfigDescriptor(),
        )
        bwd_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(_BWD_CORE, _BWD_CORE)]),
            compile_time_args=reader_ct(1),
            runtime_args=reader_rt(_BWD_CORE),
            config=ttnn.ReaderConfigDescriptor(),
        )
        bwd_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(_BWD_CORE, _BWD_CORE)]),
            compile_time_args=writer_ct(1),
            runtime_args=writer_rt(_BWD_CORE),
            config=ttnn.WriterConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer],
            semaphores=[],
            cbs=[cb_relay],
        )

        src_fabric_id = mesh_device.get_fabric_node_id(coord)

        # Forward writer (index 1): fabric connection to i+1, only if it has targets.
        if num_fwd > 0:
            fwd_neighbor = ttnn.MeshCoordinate(0, i + 1)
            fwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord, fwd_neighbor, topology)
            fwd_writer_rt_ref = program.kernels[1].runtime_args[_FWD_CORE.x][_FWD_CORE.y]
            _append_fabric_rt_args(
                fwd_writer_rt_ref, src_fabric_id, fwd_route.neighbor_id, program, _FWD_CORE, fwd_route.is_forward
            )

        # Backward writer (index 3): fabric connection to i-1, only if it has targets.
        if num_bwd > 0:
            bwd_neighbor = ttnn.MeshCoordinate(0, i - 1)
            bwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord, bwd_neighbor, topology)
            bwd_writer_rt_ref = program.kernels[3].runtime_args[_BWD_CORE.x][_BWD_CORE.y]
            _append_fabric_rt_args(
                bwd_writer_rt_ref, src_fabric_id, bwd_route.neighbor_id, program, _BWD_CORE, bwd_route.is_forward
            )

        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program

    return mesh_pd
