# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — MeshProgramDescriptor assembly (two per-coordinate programs).

Exactly TWO ``(MeshCoordinateRange, ProgramDescriptor)`` entries are installed:
one covering only ``sender_coord``, one covering only ``receiver_coord``. Every
other mesh coordinate receives an empty ``ProgramDescriptor`` from the generic-op
factory, i.e. runs no program — intermediate hops are pure fabric routing and no
Tensix on a relay chip participates.

Both participating devices use logical worker core ``(0, 0)``, which is what lets
``get_noc_addr(sem_addr)`` computed on either core name "the same semaphore, on
the chip the packet is routed to" (the GlobalSemaphore is a mesh-wide L1
allocation, so its absolute address is identical on every device).

Sender device, core (0,0)                Receiver device, core (0,0)
  NCRISC  input DRAM -> cb_shard_pages     NCRISC  ack, wait, intermediate
  BRISC   handshake, framing, fabric               read-back, de-frame
                                           BRISC   cb_output_pages -> output DRAM
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers"). The two devices run
# different programs, so the two sets never collide.
_CB_SHARD_PAGES = 0  # sender: reader (NCRISC) -> writer (BRISC), streaming
_CB_OUTPUT_PAGES = 16  # receiver: reader (NCRISC) -> writer (BRISC), streaming
_CB_PACKET_STAGING = 24  # sender: writer-owned packet scratch (reserve-once)
_CB_PACKET_LANDING = 24  # receiver: reader-owned packet scratch (reserve-once)

_LINK_IDX = 0  # single-link transfer (multi-link needs the mux connection policy)

_WORKER_CORE = ttnn.CoreCoord(0, 0)


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args (C++-only, unbound).

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as ``conn_arg_idx``; its leading
    ``has_forward`` flag doubles as the send direction, which the kernel peeks for
    ``is_forward`` before handing the cursor to ``FabricStreamSender``.

    ``setup_fabric_connection`` also MUTATES ``program`` (it appends two
    SemaphoreDescriptors), so it must be handed the same object we return.
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def _cb(index: int, num_pages: int, page_size: int, core_grid) -> ttnn.CBDescriptor:
    """One opaque-byte CB. Every CB in this op declares ``uint32``: no CB here is
    ever consumed by a compute thread, so the format is inert for pure byte
    movement, and a uint32 format keeps bfloat8_b payloads (whose CB format would
    otherwise demand tile-shaped pages) on the same code path."""
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=ttnn.uint32,
                page_size=page_size,
            )
        ],
    )


def _build_send_program(
    mesh_device,
    input_tensor,
    intermediate_tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr,
    geom,
):
    """Sender device program: NCRISC reader (kernel 0) + BRISC writer (kernel 1)."""
    core = _WORKER_CORE
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb_shard_pages = _cb(_CB_SHARD_PAGES, 2, geom["aligned_page_size"], core_grid)
    cb_packet_staging = _cb(_CB_PACKET_STAGING, 1, geom["packet_size"], core_grid)

    # ----- reader (NCRISC): input DRAM -> cb_shard_pages -----
    reader_ct = [_CB_SHARD_PAGES] + geom["input_ta"]
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [
        input_tensor.buffer_address(),
        geom["num_pages"],
        geom["page_size"],
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- writer (BRISC): handshake, framing, fabric egress -----
    # ccl_dm_route owns the fwd/bwd SIGN REVERSAL and the ring short-way choice;
    # pass .is_forward straight through and use .neighbor_id (the NEXT HOP, not
    # the destination) as the fabric connection's dst node.
    route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)

    writer_ct = [
        _CB_SHARD_PAGES,
        _CB_PACKET_STAGING,
        geom["l1_alignment"],
        geom["page_segments"],
    ] + geom["intermediate_ta"]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [
        intermediate_tensor.buffer_address(),
        geom["num_pages"],
        geom["total_packets"],
        geom["page_size"],
        geom["packet_size"],
        geom["pages_per_packet"],
        sem_addr,
        route.num_hops,
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer],
        semaphores=[],
        cbs=[cb_shard_pages, cb_packet_staging],
    )

    # Fabric block LAST, on the fabric-owning kernel (the writer, index 1). This
    # must run after the ProgramDescriptor exists and be handed that same object.
    fabric_id = mesh_device.get_fabric_node_id(sender_coord)
    ref = program.kernels[1].runtime_args[core.x][core.y]
    _append_fabric_rt_args(ref, fabric_id, route.neighbor_id, program, core, route.is_forward)

    return program


def _build_receive_program(
    mesh_device,
    intermediate_tensor,
    output_tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr,
    geom,
):
    """Receiver device program: NCRISC reader (kernel 0) + BRISC writer (kernel 1)."""
    core = _WORKER_CORE
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb_output_pages = _cb(_CB_OUTPUT_PAGES, 2, geom["aligned_page_size"], core_grid)
    cb_packet_landing = _cb(_CB_PACKET_LANDING, 1, geom["packet_size"], core_grid)

    # Route back toward the SENDER — used for the receiver's one-shot "ready" ack.
    route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)

    # ----- reader (NCRISC): ack, wait, local read-back, de-frame -----
    reader_ct = [
        _CB_PACKET_LANDING,
        _CB_OUTPUT_PAGES,
        geom["l1_alignment"],
        geom["page_segments"],
    ] + geom["intermediate_ta"]
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [
        intermediate_tensor.buffer_address(),
        geom["num_pages"],
        geom["total_packets"],
        geom["page_size"],
        geom["packet_size"],
        geom["pages_per_packet"],
        sem_addr,
        route.num_hops,
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- writer (BRISC): cb_output_pages -> output DRAM -----
    writer_ct = [_CB_OUTPUT_PAGES] + geom["output_ta"]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [
        output_tensor.buffer_address(),
        geom["num_pages"],
        geom["page_size"],
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer],
        semaphores=[],
        cbs=[cb_output_pages, cb_packet_landing],
    )

    # Fabric block LAST, on the fabric-owning kernel (the reader, index 0).
    fabric_id = mesh_device.get_fabric_node_id(receiver_coord)
    ref = program.kernels[0].runtime_args[core.x][core.y]
    _append_fabric_rt_args(ref, fabric_id, route.neighbor_id, program, core, route.is_forward)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    intermediate_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology,
    sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()

    l1_alignment = ttnn.get_l1_alignment()
    page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()

    # ccl_packet_dims owns the bfloat16 bit_floor rule and both packing regimes
    # (A: coalesce several pages per packet; B: segment one page across packets).
    dims = ttnn._ttnn.fabric.ccl_packet_dims(input_tensor.dtype, page_size, num_pages, l1_alignment)

    geom = {
        "l1_alignment": l1_alignment,
        "page_size": page_size,
        "num_pages": num_pages,
        "aligned_page_size": _round_up(page_size, l1_alignment),
        "packet_size": dims.packet_size_bytes,
        "pages_per_packet": dims.pages_per_packet,
        "page_segments": dims.page_segments,
        "total_packets": dims.total_packets,
        # MANDATORY: the kernels build every TensorAccessor with the 2-argument
        # constructor, so the per-bank stride comes from these CT args
        # (buffer.aligned_page_size()) — never from a runtime page-size override.
        "input_ta": list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        "intermediate_ta": list(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args()),
        "output_ta": list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()),
    }

    send_program = _build_send_program(
        mesh_device,
        input_tensor,
        intermediate_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
        geom,
    )
    receive_program = _build_receive_program(
        mesh_device,
        intermediate_tensor,
        output_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
        geom,
    )

    # __setitem__ is APPEND-ONLY — insert each coordinate exactly once.
    mesh_pd = ttnn.MeshProgramDescriptor()
    mesh_pd[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = send_program
    mesh_pd[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receive_program
    return mesh_pd
