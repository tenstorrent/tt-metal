# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — MeshProgramDescriptor assembly (duplex multicast + local N-way tile sum).

One ``ttnn.ProgramDescriptor`` per participating device on the 1-D line, each with
a single worker core ``(0, 0)`` running reader (NCRISC) + compute (TRISC) + writer
(BRISC):

  * READER  phase 1 — stream this device's P input pages into ``cb_broadcast_pages``.
  * WRITER  phase 1 — duplex MULTICAST them into every peer's ``gathered_tensor``
                      slot ``my_id``; the LAST page of each direction is a fused
                      write + atomic-inc (``flush=True``), so one packet carries
                      both the payload and the peer's arrival signal.
  * READER  phase 2 — ``noc_semaphore_wait_min(sem, N-1)`` then reset (the WAITING
                      half and the cache-reuse re-arm are op-owned).
  * READER  phase 3 — interleave the N contributions to each output tile into
                      ``cb_shard_tiles`` (N contiguous pages per push): slot my_id
                      from ``input_tensor``, the rest from ``gathered_tensor``.
  * COMPUTE          — fold the N tiles into one DEST register (pairwise add).
  * WRITER  phase 2 — drain ``cb_output_tiles`` to output DRAM.

Everything is TILE layout, so a page IS a tile end to end — no tilize/untilize.

Route + framing come from the bound CCL host helpers (``ccl_dm_route`` owns the
fabric fwd/bwd SIGN REVERSAL; ``ccl_packet_dims`` owns the bf16 bit_floor). The
fabric-connection runtime-arg block is the layout ``FabricConnectionManager::
build_from_args`` consumes: ``[has_forward][fwd?][has_backward][bwd?]`` — and
unlike the unidirectional CCL ops BOTH flags may be 1 (interior devices).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_BROADCAST_PAGES = 0  # reader -> writer: the local shard, staged for fabric egress
_CB_SHARD_TILES = 1  # reader -> compute: N contributions to one output tile
_CB_OUTPUT_TILES = 16  # compute -> writer: the reduced tile

_LINK_IDX = 0  # single-link transfer

# Position of the writer inside ProgramDescriptor.kernels — the fabric connection
# RT args must be appended through a LIVE reference into that kernel's arg list
# (setup_fabric_connection also mutates the program), so the index is named rather
# than spelled inline.
_KERNEL_ORDER = ("reader", "compute", "writer")
_WRITER_KERNEL_IDX = _KERNEL_ORDER.index("writer")

# align(page_size, 1) == page_size, so the on-wire size can never round up past
# the destination page. Tile pages (2048 / 4096 B) are already 32-B aligned for
# the DRAM write. (op_design.md Risk 15; matches the only shipped duplex user.)
_FABRIC_ALIGNMENT = 1

# A pure-multicast route slot is 6 uint32: [start_distance_in_hops, range_hops,
# e, w, n, s]. The trailing four are 2-D-only fields, ignored on the 1-D
# LowLatency path. An ABSENT direction's six words are zeros and are never
# programmed — arm_* allocates a header only for CONNECTED directions and every
# issue gates on DuplexConn::has(dir).
_MCAST_ROUTE_WORDS = 6


def line_direction_slots(mesh_device, i, num_devices, topology):
    """Slot device ``i``'s (at most two) line neighbours into the fabric FORWARD /
    BACKWARD slots by the direction ``ccl_dm_route`` reports.

    Returns ``(fwd, bwd)``; each is ``None`` (no neighbour on that fabric side) or
    ``{"neighbor_id", "range_hops"}``.

    ``range_hops = k`` means "k CHIPS", and ``start_distance_in_hops = 1`` means
    "starting at the immediate neighbour" — so a multicast through the ``i+1``
    neighbour must cover ``N-1-i`` chips (devices i+1..N-1) and one through the
    ``i-1`` neighbour ``i`` chips (devices 0..i-1). Together they cover every peer
    EXACTLY once.

    ``is_forward`` is NOT "toward increasing index": ``ccl_dm_route`` deliberately
    owns a fwd/bwd sign reversal. Assuming otherwise would put the wrong multicast
    range on the wrong connection and silently reduce the wrong subset of devices
    (wrong values, no hang), so the direction is queried and the two neighbours are
    asserted to land in DIFFERENT slots.
    """
    coord_i = ttnn.MeshCoordinate(0, i)

    candidates = []
    if i + 1 < num_devices:
        candidates.append((i + 1, num_devices - 1 - i))
    if i >= 1:
        candidates.append((i - 1, i))

    slots = {True: None, False: None}
    for neighbor_index, range_hops in candidates:
        coord_n = ttnn.MeshCoordinate(0, neighbor_index)
        route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_n, topology)
        is_forward = bool(route.is_forward)
        if slots[is_forward] is not None:
            raise ValueError(
                f"all_reduce: device {i}'s two line neighbours both report "
                f"is_forward={is_forward}; cannot slot them onto distinct fabric directions"
            )
        slots[is_forward] = {"neighbor_id": route.neighbor_id, "range_hops": range_hops}

    return slots[True], slots[False]


def _mcast_route_ct_args(slot):
    """The 6-word pure-multicast route block for one direction slot."""
    if slot is None:
        return [0] * _MCAST_ROUTE_WORDS
    # start_distance_in_hops = 1 (the immediate neighbour), range_hops = #chips.
    return [1, slot["range_hops"], 0, 0, 0, 0]


def _append_fabric_rt_args(rt_args_ref, src_id, program, core, fwd_slot, bwd_slot):
    """Lay out ``[has_forward][fwd conn args][has_backward][bwd conn args]``.

    This is the block ``FabricConnectionManager::build_from_args`` reads, which the
    kernel's ``FabricDuplexSender`` consumes by reference (advancing past the whole
    block). Unlike the unidirectional CCL ops BOTH flags may be 1: an interior
    device on the line drives both fabric directions from this one core.
    ``setup_fabric_connection`` also mutates ``program`` (appends SemaphoreDescriptors).
    """
    rt_args_ref.append(1 if fwd_slot is not None else 0)
    if fwd_slot is not None:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, fwd_slot["neighbor_id"], _LINK_IDX, program, core))
    rt_args_ref.append(1 if bwd_slot is not None else 0)
    if bwd_slot is not None:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, bwd_slot["neighbor_id"], _LINK_IDX, program, core))


def _build_device_program(
    mesh_device,
    i,
    num_devices,
    input_tensor,
    gathered_tensor,
    output_tensor,
    sem_addr,
    fwd_slot,
    bwd_slot,
    worker_noc,
    page_size,
    pages_per_shard,
    data_format,
    compute_config,
    input_ta,
    gathered_ta,
    output_ta,
):
    """Build the ProgramDescriptor for device ``i`` on the line."""
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ----- circular buffers -----
    cb_broadcast_pages = ttnn.CBDescriptor(
        total_size=2 * page_size,  # streaming double-buffer: prefetch p+1 while p is in flight
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_BROADCAST_PAGES, data_format=data_format, page_size=page_size)
        ],
    )
    # The compute kernel indexes this CB at two offsets in ONE add_tiles call, so
    # the N contributions to an output tile must be N CONTIGUOUS pages. num_pages
    # is an integer multiple of N and EVERY push/pop is exactly N pages, so the
    # write pointer is always at page offset 0 or N => get_write_ptr() + k*page_size
    # never wraps. Any change here MUST keep num_pages a multiple of N.
    cb_shard_tiles = ttnn.CBDescriptor(
        total_size=2 * num_devices * page_size,  # one N-tile block + one prefetch block
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SHARD_TILES, data_format=data_format, page_size=page_size)
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=2 * page_size,  # streaming double-buffer
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_OUTPUT_TILES, data_format=data_format, page_size=page_size)
        ],
    )

    # ----- reader (NCRISC) -----
    reader_ct = [_CB_BROADCAST_PAGES, _CB_SHARD_TILES, i, num_devices] + input_ta + gathered_ta
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [
        input_tensor.buffer_address(),
        gathered_tensor.buffer_address(),
        pages_per_shard,
        page_size,
        sem_addr,
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- compute (TRISC) -----
    compute_ct = [_CB_SHARD_TILES, _CB_OUTPUT_TILES, num_devices, pages_per_shard]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_compute.cpp"),
        core_ranges=core_set,
        compile_time_args=compute_ct,
        runtime_args=[],
        config=compute_config,
    )

    # ----- writer (BRISC) -----
    writer_ct = (
        [_CB_BROADCAST_PAGES, _CB_OUTPUT_TILES, _FABRIC_ALIGNMENT, i, num_devices]
        + _mcast_route_ct_args(fwd_slot)
        + _mcast_route_ct_args(bwd_slot)
        + gathered_ta
        + output_ta
    )
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [
        gathered_tensor.buffer_address(),
        output_tensor.buffer_address(),
        pages_per_shard,
        page_size,
        sem_addr,
        worker_noc.x,  # peers' worker core NoC coords: the SAME logical core (0,0)
        worker_noc.y,  # on every chip, so one pair covers every peer
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader, compute, writer],  # order must match _KERNEL_ORDER
        semaphores=[],  # reserved for the SemaphoreDescriptors setup_fabric_connection appends
        cbs=[cb_broadcast_pages, cb_shard_tiles, cb_output_tiles],
    )

    # Fabric connection args live on the WRITER, appended through a live reference
    # so setup_fabric_connection can also mutate `program`.
    fabric_id_i = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, i))
    rt_ref = program.kernels[_WRITER_KERNEL_IDX].runtime_args[core.x][core.y]
    _append_fabric_rt_args(rt_ref, fabric_id_i, program, core, fwd_slot, bwd_slot)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    gathered_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    topology: ttnn.Topology,
    sem_addr: int,
    direction_slots: list,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    num_devices = len(direction_slots)

    data_format = input_tensor.dtype
    page_size = input_tensor.buffer_page_size()  # one 32x32 tile
    pages_per_shard = input_tensor.buffer_num_pages()  # tiles per shard (== P)

    # NoC coords of the worker core; the SAME logical core (0, 0) runs on every
    # chip, so its NoC coords are uniform across the mesh and one (x, y) pair
    # addresses every peer's receive semaphore.
    worker_noc = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))

    # fp32_dest_acc_en MUST track the dtype: with fp32 CBs and a 16-bit DEST every
    # accumulation step would be silently rounded to bf16 (up to N/2 chained
    # roundings). HiFi4 + fp32-dest-acc hits a Wormhole hardware bug, so fp32 uses
    # HiFi3. add_tiles pins LoFi internally, so math_fidelity is future-proofing.
    fp32_dest_acc_en = data_format == ttnn.float32
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3 if fp32_dest_acc_en else ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    input_ta = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    gathered_ta = list(ttnn.TensorAccessorArgs(gathered_tensor).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        fwd_slot, bwd_slot = direction_slots[i]
        program = _build_device_program(
            mesh_device,
            i,
            num_devices,
            input_tensor,
            gathered_tensor,
            output_tensor,
            sem_addr,
            fwd_slot,
            bwd_slot,
            worker_noc,
            page_size,
            pages_per_shard,
            data_format,
            compute_config,
            input_ta,
            gathered_ta,
            output_ta,
        )
        coord_i = ttnn.MeshCoordinate(0, i)
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program

    return mesh_pd
