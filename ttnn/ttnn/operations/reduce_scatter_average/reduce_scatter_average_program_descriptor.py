# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter_average — MeshProgramDescriptor assembly (ONE program per device,
ONE generic_op dispatch per invocation).

Per device i, one ProgramDescriptor over three fixed logical cores:

  * (0,0) forward relay:  reader (NCRISC) + fabric writer (BRISC) — line
    store-and-forward gather flowing rightward (fabric connection -> chip i+1).
  * (0,1) backward relay: mirror, flowing leftward (-> chip i-1).
  * (0,2) reduce:         reader + compute + writer — arrival-ordered incremental
    N-way sum of this device's S-tile slice, then a 1/N broadcast-scalar multiply,
    written to the dense output.

The overlap mechanism (op_design.md T4/T7): after each block's last fabric page the
sending relay writer issues TWO counting atomic-incs on the same connection — the
receiving device's relay core (store-and-forward chain) AND its reduce core (which
starts the block's accumulate pass while the next block is still in flight).

Seven kernel descriptors per program (4 relay + 3 reduce); programs are per-device
DISTINCT (my_chip_id, send/arrival counts, slice base). Fabric connection arg
blocks are appended to the relay writers' runtime args AFTER ProgramDescriptor
construction because build_ccl_fabric_rt_args mutates the program (appends
SemaphoreDescriptors); the block is placed FIRST so the kernel consumes it with a
cursor from 0.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; see op_design.md "Circular Buffers").
_CB_CONTRIBUTIONS = 0  # reduce reader -> compute: N contributions, g-granule stream
_CB_SCALER = 8  # reduce reader -> compute: persistent 1/N bcast-scalar tile
_CB_RELAY_PAGES = 16  # relay reader -> relay writer: double-buffered page stream
_CB_AVERAGED = 17  # compute -> reduce writer: averaged slice tiles, g-granules
_CB_ACCUMULATOR = 24  # compute-only resident running sum (capacity exactly S)

_LINK_IDX = 0  # single-link transfer
_TILE = 32

# Fixed logical cores (identical logical->physical mapping assumed across devices —
# reference precedent, hardware-validated).
_FWD_CORE = ttnn.CoreCoord(0, 0)
_BWD_CORE = ttnn.CoreCoord(0, 1)
_REDUCE_CORE = ttnn.CoreCoord(0, 2)

# Kernel list indices — needed to reach the live runtime_args views post-construction.
_K_FWD_WRITER = 1
_K_BWD_WRITER = 3


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _num_line_devices(mesh_device) -> int:
    n = 1
    for d in tuple(mesh_device.shape):
        n *= d
    return n


def _float_bits(x: float) -> int:
    """IEEE-754 single-precision bit pattern."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bf16_bits(x: float) -> int:
    """bfloat16 bit pattern, round-to-nearest-even (exact anyway for 1/N, N power of 2)."""
    b = _float_bits(x)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def _scaler_ct_args(dtype, num_devices: int) -> tuple[int, int]:
    """(scaler_bits, scaler_is_fp32) for the reduce reader's cb_scaler fill."""
    inv_n = 1.0 / num_devices
    if dtype == ttnn.float32:
        return _float_bits(inv_n), 1
    bits = _bf16_bits(inv_n)
    return (bits << 16) | bits, 0


def _granule(s: int) -> int:
    """Largest of {4, 2, 1} dividing S — so no tail chunk ever exists, and
    g <= DEST_AUTO_LIMIT = 4 under fp32_dest_acc_en + SyncHalf."""
    for g in (4, 2, 1):
        if s % g == 0:
            return g
    return 1  # unreachable (1 divides everything)


def _linear_flow(i: int, num_devices: int):
    """Per-direction (sends, arrivals) for device i on the LINE (op_design.md T1/T2).

    Forward carries left->right traffic: device i fwd-sends 1 + i blocks (own shard
    first, then relays of its i fwd arrivals) iff it has a right neighbour; fwd
    arrivals = i. Backward is the mirror. Line ends send 0 in the dead direction.
    """
    num_targets_fwd = num_devices - 1 - i  # devices reachable rightward
    num_targets_bwd = i  # devices reachable leftward
    fwd_sends = 1 + num_targets_bwd if num_targets_fwd > 0 else 0
    fwd_arrivals = num_targets_bwd
    bwd_sends = 1 + num_targets_fwd if num_targets_bwd > 0 else 0
    bwd_arrivals = num_targets_fwd
    return fwd_sends, fwd_arrivals, bwd_sends, bwd_arrivals


def _build_device_program(
    mesh_device,
    i,
    num_devices,
    topology,
    input_tensor,
    gather_buffer,
    output_tensor,
    sem_fwd_addr,
    sem_bwd_addr,
    quantities,
    accessors,
    noc_coords,
):
    """Build the single-dispatch ProgramDescriptor for device ``i``."""
    (
        page_size,
        aligned_page_size,
        l1_alignment,
        P,
        Wt,
        slice_Wt,
        slice_Ht,
        S,
        g,
        scaler_bits,
        scaler_is_fp32,
        dim,
    ) = quantities
    input_ta, gather_ta, output_ta = accessors
    fwd_noc, bwd_noc, reduce_noc = noc_coords

    fwd_sends, fwd_arrivals, bwd_sends, bwd_arrivals = _linear_flow(i, num_devices)

    fwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(_FWD_CORE, _FWD_CORE)])
    bwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(_BWD_CORE, _BWD_CORE)])
    relay_set = ttnn.CoreRangeSet([ttnn.CoreRange(_FWD_CORE, _BWD_CORE)])
    reduce_set = ttnn.CoreRangeSet([ttnn.CoreRange(_REDUCE_CORE, _REDUCE_CORE)])

    data_format = input_tensor.dtype

    # ----- circular buffers -----
    # Capacity rule: every CB's capacity is a multiple of its interaction quantum, so
    # a multi-page reserve/wait never straddles the ring wrap (op_design.md CB table).
    cb_relay_pages = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming page
        core_ranges=relay_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RELAY_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_contributions = ttnn.CBDescriptor(
        total_size=2 * g * page_size,  # double-buffered g-granules
        core_ranges=reduce_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_CONTRIBUTIONS, data_format=data_format, page_size=page_size)
        ],
    )
    cb_scaler = ttnn.CBDescriptor(
        total_size=page_size,  # persistent: pushed once, never popped
        core_ranges=reduce_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SCALER, data_format=data_format, page_size=page_size)
        ],
    )
    cb_accumulator = ttnn.CBDescriptor(
        total_size=S * page_size,  # resident running sum; capacity EXACTLY S (g divides S)
        core_ranges=reduce_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_ACCUMULATOR, data_format=data_format, page_size=page_size)
        ],
    )
    cb_averaged = ttnn.CBDescriptor(
        total_size=2 * g * page_size,  # double-buffered g-granules
        core_ranges=reduce_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_AVERAGED, data_format=data_format, page_size=page_size)
        ],
    )

    # ----- relay kernels (one source per role, direction via CT args) -----
    def relay_reader_ct(direction, num_sends, num_arrivals):
        return [_CB_RELAY_PAGES, direction, i, num_devices, num_sends, num_arrivals] + input_ta + gather_ta

    def relay_writer_ct(direction, num_sends):
        return [_CB_RELAY_PAGES, direction, i, num_devices, num_sends, l1_alignment] + gather_ta

    relay_reader_rt_vals = [
        input_tensor.buffer_address(),
        gather_buffer.buffer_address(),
        P,
        page_size,
    ]

    fwd_reader_rt = ttnn.RuntimeArgs()
    fwd_reader_rt[_FWD_CORE.x][_FWD_CORE.y] = relay_reader_rt_vals + [sem_fwd_addr]
    bwd_reader_rt = ttnn.RuntimeArgs()
    bwd_reader_rt[_BWD_CORE.x][_BWD_CORE.y] = relay_reader_rt_vals + [sem_bwd_addr]

    # Relay writers start with EMPTY rt args: the fabric connection block must come
    # FIRST but can only be built against the constructed program (it mutates it), so
    # both the block and the op args are appended post-construction. Idle direction:
    # rt args stay empty and the kernel no-ops under `if constexpr (num_sends > 0)`.
    fwd_writer_rt = ttnn.RuntimeArgs()
    fwd_writer_rt[_FWD_CORE.x][_FWD_CORE.y] = []
    bwd_writer_rt = ttnn.RuntimeArgs()
    bwd_writer_rt[_BWD_CORE.x][_BWD_CORE.y] = []

    fwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_relay_reader.cpp"),
        core_ranges=fwd_set,
        compile_time_args=relay_reader_ct(0, fwd_sends, fwd_arrivals),
        runtime_args=fwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    fwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_relay_writer.cpp"),
        core_ranges=fwd_set,
        compile_time_args=relay_writer_ct(0, fwd_sends),
        runtime_args=fwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    bwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_relay_reader.cpp"),
        core_ranges=bwd_set,
        compile_time_args=relay_reader_ct(1, bwd_sends, bwd_arrivals),
        runtime_args=bwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    bwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_relay_writer.cpp"),
        core_ranges=bwd_set,
        compile_time_args=relay_writer_ct(1, bwd_sends),
        runtime_args=bwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ----- reduce kernels (core (0,2)) -----
    reduce_reader_ct = (
        [
            _CB_CONTRIBUTIONS,
            _CB_SCALER,
            i,  # my_chip_id
            num_devices,
            fwd_arrivals,
            bwd_arrivals,
            S,
            g,
            Wt,
            slice_Wt,
            slice_Ht,
            P,
            dim,
            scaler_bits,
            scaler_is_fp32,
        ]
        + input_ta
        + gather_ta
    )
    reduce_reader_rt = ttnn.RuntimeArgs()
    reduce_reader_rt[_REDUCE_CORE.x][_REDUCE_CORE.y] = [
        input_tensor.buffer_address(),
        gather_buffer.buffer_address(),
        page_size,
        sem_fwd_addr,
        sem_bwd_addr,
    ]
    reduce_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_reduce_reader.cpp"),
        core_ranges=reduce_set,
        compile_time_args=reduce_reader_ct,
        runtime_args=reduce_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    reduce_compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_reduce_compute.cpp"),
        core_ranges=reduce_set,
        compile_time_args=[_CB_CONTRIBUTIONS, _CB_SCALER, _CB_ACCUMULATOR, _CB_AVERAGED, num_devices, S, g],
        runtime_args=[],
        # fp32 DEST accumulation covers both the bf16 sum-of-N rounding budget and the
        # float32 secondary dtype, and fixes DEST_AUTO_LIMIT = 4 (g <= 4 by construction).
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )

    reduce_writer_rt = ttnn.RuntimeArgs()
    reduce_writer_rt[_REDUCE_CORE.x][_REDUCE_CORE.y] = [output_tensor.buffer_address(), page_size]
    reduce_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_average_reduce_writer.cpp"),
        core_ranges=reduce_set,
        compile_time_args=[_CB_AVERAGED, S, g] + output_ta,
        runtime_args=reduce_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer, reduce_reader, reduce_compute, reduce_writer],
        semaphores=[],
        cbs=[cb_relay_pages, cb_contributions, cb_scaler, cb_accumulator, cb_averaged],
    )

    # ----- fabric connection blocks (appended AFTER construction — they mutate the
    # program) + the relay writers' op args, in the layout the kernel consumes:
    # [conn block][gather_addr, P, page_size, num_hops, sem_addr, relay_xy, reduce_xy].
    coord_i = ttnn.MeshCoordinate(0, i)
    fabric_id_i = mesh_device.get_fabric_node_id(coord_i)

    def _wire_direction(kernel_idx, core, neighbor_idx, sem_addr, relay_noc):
        coord_next = ttnn.MeshCoordinate(0, neighbor_idx)
        route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_next, topology)
        # Store-and-forward invariant: every hop is to the physical neighbour. The
        # route owns the fabric fwd/bwd sign reversal — never hand-derive is_forward.
        assert route.num_hops == 1, f"expected 1-hop neighbour route, got {route.num_hops}"
        ref = program.kernels[kernel_idx].runtime_args[core.x][core.y]
        block = ttnn._ttnn.fabric.build_ccl_fabric_rt_args(
            fabric_id_i, route.neighbor_id, _LINK_IDX, program, core, route.is_forward
        )
        ref.extend(block)
        ref.extend(
            [
                gather_buffer.buffer_address(),
                P,
                page_size,
                route.num_hops,
                sem_addr,
                relay_noc.x,  # neighbour's same-direction relay core (counting-sem target 1)
                relay_noc.y,
                reduce_noc.x,  # neighbour's reduce core (counting-sem target 2 — overlap)
                reduce_noc.y,
            ]
        )

    if fwd_sends > 0:
        _wire_direction(_K_FWD_WRITER, _FWD_CORE, i + 1, sem_fwd_addr, fwd_noc)
    if bwd_sends > 0:
        _wire_direction(_K_BWD_WRITER, _BWD_CORE, i - 1, sem_bwd_addr, bwd_noc)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    gather_buffer: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_devices: int,
    dim: int,
    topology,
    sem_fwd_addr: int,
    sem_bwd_addr: int,
) -> ttnn.MeshProgramDescriptor:
    """One ProgramDescriptor per mesh coordinate; ``dim`` is the CANONICAL (positive)
    scatter dim (3 or 2, gated upstream by SUPPORTED)."""
    mesh_device = input_tensor.device()
    assert _num_line_devices(mesh_device) == num_devices

    l1_alignment = ttnn.get_l1_alignment()
    page_size = input_tensor.buffer_page_size()
    aligned_page_size = _round_up(page_size, l1_alignment)
    P = input_tensor.buffer_num_pages()  # tiles per shard

    shape = list(input_tensor.shape)
    Wt = shape[3] // _TILE
    Ht = shape[2] // _TILE
    # Dim-aware slice quantities: the reduce reader uses slice_Wt for the dim=3 walk
    # and slice_Ht for the dim=2 walk; the OTHER one is kernel-unused (validate()
    # guarantees only shape[dim] is N*TILE-divisible, so the unused floor-division
    # may legitimately be 0).
    slice_Wt = Wt // num_devices
    slice_Ht = Ht // num_devices
    S = P // num_devices  # output tiles per device (dim-independent: P is a product)
    g = _granule(S)
    scaler_bits, scaler_is_fp32 = _scaler_ct_args(input_tensor.dtype, num_devices)

    quantities = (
        page_size,
        aligned_page_size,
        l1_alignment,
        P,
        Wt,
        slice_Wt,
        slice_Ht,
        S,
        g,
        scaler_bits,
        scaler_is_fp32,
        dim,
    )

    accessors = (
        list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        list(ttnn.TensorAccessorArgs(gather_buffer).get_compile_time_args()),
        list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()),
    )

    # NoC coords of the three worker cores (uniform across the mesh): the counting
    # incs target the SAME logical cores on the neighbour device.
    noc_coords = (
        mesh_device.worker_core_from_logical_core(_FWD_CORE),
        mesh_device.worker_core_from_logical_core(_BWD_CORE),
        mesh_device.worker_core_from_logical_core(_REDUCE_CORE),
    )

    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        program = _build_device_program(
            mesh_device,
            i,
            num_devices,
            topology,
            input_tensor,
            gather_buffer,
            output_tensor,
            sem_fwd_addr,
            sem_bwd_addr,
            quantities,
            accessors,
            noc_coords,
        )
        coord_i = ttnn.MeshCoordinate(0, i)
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program

    return mesh_pd
