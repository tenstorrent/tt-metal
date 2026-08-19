# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Copies ONE mesh device's interleaved shard of a mesh-sharded tensor to ANOTHER
mesh device across the TT-Fabric, leaving every other device's shard untouched::

    output_shard[receiver_coord][i] = input_shard[sender_coord][i]
    output_shard[c]                 = output_shard_on_entry[c]     for c != receiver_coord

It performs NO arithmetic — the oracle is identity, and there is NO compute
kernel: the op runs only dataflow kernels (NCRISC reader + BRISC writer) on
logical core ``(0, 0)`` of each of the two participating devices.

Newly authored dataflow kernels under ``kernels/`` are assembled by a
``ttnn.generic_op`` over a ``ttnn.MeshProgramDescriptor`` holding exactly two
``(MeshCoordinateRange, ProgramDescriptor)`` entries (sender + receiver); every
other mesh coordinate receives an empty descriptor from the generic-op factory
and runs no program. This op does NOT wrap, import, call, or dispatch to
``ttnn.point_to_point`` / ``ttnn._ttnn.operations.point_to_point``.

Cross-device coordination uses ONE cached op-internal ``GlobalSemaphore``:

  1. receiver fabric-incs the sender ("ready"),
  2. sender waits, re-arms, streams ``total_packets`` fabric writes into the
     receiver's intermediate DRAM, fabric-incs the receiver ("done"),
  3. receiver waits, reads the packets back locally, de-frames them into the
     output shard, and re-arms.
"""

from __future__ import annotations

import ttnn

# Topology lives on the C++ module; the top-level ttnn.Topology alias only binds
# AFTER ttnn.operations is auto-imported, so reference the source module directly.
from ttnn._ttnn.operations.ccl import Topology as _Topology

try:  # registry-model refusal types; fall back when the shared module is absent.
    from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
except ImportError:  # pragma: no cover

    class UnsupportedAxisValue(NotImplementedError):
        pass

    class ExcludedCell(NotImplementedError):
        pass


from .point_to_point_program_descriptor import create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# point_to_point is a pure byte copy: nothing in the data path inspects the
# element type or the tile grid. The physical pages (padded tiles for TILE,
# last-dim rows for ROW_MAJOR) are moved verbatim, so a non-tile-aligned shard
# transfers exactly like an aligned one — `alignment` is tagged (and supported)
# rather than gated.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = tuple(inputs[0])
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
# Every dtype x layout x topology x alignment cell is reachable through a single
# code path: the CBs carry opaque bytes (declared uint32), packet framing goes
# through the bound `ccl_packet_dims` host helper (which owns the only
# dtype-dependent rule — the bfloat16 bit_floor on the channel buffer size), and
# every TensorAccessor is built with the 2-argument constructor so the per-bank
# stride is the buffer's own aligned page size.
SUPPORTED = {
    "dtype": [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.uint16,
        ttnn.int32,
        ttnn.uint32,
    ],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "topology": [_Topology.Linear, _Topology.Ring],
    "alignment": ["tile_aligned", "non_tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: created ONCE per mesh_device (+ exactly one
# synchronize_device, right after creation), reused across program-cache hits,
# never recreated and never followed by a per-call barrier.
_SEMAPHORE_CACHE: dict = {}


def _get_or_create_semaphore(mesh_device):
    key = id(mesh_device)
    sem = _SEMAPHORE_CACHE.get(key)
    if sem is None:
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        worker_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        ttnn.synchronize_device(mesh_device)
        _SEMAPHORE_CACHE[key] = sem
    return sem


def _packet_dims(input_tensor):
    """Packet framing for this shard — delegated to the bound host helper.

    `ccl_packet_dims` owns the bfloat16 `std::bit_floor` special case on the
    fabric channel buffer size and both packing regimes (coalesce / segment);
    never reimplement it here.
    """
    l1_align = ttnn.get_l1_alignment()
    return ttnn._ttnn.fabric.ccl_packet_dims(
        input_tensor.dtype,
        input_tensor.buffer_page_size(),
        input_tensor.buffer_num_pages(),
        l1_align,
    )


def _resolve_intermediate_spec(input_tensor):
    """The staging tensor is a raw-byte packet buffer, decoupled from the payload
    dtype/layout so one code path serves every cell: one row == one packet."""
    dims = _packet_dims(input_tensor)
    # packet_size_bytes is always a multiple of 16 (see op_design.md "Packet
    # framing"), so the //4 is exact and buffer_page_size() == packet_size_bytes.
    shape = [dims.total_packets, dims.packet_size_bytes // 4]
    return shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG


def _coord_in_mesh(coord, mesh_shape) -> bool:
    comps = tuple(coord)
    if len(comps) != len(mesh_shape):
        return False
    return all(0 <= c < dim for c, dim in zip(comps, mesh_shape))


def _same_row_or_column(sender_coord, receiver_coord) -> bool:
    """ccl_dm_route routes along a row OR a column; anything else it rejects."""
    s, r = tuple(sender_coord), tuple(receiver_coord)
    return s[0] == r[0] or s[1] == r[1]


def validate(input_tensor, sender_coord, receiver_coord, *, topology, output_tensor, intermediate_tensor):
    """Runtime gate.

    Structural misuse raises ValueError; axis refusals raise the registry-model
    UnsupportedAxisValue / ExcludedCell. Checks 1-8 run BEFORE the axis gate so
    that structural misuse is never mistaken for an unsupported-feature refusal.
    """
    # 1. MeshDevice.
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("point_to_point: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)

    # 2. No self-send.
    if tuple(sender_coord) == tuple(receiver_coord):
        raise ValueError(
            f"point_to_point: cannot send to self (sender_coord == receiver_coord == {tuple(sender_coord)})"
        )

    # 3. Both coordinates inside the mesh, and on a shared row or column.
    for name, coord in (("sender_coord", sender_coord), ("receiver_coord", receiver_coord)):
        if not _coord_in_mesh(coord, mesh_shape):
            raise ValueError(f"point_to_point: {name}={tuple(coord)} is outside the mesh (shape {mesh_shape})")
    if not _same_row_or_column(sender_coord, receiver_coord):
        raise ValueError(
            f"point_to_point: sender_coord={tuple(sender_coord)} and receiver_coord={tuple(receiver_coord)} "
            "share neither a mesh row nor a mesh column (1-D fabric routing is impossible)"
        )

    # 4. Interleaved only.
    if input_tensor.is_sharded():
        raise ValueError("point_to_point: sharded input not yet supported (interleaved only)")

    # 5. Rank.
    if len(input_tensor.shape) < 2:
        raise ValueError(f"point_to_point: input rank must be >= 2, got {len(input_tensor.shape)}")

    # 6. Page-size alignment. Load-bearing: the fabric write puts
    # align(packet_size, l1_alignment) bytes on the wire, and the intra-packet
    # page stride is round_up(page_size, l1_alignment). A 16-byte-aligned page
    # makes both round-ups no-ops for the payload pages themselves.
    l1_align = ttnn.get_l1_alignment()
    page = input_tensor.buffer_page_size()
    if page % l1_align != 0:
        raise ValueError(f"point_to_point: per-shard page size ({page} B) must be a multiple of {l1_align} B")

    # 7. output_tensor spec == input spec.
    if output_tensor is not None:
        if (
            list(output_tensor.shape) != list(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config() != input_tensor.memory_config()
        ):
            raise ValueError("point_to_point: output_tensor spec must equal the input tensor's spec")

    # 8. intermediate_tensor spec == the resolved staging spec.
    if intermediate_tensor is not None:
        shape, dtype, layout, mem_cfg = _resolve_intermediate_spec(input_tensor)
        if (
            list(intermediate_tensor.shape) != shape
            or intermediate_tensor.dtype != dtype
            or intermediate_tensor.layout != layout
            or intermediate_tensor.memory_config() != mem_cfg
        ):
            raise ValueError(
                f"point_to_point: intermediate_tensor spec must equal the resolved staging spec "
                f"(shape={shape}, dtype={dtype}, layout={layout})"
            )

    # 9. Axis gate (registry model).
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"point_to_point: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"point_to_point: unsupported combination (refinement candidate): {exc}")


def point_to_point(
    input_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
    intermediate_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Copy ``sender_coord``'s shard to ``receiver_coord`` over the TT-Fabric.

    Args:
        input_tensor: mesh-sharded, interleaved tensor (rank >= 2).
        sender_coord: mesh coordinate holding the shard to send.
        receiver_coord: mesh coordinate that receives the shard.
        topology: ``Topology.Linear`` (default) or ``Topology.Ring``.
        output_tensor: optional pre-allocated output written in place and returned.
        intermediate_tensor: optional packet staging tensor (see the resolved spec).

    Returns:
        The output tensor: ``receiver_coord``'s shard equals ``sender_coord``'s
        input shard; every other coordinate's shard is what it was on entry.
    """
    validate(
        input_tensor,
        sender_coord,
        receiver_coord,
        topology=topology,
        output_tensor=output_tensor,
        intermediate_tensor=intermediate_tensor,
    )

    mesh_device = input_tensor.device()

    # The programs write ONLY the receiver device's shard, so the default output
    # must be SEEDED for "every other device's shard is untouched" to be a total
    # statement rather than an undefined one. ttnn.clone with no dtype/memory
    # override is a same-dtype, same-layout, same-memory device copy.
    if output_tensor is None:
        output_tensor = ttnn.clone(input_tensor)

    if intermediate_tensor is None:
        shape, dtype, layout, mem_cfg = _resolve_intermediate_spec(input_tensor)
        intermediate_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            dtype,
            layout,
            mesh_device,
            mem_cfg,
        )

    sem = _get_or_create_semaphore(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(sem)

    mesh_program_descriptor = create_mesh_program_descriptor(
        input_tensor,
        intermediate_tensor,
        output_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
    )
    # Park the semaphore so the framework keeps its L1 alive across cache hits.
    # (Excluded from the program-cache hash, so this does not defeat caching.)
    mesh_program_descriptor.semaphores = [sem]

    # Output tensor last. NO post-dispatch synchronize_device — the parked
    # semaphore is what keeps the GlobalSemaphore alive across cache hits.
    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
