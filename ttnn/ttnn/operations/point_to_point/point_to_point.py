# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Copies one mesh device's interleaved shard of a mesh-sharded tensor to another
device over the Tenstorrent fabric. It is pure data movement (identity byte copy,
no arithmetic; PCC ~1.0): after the op the RECEIVER device's shard equals the
SENDER device's input shard bit-for-bit, and every other device's shard is
unchanged.

Newly authored sender/receiver dataflow kernels under ``kernels/`` are assembled
by a ``ttnn.generic_op`` over a ``ttnn.MeshProgramDescriptor`` (one
``ProgramDescriptor`` per participating coordinate). This op does NOT wrap,
import, call, or dispatch to the bound C++ ``ttnn.point_to_point``.

Cross-chip coordination uses ONE op-internal ``GlobalSemaphore`` (created once
per mesh_device, parked on ``MeshProgramDescriptor.semaphores`` so its L1 survives
program-cache hits) carrying a two-phase handshake:
  (1) receiver -> sender "ready", then (2) sender -> receiver "done".
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


from .point_to_point_program_descriptor import PacketDims, create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# point_to_point is pure byte movement (never tilizes/untilizes), so it is
# format-agnostic in principle. `alignment` IS a shape-derived axis tagged from
# the per-device shard's last two dims: the op copies the physical pages (padded
# tiles for TILE, last-dim rows for ROW_MAJOR) verbatim, so non-tile-aligned
# shards transfer just like aligned ones.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    # Pure data movement: every fixed-width dtype is correct in principle. The
    # proven primary set is the acceptance-test dtypes (bf16/fp32/bf8b). Integer
    # passthrough (uint16/int32/uint32) stays a refinement candidate.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    # Both topologies run on the same FABRIC_1D fabric; the route (and Ring
    # short-way choice) is owned by ccl_dm_route, so the kernels are identical.
    "topology": [_Topology.Linear, _Topology.Ring],
    "alignment": ["tile_aligned", "non_tile_aligned"],
}

EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: created ONCE per mesh_device (+ one
# synchronize_device), reused across program-cache hits, never recreated.
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


def _coord_in_mesh(coord, mesh_shape) -> bool:
    return 0 <= coord[0] < mesh_shape[0] and 0 <= coord[1] < mesh_shape[1]


def _compute_packet_dims(input_tensor) -> PacketDims:
    """Frame the input shard's pages into fabric packets (owns bf16 bit_floor +
    both packing regimes). Drives the intermediate shape and all packet args."""
    l1_alignment = ttnn.get_l1_alignment()
    pd = ttnn._ttnn.fabric.ccl_packet_dims(
        input_tensor.dtype,
        input_tensor.buffer_page_size(),
        input_tensor.buffer_num_pages(),
        l1_alignment,
    )
    return PacketDims(
        packet_size_bytes=pd.packet_size_bytes,
        pages_per_packet=pd.pages_per_packet,
        page_segments=pd.page_segments,
        total_packets=pd.total_packets,
    )


def _datum_size(dtype) -> int:
    """Bytes per datum, block-float-safe. ttnn.element_size (tt::datum_size) throws
    for block-float formats (bfp8/bfp4/...) because they have no fixed per-element
    byte width. The intermediate is a pure byte landing zone whose addressing is
    driven by the packet_size override (not the datum), so a byte-level datum of 1
    is correct for block-float: packet_page_dim = packet_size_bytes safely
    over-allocates the (tile-padded) intermediate buffer."""
    try:
        return ttnn.element_size(dtype)
    except (ValueError, RuntimeError):
        return 1


def _resolve_intermediate_spec(input_tensor, packet_dims: PacketDims):
    """Intermediate (fabric landing zone) shape/dtype/layout/memory match the input
    layout, but the shape is packet-framed: {total_packets, packet_page_dim}."""
    packet_page_dim = packet_dims.packet_size_bytes // _datum_size(input_tensor.dtype)
    return ttnn.Shape([packet_dims.total_packets, packet_page_dim])


def validate(input_tensor, sender_coord, receiver_coord, *, topology, output_tensor, intermediate_tensor, packet_dims):
    """Runtime gate. Structural errors raise ValueError; axis refusals raise the
    registry-model UnsupportedAxisValue / ExcludedCell. Structural checks first."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("point_to_point: input must be on a MeshDevice")

    mesh_shape = tuple(device.shape)

    # 2. cannot send to self
    if sender_coord == receiver_coord:
        raise ValueError("point_to_point: cannot send to self (sender_coord == receiver_coord)")
    # 3 / 4. coords inside the mesh view
    if not _coord_in_mesh(sender_coord, mesh_shape):
        raise ValueError(f"point_to_point: sender_coord {tuple(sender_coord)} outside mesh {mesh_shape}")
    if not _coord_in_mesh(receiver_coord, mesh_shape):
        raise ValueError(f"point_to_point: receiver_coord {tuple(receiver_coord)} outside mesh {mesh_shape}")
    # 5. 1-D fabric: coords must share a row or a column (diagonal is illegal)
    if sender_coord[0] != receiver_coord[0] and sender_coord[1] != receiver_coord[1]:
        raise ValueError("point_to_point: sender_coord and receiver_coord must share a row or column")
    # 6. interleaved only
    if input_tensor.is_sharded():
        raise ValueError("point_to_point: sharded input not yet supported (interleaved only)")

    # 7. Load-bearing: the fabric writer sends align(page_size, l1_alignment) bytes
    # per page (ccl_helpers_dataflow.inl:35). The output TensorAccessor spaces pages
    # by the raw page_size, so a non-16-aligned page would overrun into the next
    # output page. Requiring a 16B-aligned page makes the round-up a no-op.
    page = input_tensor.buffer_page_size()
    l1_alignment = ttnn.get_l1_alignment()
    if page % l1_alignment != 0 and page != l1_alignment:
        raise ValueError(f"point_to_point: page size ({page} B) must be 16-byte aligned")

    # 8. supplied output_tensor spec must equal the resolved output spec (== input).
    if output_tensor is not None:
        if (
            list(output_tensor.shape) != list(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("point_to_point: output_tensor spec must equal input spec")

    # 9. supplied intermediate_tensor spec must equal the resolved intermediate spec.
    if intermediate_tensor is not None:
        expected_shape = _resolve_intermediate_spec(input_tensor, packet_dims)
        if (
            list(intermediate_tensor.shape) != list(expected_shape)
            or intermediate_tensor.dtype != input_tensor.dtype
            or intermediate_tensor.layout != input_tensor.layout
            or intermediate_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("point_to_point: intermediate_tensor spec mismatch")

    # 10. Axis gate (registry model). No index/sign axis to canonicalize.
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
    *,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
    intermediate_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Send ``input_tensor``'s shard from ``sender_coord`` to ``receiver_coord``.

    After the op the receiver device's shard equals the sender's input shard;
    every other device's shard is unchanged.

    When ``output_tensor is None`` the op aliases ``input_tensor`` (in-place):
    only the receiver device's shard is overwritten, so the sender's own shard
    and every non-participating device's shard stay bit-identical to the input.
    """
    packet_dims = _compute_packet_dims(input_tensor)

    validate(
        input_tensor,
        sender_coord,
        receiver_coord,
        topology=topology,
        output_tensor=output_tensor,
        intermediate_tensor=intermediate_tensor,
        packet_dims=packet_dims,
    )

    mesh_device = input_tensor.device()

    # Output defaults to the input tensor (in-place alias): only the receiver's
    # shard is written; every other shard stays equal to the input.
    if output_tensor is None:
        output_tensor = input_tensor

    # The intermediate (fabric landing zone) is a mesh tensor at the SAME
    # device-local address on both endpoints — the sender fabric-writes packets
    # into the receiver's copy; the receiver reads its local copy back.
    if intermediate_tensor is None:
        intermediate_shape = _resolve_intermediate_spec(input_tensor, packet_dims)
        intermediate_tensor = ttnn.allocate_tensor_on_device(
            intermediate_shape,
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
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
        packet_dims,
    )
    # Park the semaphore so the framework keeps its L1 alive across cache hits.
    mesh_program_descriptor.semaphores = [sem]

    # io_tensors: input, intermediate, output (output last). When output aliases
    # input the last element is the input tensor itself (in-place).
    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
