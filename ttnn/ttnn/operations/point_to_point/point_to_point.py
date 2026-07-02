# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Sends one mesh device's interleaved shard to another device over the Tenstorrent
fabric. Pure data movement (no arithmetic): the receiver device's output shard
becomes a bit-for-bit copy of the sender's input shard, and every other device's
output shard is left equal to its own input shard.

The op (a) seeds ``output == input`` on every device, (b) dispatches a two-program
mesh workload (SEND at ``sender_coord``, RECEIVE at ``receiver_coord``) that
overwrites only the receiver shard via the fabric, coordinated by one cached
op-internal ``GlobalSemaphore``.
"""

from __future__ import annotations

import ttnn

# Topology lives on the C++ module; the top-level ``ttnn.Topology`` alias is only
# bound AFTER ``ttnn.operations`` is auto-imported, so reference the source module
# directly to stay safe at eager-import time.
from ttnn._ttnn.operations.ccl import Topology as _Topology

try:  # registry-model refusal types; fall back when the shared module is absent.
    from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
except ImportError:  # pragma: no cover

    class UnsupportedAxisValue(NotImplementedError):
        pass

    class ExcludedCell(NotImplementedError):
        pass


from .point_to_point_program_descriptor import create_mesh_program_descriptor, resolve_intermediate_spec


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# The 16-byte page-size constraint is a shape x dtype validate() gate (kept
# satisfiable by INPUTS), not an axis. `alignment` IS a shape-derived axis
# (the golden suite tags it from the per-device shard's last two dims): the op
# is pure byte movement and never tilizes/untilizes, so it copies the physical
# pages (padded tiles for TILE, last-dim rows for ROW_MAJOR) verbatim and is
# alignment-agnostic — both tile_aligned and non_tile_aligned are supported.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    # Pure byte movement: every fixed-width dtype is correct. bfloat8_b is a
    # tiled block-float format (TILE only — the {bf8b, ROW_MAJOR} cell is
    # structurally INVALID and ttnn cannot construct it).
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
    # Shape-derived (tagged from the shard's last two dims). Format is preserved
    # end to end, so non-tile-aligned shards transfer just like aligned ones.
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


def validate(input_tensor, sender_coord, receiver_coord, *, topology, output_tensor, intermediate_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("point_to_point: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2:
        raise ValueError(f"point_to_point: expected a 2-D mesh view, got shape {mesh_shape}")

    s = (sender_coord[0], sender_coord[1])
    r = (receiver_coord[0], receiver_coord[1])
    if s == r:
        raise ValueError("point_to_point: cannot send to self (sender_coord == receiver_coord)")
    for name, c in (("sender_coord", s), ("receiver_coord", r)):
        if not (0 <= c[0] < mesh_shape[0] and 0 <= c[1] < mesh_shape[1]):
            raise ValueError(f"point_to_point: {name} {c} is outside the mesh {mesh_shape}")
    if s[0] != r[0] and s[1] != r[1]:
        raise ValueError(
            "point_to_point: sender_coord and receiver_coord must share a row or column (1-D fabric route)"
        )

    if input_tensor.is_sharded():
        raise ValueError("point_to_point: sharded input not yet supported (interleaved only)")

    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"point_to_point: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        if (
            tuple(output_tensor.shape) != tuple(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("point_to_point: output_tensor spec must equal the resolved output spec (== input spec)")

    if intermediate_tensor is not None:
        spec = resolve_intermediate_spec(input_tensor)
        if (
            tuple(intermediate_tensor.shape) != tuple(spec.shape)
            or intermediate_tensor.dtype != spec.dtype
            or intermediate_tensor.layout != spec.layout
        ):
            raise ValueError("point_to_point: intermediate_tensor spec must equal the resolved intermediate spec")

    # Axis gate (registry model).
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, "topology": topology}
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
    """Send ``input_tensor``'s shard at ``sender_coord`` to ``receiver_coord``.

    Returns the output tensor: ``output[receiver_coord] == input[sender_coord]``
    and ``output[d] == input[d]`` for every other device ``d``.
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

    # Op-internal per-packet staging buffer that lands the fabric payload.
    if intermediate_tensor is None:
        intermediate_tensor = ttnn.allocate_tensor_on_device(resolve_intermediate_spec(input_tensor), mesh_device)

    # Seed output == input on EVERY device (guarantees non-receiver shards stay
    # unchanged). The receiver's writer overwrites only the receiver shard.
    if output_tensor is None:
        output_tensor = ttnn.clone(input_tensor, memory_config=input_tensor.memory_config())
    else:
        ttnn.copy(input_tensor, output_tensor)

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
    mesh_program_descriptor.semaphores = [sem]

    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
