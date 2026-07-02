# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Gathers every device's shard of a mesh-sharded tensor and concatenates all shards
along ``gather_dim`` so that AFTER the op EVERY participating device on the 1-D line
holds the full concatenated tensor (identical on every device). Pure data movement
(identity gather, no arithmetic): PCC ~1.0.

Every device runs the same bidirectional store-and-forward ring role — seed its own
shard, receive from a neighbour, forward to the next hop — coordinated by two cached
op-internal ``GlobalSemaphore``s (an N-party startup barrier + store-and-forward flow
control). This op does NOT wrap, import, or dispatch to any existing all_gather /
all_gather_async op.
"""

from __future__ import annotations

from math import prod

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


from .all_gather_program_descriptor import create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# Pure byte movement: the op copies physical pages verbatim and never
# tilizes/untilizes, so it is dtype/layout/alignment-agnostic. `gather_dim` is an
# index axis handled by canonicalization in validate() (not literal membership):
# only the outermost dim (page-contiguous concat) is supported for now.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "topology": [_Topology.Linear],
    # Index axis (checked by CANONICALIZATION, not literal membership — see validate).
    # Only the outermost dim (page-contiguous concat) is supported for now; -4 is the
    # outermost dim for the rank-4 shards this op is proven on. Higher dims (-3,-2,-1)
    # are the strided-concat refinement path.
    "gather_dim": [-4],
    # Shape-derived (tagged from the shard's last two dims). Format is preserved end
    # to end, so non-tile-aligned shards transfer just like aligned ones.
    "alignment": ["tile_aligned", "non_tile_aligned"],
}

EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: the TWO op-internal semaphores are created ONCE
# per mesh_device (+ one synchronize_device), reused across program-cache hits, never
# recreated.
_SEMAPHORE_CACHE: dict = {}


def _get_or_create_semaphores(mesh_device):
    """Return (barrier_sem, counting_sem), creating + caching them once per device."""
    key = id(mesh_device)
    sems = _SEMAPHORE_CACHE.get(key)
    if sems is None:
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        worker_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        barrier_sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        counting_sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        ttnn.synchronize_device(mesh_device)
        sems = (barrier_sem, counting_sem)
        _SEMAPHORE_CACHE[key] = sems
    return sems


def _canonical_gather_dim(gather_dim: int, rank: int) -> int:
    """Canonicalize to a single (negative) sign convention before the support check."""
    return gather_dim if gather_dim < 0 else gather_dim - rank


def _resolved_output_shape(input_tensor, num_devices, canonical_dim):
    shape = list(input_tensor.shape)
    shape[canonical_dim] *= num_devices
    return shape


def validate(input_tensor, gather_dim, *, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise the
    registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("all_gather: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"all_gather: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")
    num_devices = prod(mesh_shape)
    if num_devices < 2:
        raise ValueError("all_gather: requires at least 2 mesh devices")

    if input_tensor.is_sharded():
        raise ValueError("all_gather: sharded input not yet supported (interleaved only)")

    # gather_dim: canonicalize to the NEGATIVE sign convention BEFORE the axis check so
    # a positive alias (gather_dim=0 ≡ -rank) is not rejected by a literal membership
    # test. The SUPPORTED["gather_dim"] gate below then decides support.
    rank = len(input_tensor.shape)
    canonical_dim = _canonical_gather_dim(gather_dim, rank)
    if not (-rank <= canonical_dim < 0):
        raise ValueError(f"all_gather: gather_dim {gather_dim} is out of range for rank {rank}")

    if output_tensor is not None:
        expected_shape = _resolved_output_shape(input_tensor, num_devices, canonical_dim)
        if (
            list(output_tensor.shape) != expected_shape
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("all_gather: output_tensor spec must equal the resolved output spec")

    # Axis gate (registry model). gather_dim enters in its canonicalized (negative) form.
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
        "gather_dim": canonical_dim,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather: unsupported combination (refinement candidate): {exc}")


def all_gather(
    input_tensor: ttnn.Tensor,
    gather_dim: int,
    *,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Gather every device's shard and concatenate along ``gather_dim``.

    Returns the output tensor, identical on every device == host-side concat of all N
    input shards along ``gather_dim``.
    """
    validate(input_tensor, gather_dim, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()
    num_devices = prod(tuple(mesh_device.shape))
    canonical_dim = _canonical_gather_dim(gather_dim, len(input_tensor.shape))

    # Allocate the full-gathered, replicated output (uniform address across devices, so
    # a fabric write of a block's canonical page range lands in the neighbour's identical
    # range). Every output page is produced by the op, so no input-seeding is needed.
    if output_tensor is None:
        output_shape = _resolved_output_shape(input_tensor, num_devices, canonical_dim)
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(output_shape),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    barrier_sem, counting_sem = _get_or_create_semaphores(mesh_device)
    barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_sem)
    counting_sem_addr = ttnn.get_global_semaphore_address(counting_sem)

    mesh_program_descriptor = create_mesh_program_descriptor(
        input_tensor,
        output_tensor,
        num_devices,
        topology,
        barrier_sem_addr,
        counting_sem_addr,
    )
    # Park both semaphores so the framework keeps their L1 alive across cache hits.
    mesh_program_descriptor.semaphores = [barrier_sem, counting_sem]

    ttnn.generic_op([input_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
