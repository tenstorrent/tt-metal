# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Gathers every device's shard of a mesh-sharded tensor and concatenates all shards
along ``gather_dim`` so that AFTER the op EVERY participating device holds the full
concatenated tensor (identical on every device). Pure data movement (identity gather,
no arithmetic): the output is a bit-for-bit gather (PCC ~ 1.0).

The op allocates a replicated full-shape output (mirrored addresses across the line),
creates three cached op-internal ``GlobalSemaphore``s, and dispatches a per-device
``ttnn.MeshProgramDescriptor`` running a bidirectional store-and-forward ring
(one program per ``MeshCoordinate`` on the (1, N) line).
"""

from __future__ import annotations

import ttnn

# Topology lives on the C++ module; the top-level ``ttnn.Topology`` alias is only bound
# AFTER ``ttnn.operations`` is auto-imported, so reference the source module directly.
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
# all_gather is pure byte movement (never tilizes/untilizes), so it is format- and
# alignment-agnostic in principle. Phase 2 (this milestone) proves gather_dim=0,
# TILE_LAYOUT, bfloat16/float32, Linear. The 16-byte page-size constraint is a
# shape x dtype validate() gate (kept satisfiable by INPUTS), not an axis.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    # bfloat16 primary, float32 second required dtype (bfloat8_b is a TARGET refinement).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # TILE_LAYOUT proven (ROW_MAJOR is a TARGET refinement).
    "layout": [ttnn.TILE_LAYOUT],
    # Linear line proven (Ring is a TARGET refinement).
    "topology": [_Topology.Linear],
    # gather_dim is an index axis in the NEGATIVE convention. -4 == gather_dim 0 for
    # the rank-4 shards (the proven contiguous-slice case); -3/-2/-1 are refinements.
    "gather_dim": [-4],
    # Shape-derived: TILE pads sub-tile shards to full tiles, so byte movement is
    # identical for non-tile-aligned shards.
    "alignment": ["tile_aligned", "non_tile_aligned"],
}

EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: the three ring semaphores are created ONCE per
# mesh_device (+ one synchronize_device), reused across program-cache hits.
_SEMAPHORE_CACHE: dict = {}


def _get_or_create_semaphores(mesh_device):
    """(barrier_sem, forward_sem, backward_sem), created once per mesh on the two
    worker cores (0,0)-(0,1) and cached on the module."""
    key = id(mesh_device)
    sems = _SEMAPHORE_CACHE.get(key)
    if sems is None:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))])
        barrier_sem = ttnn.create_global_semaphore(mesh_device, cores, 0)
        forward_sem = ttnn.create_global_semaphore(mesh_device, cores, 0)
        backward_sem = ttnn.create_global_semaphore(mesh_device, cores, 0)
        ttnn.synchronize_device(mesh_device)
        sems = (barrier_sem, forward_sem, backward_sem)
        _SEMAPHORE_CACHE[key] = sems
    return sems


def _canonical_gather_dim(gather_dim, rank):
    """Canonicalize to the NEGATIVE convention (matches feature_spec TARGET)."""
    return gather_dim if gather_dim < 0 else gather_dim - rank


def validate(input_tensor, gather_dim, *, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise the
    registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("all_gather: input_tensor must be on a MeshDevice line of >= 2 devices")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2:
        raise ValueError(f"all_gather: expected a 2-D mesh view, got shape {mesh_shape}")
    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("all_gather: input must be on a MeshDevice line of >= 2 devices")
    if mesh_shape[0] != 1:
        raise ValueError(f"all_gather: expected a 1-D mesh line (1, N), got {mesh_shape}")

    rank = len(input_tensor.shape)
    if not (-rank <= gather_dim < rank):
        raise ValueError(f"all_gather: gather_dim {gather_dim} out of range for rank {rank}")

    if input_tensor.is_sharded():
        raise ValueError("all_gather: sharded input not yet supported (interleaved only)")

    page = input_tensor.buffer_page_size()
    if page % 16 != 0 and page != 16:
        raise ValueError(f"all_gather: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        gd = _canonical_gather_dim(gather_dim, rank)
        axis = gd % rank
        expected = list(input_tensor.shape)
        expected[axis] *= num_devices
        if (
            tuple(output_tensor.shape) != tuple(expected)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("all_gather: output_tensor spec must equal the resolved output spec")

    # Axis gate (registry model). gather_dim is canonicalized BEFORE the membership check.
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
        "gather_dim": _canonical_gather_dim(gather_dim, rank),
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

    Returns an output tensor whose content is identical on every device: the full
    ``concat(S_0, ..., S_{N-1}, dim=gather_dim)``.
    """
    validate(input_tensor, gather_dim, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()
    num_devices = tuple(mesh_device.shape)[0] * tuple(mesh_device.shape)[1]
    rank = len(input_tensor.shape)
    axis = _canonical_gather_dim(gather_dim, rank) % rank

    # Replicated full-shape output (mirrored addresses across the line): the own slot is
    # a local write, the remote slots land via the fabric. No host-side seeding needed.
    if output_tensor is None:
        output_shape = list(input_tensor.shape)
        output_shape[axis] *= num_devices
        output_spec = ttnn.TensorSpec(
            ttnn.Shape(output_shape),
            input_tensor.dtype,
            input_tensor.layout,
            input_tensor.memory_config().buffer_type,
        )
        output_tensor = ttnn.allocate_tensor_on_device(output_spec, mesh_device)

    barrier_sem, forward_sem, backward_sem = _get_or_create_semaphores(mesh_device)

    mesh_program_descriptor = create_mesh_program_descriptor(
        input_tensor,
        output_tensor,
        topology,
        ttnn.get_global_semaphore_address(barrier_sem),
        ttnn.get_global_semaphore_address(forward_sem),
        ttnn.get_global_semaphore_address(backward_sem),
    )
    # Park the semaphores so the framework keeps their L1 alive across cache hits.
    mesh_program_descriptor.semaphores = [barrier_sem, forward_sem, backward_sem]

    ttnn.generic_op([input_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
