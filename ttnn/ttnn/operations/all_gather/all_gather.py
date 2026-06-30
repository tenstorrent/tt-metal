# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Gathers every device's shard of a mesh-sharded tensor and concatenates all
shards along ``gather_dim`` so that AFTER the op EVERY participating device on
the 1-D line holds the full concatenated tensor (identical on every device). It
is pure data movement (identity gather, no arithmetic; PCC ~1.0).

Newly authored ring/line dataflow kernels under ``kernels/`` are assembled by a
``ttnn.generic_op`` over a ``ttnn.MeshProgramDescriptor`` — this op does NOT
wrap, import, call, or dispatch to any existing all_gather / all_gather_async
op. Each device runs a bidirectional store-and-forward ring role (seed its own
shard, receive from a neighbour, forward to the next hop), coordinated by one
cached op-internal ``GlobalSemaphore``.

Primary proven case: ``gather_dim=0`` (page-contiguous concat), bfloat16,
TILE_LAYOUT, Linear topology, on a Wormhole T3K ``(1, 8)`` line mesh with
``FABRIC_1D``.
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


from .all_gather_program_descriptor import create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# all_gather is pure byte movement (never tilizes/untilizes), so it is
# format-agnostic in principle. `alignment` IS a shape-derived axis tagged from
# the per-device shard's last two dims: the op copies the physical pages (padded
# tiles for TILE, last-dim rows for ROW_MAJOR) verbatim, so non-tile-aligned
# shards gather just like aligned ones.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    # Pure data movement: every fixed-width dtype is correct in principle. The
    # proven primary set is bfloat16 + float32 (the acceptance-test dtypes).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    # Ring is a noted extension (kernel selects it via the slice-walk modulo math);
    # Linear is the proven primary topology.
    "topology": [_Topology.Linear],
    # Index axis, canonicalized to NEGATIVE before the membership test.
    # gather_dim=0 (page-contiguous concat) is the proven primary case: -4 for
    # the rank-4 per-device shards in the acceptance test.
    "gather_dim": [-4],
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


def _canonical_gather_dim(gather_dim: int, rank: int) -> int:
    """Canonicalize to a single (negative) sign convention before the support check."""
    return gather_dim if gather_dim < 0 else gather_dim - rank


def _resolve_output_shape(input_shape, gather_dim, num_devices):
    out = list(input_shape)
    out[gather_dim] = out[gather_dim] * num_devices
    return out


def validate(input_tensor, gather_dim, *, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("all_gather: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"all_gather: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")

    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("all_gather: requires at least 2 mesh devices on the line")

    if input_tensor.is_sharded():
        raise ValueError("all_gather: sharded input not yet supported (interleaved only)")

    rank = len(input_tensor.shape)
    gd = _canonical_gather_dim(gather_dim, rank)

    # Load-bearing: the fabric writer sends align(page_size, l1_alignment) bytes per
    # page (ccl_helpers_dataflow.inl:35). The output TensorAccessor spaces pages by the
    # raw page_size, so a non-16-aligned page_size would make the on-wire (rounded-up)
    # payload overrun into the next output page. Requiring 16B-aligned pages makes the
    # round-up a no-op. (page % 16 == 0 already implies page == 16 is fine; no extra clause.)
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"all_gather: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        expected = _resolve_output_shape(tuple(input_tensor.shape), gd, num_devices)
        if (
            list(output_tensor.shape) != list(expected)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("all_gather: output_tensor spec must equal the resolved output spec")

    # Axis gate (registry model).
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
        "gather_dim": gd,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather: unsupported combination (refinement candidate): {exc}")

    return gd, num_devices


def all_gather(
    input_tensor: ttnn.Tensor,
    gather_dim: int,
    *,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Gather every device's shard and concat along ``gather_dim``.

    After the op every participating device holds the full concatenated tensor
    (identical on every device).
    """
    gd, num_devices = validate(input_tensor, gather_dim, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()

    # Allocate the (replicated) output if not supplied; the op overwrites every
    # output page, so no input-seeding/clone is required.
    if output_tensor is None:
        out_shape = _resolve_output_shape(list(input_tensor.shape), gd, num_devices)
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    sem = _get_or_create_semaphore(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(sem)

    mesh_program_descriptor = create_mesh_program_descriptor(input_tensor, output_tensor, topology, sem_addr)
    # Park the semaphore so the framework keeps its L1 alive across cache hits.
    mesh_program_descriptor.semaphores = [sem]

    ttnn.generic_op([input_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
