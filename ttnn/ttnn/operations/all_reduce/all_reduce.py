# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — self-contained Python CCL op WITH a compute stage (generic_op + MeshProgramDescriptor).

Sums every device's shard element-wise across all N devices on a 1-D MeshDevice
line and leaves the IDENTICAL sum on every device (same shape/dtype/layout as one
input shard). Unlike the pure-movement CCLs (point_to_point identity, all_gather
concat), the element values change (they are summed), so this op combines the
fabric dataflow helper (cross-device movement) with the compute (TRISC) helpers
(the element-wise add).

Algorithm — gather-then-reduce, two ordered ``ttnn.generic_op`` dispatches on the
same command queue:

  * Phase A (fabric): a line store-and-forward gather lands all N shards into an
    op-internal ``gather_buffer`` (block c at pages ``[c*P, (c+1)*P)``), identical
    on every device. Structurally the all_gather ``gather_dim=0`` pattern.
  * Phase B (compute): a local element-wise N-way tile sum reduces the N blocks
    into the output shard (``output[i] = Σ_c gather_buffer[c*P + i]``).

Because both dispatches share the device command queue, Phase A completes on
device i before Phase B reads its ``gather_buffer`` — no extra cross-device
barrier is needed. This op does NOT wrap, import, call, or dispatch to any
existing all_reduce / reduce_scatter / all_gather op.

Primary proven case: bfloat16, TILE_LAYOUT, Linear topology, on a Wormhole T3K
``(1, 8)`` line mesh with ``FABRIC_1D``.
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


from .all_reduce_program_descriptor import (
    create_gather_mesh_program_descriptor,
    create_reduce_mesh_program_descriptor,
)


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# The reduction is always the full element-wise SUM across devices — there is no
# reduce-dim parameter and no shape-derived axis (every INPUT is tile-aligned by
# construction: the reduction is a tile compute on TILE_LAYOUT). So INPUT_TAGGERS
# is empty and the axis set is just the two float dtypes, TILE, and Linear.

INPUT_TAGGERS: dict = {}

SUPPORTED = {
    # A bf16 sum of N terms accumulates rounding (threshold 0.99, not 0.995), and
    # float32 is the higher-precision secondary dtype (fp32_dest_acc in Phase B).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # The reduction is a tile compute — TILE_LAYOUT only.
    "layout": [ttnn.TILE_LAYOUT],
    # Linear is the proven primary (and only verified) topology.
    "topology": [_Topology.Linear],
}

EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: created ONCE per mesh_device (+ one
# synchronize_device), reused across program-cache hits, never recreated. Only
# Phase A uses it (Phase B has no cross-device sync).
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


def _num_line_devices(mesh_device) -> int:
    """Number of devices on the 1-D line (mesh view is (1, N))."""
    n = 1
    for d in tuple(mesh_device.shape):
        n *= d
    return n


def validate(input_tensor, *, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("all_reduce: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"all_reduce: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")

    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("all_reduce: requires at least 2 mesh devices on the line")

    if input_tensor.is_sharded():
        raise ValueError("all_reduce: sharded input not yet supported (interleaved only)")

    # Load-bearing: the Phase-A fabric writer sends align(page_size, l1_alignment)
    # bytes per page. The gather_buffer TensorAccessor spaces pages by the raw
    # page_size, so a non-16-aligned page_size would make the on-wire (rounded-up)
    # payload overrun into the next page. TILE pages are already 16-aligned; keep
    # the guard explicit to mirror all_gather.
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"all_reduce: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        if (
            list(output_tensor.shape) != list(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("all_reduce: output_tensor spec must equal the input shard spec")

    # Axis gate (registry model).
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_reduce: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_reduce: unsupported combination (refinement candidate): {exc}")

    return num_devices


def all_reduce(
    input_tensor: ttnn.Tensor,
    *,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Sum every device's shard element-wise across all N devices on the line.

    After the op every participating device holds the IDENTICAL element-wise sum
    of all N input shards (same shape/dtype/layout as one input shard).
    """
    num_devices = validate(input_tensor, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()

    # Output shard (= input shard shape). Allocate a replicated buffer if not
    # supplied; the op overwrites every output page, so no seeding is required.
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    # Op-internal gather_buffer: N shard-blocks stacked on the outer dim. Mesh-
    # allocated interleaved => uniform buffer address across devices, which is what
    # lets the Phase-A fabric write_page target a neighbour's block via the LOCAL
    # accessor base address routed one hop.
    gb_shape = [input_tensor.shape[0] * num_devices, *list(input_tensor.shape)[1:]]
    gather_buffer = ttnn.allocate_tensor_on_device(
        ttnn.Shape(gb_shape),
        input_tensor.dtype,
        input_tensor.layout,
        mesh_device,
        input_tensor.memory_config(),
    )

    # --- Phase A: line store-and-forward gather (fabric) ---
    sem = _get_or_create_semaphore(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(sem)
    gather_mpd = create_gather_mesh_program_descriptor(input_tensor, gather_buffer, topology, sem_addr)
    # Park the semaphore so the framework keeps its L1 alive across cache hits. The
    # module-level _SEMAPHORE_CACHE already holds a live reference (the GlobalSemaphore
    # allocation persists), so parking is belt-and-suspenders; guard it for older
    # _ttnn bindings that predate MeshProgramDescriptor.semaphores.
    if hasattr(gather_mpd, "semaphores"):
        gather_mpd.semaphores = [sem]
    ttnn.generic_op([input_tensor, gather_buffer], gather_mpd)

    # --- Phase B: local element-wise N-way tile sum (compute) ---
    reduce_mpd = create_reduce_mesh_program_descriptor(gather_buffer, output_tensor, num_devices)
    ttnn.generic_op([gather_buffer, output_tensor], reduce_mpd)

    return output_tensor
