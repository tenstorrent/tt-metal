# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter — self-contained Python CCL op WITH a compute stage (generic_op +
MeshProgramDescriptor).

Sums every device's shard element-wise across all N devices on a 1-D MeshDevice
line, then SCATTERS the sum: device i's output is the i-th of N equal slices of
the summed tensor along ``dim`` (Phase-0: dim=3, the last dim). Per-device
DISTINCT outputs — unlike all_reduce's identical-everywhere sum.

Algorithm — GATHER-THEN-REDUCE-LOCAL-SLICE, two ordered ``ttnn.generic_op``
dispatches on the same command queue:

  * Phase A (fabric): a line store-and-forward gather lands all N full shards into
    an op-internal ``gather_buffer`` (block c at pages ``[c*P_shard, (c+1)*P_shard)``),
    identical on every device — the proven all_reduce Phase-A structure verbatim.
  * Phase B (compute): the scatter collapses into pure local addressing — per
    output-tile position, sum the N gathered blocks' tiles at the slice-i source
    index (``sum_blocks``) and write to the output slice.

Because both dispatches share the device command queue, Phase A completes on
device i before Phase B reads its ``gather_buffer`` — no extra cross-device
barrier. This op does NOT wrap, import, call, or dispatch to any existing
reduce_scatter / all_reduce / all_gather op.

Primary proven case: bfloat16, TILE_LAYOUT, dim=3, Linear topology, on a 4-chip
Blackhole ``(1, 4)`` line mesh with ``FABRIC_1D``.
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


from .reduce_scatter_program_descriptor import (
    create_gather_mesh_program_descriptor,
    create_reduce_mesh_program_descriptor,
)

_TILE = 32  # tile edge (elements)


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# The reduction is always the full element-wise SUM across devices; the scatter
# dim is an op kwarg axis (Phase-0: dim 3 only, canonicalized before the gate).
# Every accepted input is tile-aligned by construction (TILE_LAYOUT + the
# W % (N*32) structural gate), so INPUT_TAGGERS is empty.

INPUT_TAGGERS: dict = {}

SUPPORTED = {
    # A bf16 sum of N terms accumulates rounding (threshold 0.99, not 0.995), and
    # float32 is the higher-precision secondary dtype (fp32_dest_acc in Phase B).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # The reduction is a tile compute — TILE_LAYOUT only.
    "layout": [ttnn.TILE_LAYOUT],
    # Linear is the proven primary (and only verified) topology.
    "topology": [_Topology.Linear],
    # Index axis, canonicalized to POSITIVE (rank 4) BEFORE the membership test:
    # dim=-1 ≡ 3. A literal test on the raw value would reject the legal alias.
    "dim": [3],
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


def validate(input_tensor, *, dim, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell (NotImplementedError
    subclasses).

    Returns ``(num_devices, canonical_dim)``.
    """
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("reduce_scatter: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"reduce_scatter: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")

    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("reduce_scatter: requires at least 2 mesh devices on the line")

    if input_tensor.is_sharded():
        raise ValueError("reduce_scatter: sharded input not yet supported (interleaved only)")

    shape = list(input_tensor.shape)
    rank = len(shape)
    if rank != 4:
        raise ValueError(f"reduce_scatter: expected a rank-4 input shard, got rank {rank} ({shape})")

    # Load-bearing: the Phase-A fabric writer sends align(page_size, l1_alignment)
    # bytes per page while the gather_buffer TensorAccessor spaces pages by the raw
    # page_size — a non-16-aligned page_size would overrun into the next page on
    # the wire. TILE pages are already 16-aligned; keep the guard explicit.
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"reduce_scatter: per-shard page size ({page} B) must be 16-byte aligned")

    # Canonicalize the index axis BEFORE the SUPPORTED membership test (dim=-1 ≡ 3).
    canonical_dim = dim if dim >= 0 else dim + rank
    if not (0 <= canonical_dim < rank):
        raise ValueError(f"reduce_scatter: dim={dim} out of range for rank-{rank} input")

    # Axis gate (registry model) — runs BEFORE the dim-specific shape gate, so an
    # out-of-SUPPORTED dim refuses with UnsupportedAxisValue, not ValueError.
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
        "dim": canonical_dim,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"reduce_scatter: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"reduce_scatter: unsupported combination (refinement candidate): {exc}")

    # Structural: every device's output slice must be a whole number of tiles —
    # rejected loudly, never padded.
    if shape[canonical_dim] % (num_devices * _TILE) != 0:
        raise ValueError(
            f"reduce_scatter: shape[{canonical_dim}]={shape[canonical_dim]} must be divisible by "
            f"num_devices * tile ({num_devices} * {_TILE} = {num_devices * _TILE}); no padding is applied"
        )

    if output_tensor is not None:
        out_shape = list(shape)
        out_shape[canonical_dim] //= num_devices
        if (
            list(output_tensor.shape) != out_shape
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError(
                f"reduce_scatter: output_tensor spec must equal the derived output spec "
                f"(shape {out_shape}, dtype {input_tensor.dtype}, layout {input_tensor.layout})"
            )

    return num_devices, canonical_dim


def reduce_scatter(
    input_tensor: ttnn.Tensor,
    dim: int = 3,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Sum every device's shard element-wise across all N devices on the line, then
    scatter: device i keeps slice i (of N equal slices along ``dim``) of the sum.

    Output shard shape = input shard shape with ``shape[dim] / N``; same
    dtype/layout. Returns the supplied ``output_tensor`` handle when given.
    """
    num_devices, canonical_dim = validate(input_tensor, dim=dim, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()

    # Output slice (shape[dim] / N). Allocate if not supplied; every output page is
    # overwritten, so no seeding is required.
    if output_tensor is None:
        out_shape = list(input_tensor.shape)
        out_shape[canonical_dim] //= num_devices
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    # Op-internal gather_buffer: N full-shard blocks stacked on dim 0. Mesh-
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
    # module-level _SEMAPHORE_CACHE already holds a live reference, so parking is
    # belt-and-suspenders; guard it for older _ttnn bindings.
    if hasattr(gather_mpd, "semaphores"):
        gather_mpd.semaphores = [sem]
    ttnn.generic_op([input_tensor, gather_buffer], gather_mpd)

    # --- Phase B: scatter-reduce (local compute; the scatter IS the addressing) ---
    reduce_mpd = create_reduce_mesh_program_descriptor(gather_buffer, output_tensor, num_devices, canonical_dim)
    ttnn.generic_op([gather_buffer, output_tensor], reduce_mpd)

    return output_tensor
