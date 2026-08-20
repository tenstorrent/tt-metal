# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter — self-contained Python CCL op WITH a compute stage (generic_op +
MeshProgramDescriptor). The compute-CCL probe.

Sums every device's shard element-wise across the N devices of a 1-D MeshDevice
line, then SCATTERS the sum: device i keeps only slice i (of N equal slices along
``dim``) of the sum — a per-device-DISTINCT output, unlike all_reduce:

    output_i[...] = (Σ_{c=0}^{N-1} shard_c[...])[slice i along dim]
    output.shape[dim] = input.shape[dim] / N

Algorithm — GATHER-THEN-REDUCE-LOCAL-SLICE, two ordered ``ttnn.generic_op``
dispatches on the same command queue:

  * Phase A (fabric): a store-and-forward gather lands all N full shards into
    an op-internal ``gather_buffer`` (block c at pages ``[c*P, (c+1)*P)``),
    identical on every device — the proven all_reduce Phase-A structure. The
    topology selects the block-flow table only: Linear relays every block all
    the way down the line; Ring (Refinement 1) relays each direction's
    SHORT-WAY half of the ring across the wrap link (~2x less relay traffic,
    same landed contents).
  * Phase B (compute): a local N-way tile sum over ONLY the tile positions of
    device i's slice (shared-schedule SliceRowWalker addressing +
    compute_kernel_lib::sum_blocks), written to the ``[dim]/N`` output.

Because both dispatches share the device command queue, Phase A completes on
device i before Phase B reads its ``gather_buffer`` — no extra cross-device
barrier is needed. This op does NOT wrap, import, call, or dispatch to any
existing reduce_scatter / all_reduce / all_gather op.

Primary proven case: bfloat16, TILE_LAYOUT, dim=3, Linear topology, on a
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

_RANK = 4  # rank pinned to 4 (dim canonicalization is `dim + 4`)
_TILE = 32


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# No shape-derived axis: every golden INPUT is chosen valid for every TARGET dim,
# so INPUT_TAGGERS is empty. `dim` MUST be a key even though Phase-0 has a single
# value — the golden harness derives xfail marks by iterating SUPPORTED, and a
# missing axis surfaces unimplemented dim=2 as a hard failure instead of the
# expected UnsupportedAxisValue.

INPUT_TAGGERS: dict = {}

SUPPORTED = {
    # A bf16 sum of N terms accumulates rounding (threshold 0.99, not 0.995);
    # float32 is the higher-precision secondary dtype (fp32_dest_acc in Phase B).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # The reduction is a tile compute — TILE_LAYOUT only.
    "layout": [ttnn.TILE_LAYOUT],
    # Linear is the proven primary topology. Ring (Refinement 1) reuses the same
    # store-and-forward gather with a ring-modular block-flow table: each
    # direction carries only its short-way half of the ring (fwd N//2 blocks,
    # bwd (N-1)//2), and the wrap-link hops route via ccl_dm_route(.., Ring)
    # under the SAME FABRIC_1D fabric config as Linear (hardware-probed).
    "topology": [_Topology.Linear, _Topology.Ring],
    # Scatter dims, POSITIVE convention. Negative aliases are canonicalized
    # BEFORE the membership test (-1 ≡ 3, -2 ≡ 2). dim=2 was promoted by the
    # verifier: the host slice rows (_slice_quantities), the kernel's
    # is_supported_scatter_dim static_assert, and the SliceRowWalker math all
    # generalize over the scatter dim; hardware-verified on the (1, 4) line.
    "dim": [3, 2],
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


def _canonicalize_dim(dim: int) -> int:
    """Positive-convention canonicalization (rank pinned to 4): -1 ≡ 3, -2 ≡ 2."""
    return dim if dim >= 0 else dim + _RANK


def validate(input_tensor, *, dim, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell.

    Ordering: universal structural checks (needed to even form the axes dict)
    raise ValueError first; then the AXIS GATE (typed refusals); then the
    axis-value-DEPENDENT structural checks (tile alignment, slice divisibility,
    output spec). The gate runs before the dependent checks so an
    out-of-SUPPORTED axis value (e.g. an unimplemented scatter dim or layout)
    always yields the typed UnsupportedAxisValue the registry contract
    requires, never a shape-derived ValueError computed under the wrong axis.

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

    shape = list(input_tensor.shape)
    if len(shape) != _RANK:
        raise ValueError(f"reduce_scatter: expected a rank-4 input shard, got rank {len(shape)}")

    canonical_dim = _canonicalize_dim(dim)
    if not (0 <= canonical_dim < _RANK):
        raise ValueError(f"reduce_scatter: dim={dim} out of range for a rank-4 tensor")

    if input_tensor.is_sharded():
        raise ValueError("reduce_scatter: sharded input not yet supported (interleaved only)")

    # Axis gate (registry model). dim uses the canonical POSITIVE convention.
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

    # --- Axis-value-dependent structural checks (all axes are in SUPPORTED here) ---
    if shape[2] % _TILE != 0 or shape[3] % _TILE != 0:
        raise ValueError(
            f"reduce_scatter: shard H and W must be tile-aligned (multiples of {_TILE}); got H={shape[2]}, W={shape[3]}"
        )

    # Slice divisibility AND tile alignment on the scatter dim: every device's
    # output slice must be a whole number of tiles. Reject loudly — never pad.
    if shape[canonical_dim] % (num_devices * _TILE) != 0:
        raise ValueError(
            f"reduce_scatter: shape[{canonical_dim}]={shape[canonical_dim]} must be divisible by "
            f"num_devices*{_TILE} = {num_devices * _TILE} so each device's slice is tile-aligned"
        )

    # Load-bearing: the Phase-A fabric writer sends align(page_size, l1_alignment)
    # bytes per page. The gather_buffer TensorAccessor spaces pages by the raw
    # page_size, so a non-16-aligned page_size would make the on-wire (rounded-up)
    # payload overrun into the next page. TILE pages are already 16-aligned; keep
    # the guard explicit, mirroring all_gather / all_reduce.
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"reduce_scatter: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        expected_shape = list(shape)
        expected_shape[canonical_dim] //= num_devices
        if (
            list(output_tensor.shape) != expected_shape
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError(
                "reduce_scatter: output_tensor spec must equal the derived output spec "
                f"(shape {expected_shape}, input dtype/layout/buffer_type)"
            )

    return num_devices, canonical_dim


def reduce_scatter(
    input_tensor: ttnn.Tensor,
    dim: int = 3,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Sum every device's shard element-wise across the line, then scatter: device
    i's output is slice i (of N equal slices along ``dim``) of the sum.

    Args:
        input_tensor: sharded across a MeshDevice line; each device holds one
            SAME-shape shard (distinct values). TILE_LAYOUT, interleaved.
        dim: scatter dimension (3 or 2; negative aliases -1/-2 accepted).
        topology: Linear (line relay) or Ring (short-way relay over the wrap
            link — Refinement 1). Output is identical; only the Phase-A
            communication pattern differs.
        output_tensor: optional pre-allocated output (shape = shard with
            ``[dim] / N``); written into and returned when supplied.
    """
    num_devices, canonical_dim = validate(input_tensor, dim=dim, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()
    shard_shape = list(input_tensor.shape)

    # Output shard: input shard shape with shape[dim] / N. Every output page is
    # written — no seeding required.
    if output_tensor is None:
        out_shape = list(shard_shape)
        out_shape[canonical_dim] //= num_devices
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    # Op-internal gather_buffer: N shard-blocks stacked on dim 0. Mesh-allocated
    # interleaved => uniform buffer address across devices, which is what lets the
    # Phase-A fabric write_page target a neighbour's block via the LOCAL accessor
    # base address routed one hop (§R3).
    gb_shape = [shard_shape[0] * num_devices, *shard_shape[1:]]
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

    # --- Phase B: local N-way slice-tile sum (compute) ---
    # Queue order IS the phase barrier (§R2): both dispatches share the CQ.
    reduce_mpd = create_reduce_mesh_program_descriptor(
        gather_buffer, output_tensor, num_devices, canonical_dim, shard_shape
    )
    ttnn.generic_op([gather_buffer, output_tensor], reduce_mpd)

    return output_tensor
