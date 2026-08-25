# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — self-contained Python compute-CCL op (generic_op +
MeshProgramDescriptor), ONE dispatch per invocation.

Element-wise SUM of all N devices' same-shape shards on a 1-D MeshDevice line;
EVERY device ends up holding the identical full sum:

    output[...] = Σ_{c=0}^{N-1} shard_c[...]        (identical on every device)
    output shape/dtype/layout == a single input shard

Algorithm — line store-and-forward GATHER of whole shards fused, in the SAME
program, with an ARRIVAL-ORDERED incremental reduce over the FULL shard
(op_design.md "Dataflow Strategy"): every device receives all N-1 remote shards
into a local gather_buffer; a dedicated reduce core consumes contributions one at a
time — own shard first, then each arrival the moment its counting semaphore lands —
so the accumulate of contribution k overlaps the fabric flight of contribution k+1.
After the last accumulate, a helper copy drains the resident sum to the output
writer. There is no scatter (output = full shard on every device) and no scaling
(SUM). This op does NOT wrap, import, call, or dispatch to any existing CCL op.

Primary proven case: bfloat16, TILE_LAYOUT, Linear topology, on a Blackhole
``(1, 4)`` line mesh with ``FABRIC_1D``.
"""

from __future__ import annotations

import ttnn

# Topology lives on the C++ module; the top-level ttnn.Topology alias only binds
# AFTER ttnn.operations is auto-imported, so reference the source module directly (R16).
from ttnn._ttnn.operations.ccl import Topology as _Topology

try:  # registry-model refusal types; fall back when the shared module is absent.
    from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
except ImportError:  # pragma: no cover

    class UnsupportedAxisValue(NotImplementedError):
        pass

    class ExcludedCell(NotImplementedError):
        pass


from .all_reduce_program_descriptor import create_mesh_program_descriptor

_RANK = 4  # rank pinned to 4
_TILE = 32
# L1 growth cliff: cb_accumulator holds the whole P-page shard resident on the
# reduce core (op_design.md "Circular Buffers"). Larger-P spill is beyond-TARGET.
_MAX_ACCUMULATOR_BYTES = 512 * 1024


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# No shape-derived axis: every golden INPUT is tile-aligned by construction, so
# INPUT_TAGGERS is empty. There is no `dim` axis — all_reduce has no
# scatter/gather dimension.

INPUT_TAGGERS: dict = {}

SUPPORTED = {
    # bf16 is the proven primary dtype (PCC 0.99 — a sum of N terms accumulates
    # rounding); float32 the higher-precision secondary (fp32_dest_acc_en in the
    # reduce compute covers both).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # The reduction is a tile compute — TILE_LAYOUT only.
    "layout": [ttnn.TILE_LAYOUT],
    # Linear line relay is Phase-0. The kernels' block indices are already
    # ring-modular (T3), so Ring is a beyond-TARGET host-table change.
    "topology": [_Topology.Linear],
}

EXCLUSIONS: list = []


# Module-level GlobalSemaphore cache: the (sem_fwd, sem_bwd) pair is created ONCE
# per mesh_device (+ one synchronize_device inside the miss branch), reused across
# program-cache hits, never recreated.
_SEMAPHORE_CACHE: dict = {}


def _get_or_create_semaphores(mesh_device):
    key = id(mesh_device)
    sems = _SEMAPHORE_CACHE.get(key)
    if sems is None:
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        worker_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        sem_fwd = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        sem_bwd = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        ttnn.synchronize_device(mesh_device)  # ONE cache-miss barrier for both
        sems = (sem_fwd, sem_bwd)
        _SEMAPHORE_CACHE[key] = sems
    return sems


def validate(input_tensor, *, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell.

    Ordering (op_design.md "Validation & Registry Contract"): universal structural
    checks (needed to even form the axes dict) raise ValueError first; then the
    AXIS GATE (typed refusals); then the axis-value-DEPENDENT structural checks
    (fabric-payload gate, accumulator L1 budget, output spec) — so an
    out-of-SUPPORTED axis value always yields the typed refusal, never a
    shape-derived ValueError computed under the wrong axis.

    Returns ``num_devices``.
    """
    # --- Universal structural checks ---
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("all_reduce: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"all_reduce: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")

    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("all_reduce: requires at least 2 mesh devices on the line")

    shape = list(input_tensor.shape)
    if len(shape) != _RANK:
        raise ValueError(f"all_reduce: expected a rank-4 input shard, got rank {len(shape)}")

    if input_tensor.is_sharded():
        raise ValueError("all_reduce: sharded input not supported (interleaved only)")

    if shape[2] % _TILE != 0 or shape[3] % _TILE != 0:
        raise ValueError(
            f"all_reduce: shard H and W must be tile-aligned (multiples of {_TILE}); " f"got H={shape[2]}, W={shape[3]}"
        )

    # --- Axis gate (registry model) ---
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

    # --- Axis-value-dependent structural checks (all axes are in SUPPORTED here) ---
    # Load-bearing: the fabric writer sends align(page_size, l1_alignment) bytes per
    # page while the gather_buffer accessor spaces pages by the raw page_size; a
    # non-16-aligned page would overrun into the next page. TILE pages are already
    # aligned; keep the guard explicit (reference precedent).
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"all_reduce: per-shard page size ({page} B) must be 16-byte aligned")

    # L1 growth cliff: the reduce core keeps the whole P-tile running sum resident.
    P = input_tensor.buffer_num_pages()
    if P * page > _MAX_ACCUMULATOR_BYTES:
        raise ValueError(
            f"all_reduce: shard of {P} tiles x {page} B exceeds the resident accumulator "
            f"budget ({_MAX_ACCUMULATOR_BYTES} B); larger shards are a refinement candidate"
        )

    if output_tensor is not None:
        if (
            list(output_tensor.shape) != shape
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError(
                "all_reduce: output_tensor spec must equal the input shard spec "
                f"(shape {shape}, input dtype/layout/buffer_type)"
            )

    return num_devices


def all_reduce(
    input_tensor: ttnn.Tensor,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Element-wise SUM of every device's shard across the line; every device's
    output holds the identical full sum (same shape/dtype/layout as one shard).

    Args:
        input_tensor: sharded across a MeshDevice line; each device holds one
            SAME-shape shard (distinct values). TILE_LAYOUT, interleaved.
        topology: Linear (line relay) — Phase-0.
        output_tensor: optional pre-allocated output (same spec as one input
            shard); written into and the SAME handle returned when supplied.
    """
    num_devices = validate(input_tensor, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()
    shard_shape = list(input_tensor.shape)

    # Output shard: same spec as one input shard. Every output page is written —
    # no seeding required.
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shard_shape),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    # Op-internal gather_buffer: N shard-blocks stacked on dim 0, allocated FRESH per
    # call (R14) and passed in io_tensors so dispatch resolves/keeps it alive.
    # Mesh-allocated interleaved => uniform buffer address across devices, which is
    # what lets a fabric write_page target the neighbour's block through the LOCAL
    # accessor routed one hop (T3). Block my_chip_id is never written (the reduce
    # reader takes the own contribution directly from the input tensor).
    gb_shape = [shard_shape[0] * num_devices, *shard_shape[1:]]
    gather_buffer = ttnn.allocate_tensor_on_device(
        ttnn.Shape(gb_shape),
        input_tensor.dtype,
        input_tensor.layout,
        mesh_device,
        input_tensor.memory_config(),
    )

    # Cross-device sync: two op-internal counting GlobalSemaphores (one per
    # direction), created once per mesh_device and parked on the descriptor so the
    # framework keeps their L1 alive across program-cache hits. No per-call
    # post-dispatch barrier.
    sem_fwd, sem_bwd = _get_or_create_semaphores(mesh_device)
    sem_fwd_addr = ttnn.get_global_semaphore_address(sem_fwd)
    sem_bwd_addr = ttnn.get_global_semaphore_address(sem_bwd)

    mesh_pd = create_mesh_program_descriptor(
        input_tensor,
        gather_buffer,
        output_tensor,
        num_devices,
        topology,
        sem_fwd_addr,
        sem_bwd_addr,
    )
    if hasattr(mesh_pd, "semaphores"):
        mesh_pd.semaphores = [sem_fwd, sem_bwd]

    # ONE dispatch: gather and arrival-overlapped full-shard reduce in a single
    # program per device. Output preallocated and LAST.
    ttnn.generic_op([input_tensor, gather_buffer, output_tensor], mesh_pd)

    return output_tensor
