# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter_average — self-contained Python compute-CCL op (generic_op +
MeshProgramDescriptor), ONE dispatch per invocation.

Element-wise MEAN of all N devices' same-shape shards on a 1-D MeshDevice line,
scattered: device i keeps only slice i (of N equal slices along ``dim``) of the
mean. The 1/N scaling is part of the op — the caller passes nothing but the tensor:

    output_i[...] = ((Σ_{c=0}^{N-1} shard_c) / N)[slice i along dim]
    output.shape[dim] = input.shape[dim] / N

Algorithm — store-and-forward GATHER of whole shards fused, in the SAME program,
with an ARRIVAL-ORDERED incremental reduce (op_design.md "Dataflow Strategy"):
every device receives all N-1 remote shards into a local gather_buffer; a
dedicated reduce core consumes contributions one at a time — own shard first,
then each arrival the moment its counting semaphore lands — so the accumulate of
contribution k overlaps the fabric flight of contribution k+1. After the last
accumulate, a 1/N broadcast-scalar multiply streams the output slice. The
topology (Linear or Ring) selects only the host-side block-flow table and
neighbour wiring — Linear relays everything down the line, Ring carries each
direction's SHORT-WAY half over the wrap link — the kernels are
topology-agnostic (ring-modular block indices). This op does NOT wrap, import,
call, or dispatch to any existing CCL op.

Primary proven case: bfloat16, TILE_LAYOUT, dim=3, Linear topology, on a Wormhole
``(1, 8)`` line mesh with ``FABRIC_1D``; Ring verified on a (1, 4) Blackhole ring
under the same FABRIC_1D config (topology selected by the kwarg alone).
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


from .reduce_scatter_average_program_descriptor import create_mesh_program_descriptor

_RANK = 4  # rank pinned to 4 (dim canonicalization is `dim % 4`)
_TILE = 32
# Conservative L1 growth cliff: cb_accumulator holds S = P/N whole pages resident on
# the reduce core (op_design.md "Circular Buffers"). Refinement 5 lifts it.
_MAX_SLICE_TILES = 256


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# No shape-derived axis: every golden INPUT is tile-aligned by construction, so
# INPUT_TAGGERS is empty. `dim` MUST be a SUPPORTED key even single-valued — the
# golden harness derives xfail marks only for declared axes.

INPUT_TAGGERS: dict = {}

SUPPORTED = {
    # bf16 is the proven primary dtype (PCC 0.99 — a sum of N terms plus the 1/N
    # scale accumulates rounding); float32 the higher-precision secondary
    # (fp32_dest_acc_en in the reduce compute covers both).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # The reduction and scaling are tile computes — TILE_LAYOUT only.
    "layout": [ttnn.TILE_LAYOUT],
    # Linear line relay is Phase-0. Ring landed in Refinement 2: the kernels'
    # block indices were already ring-modular, so Ring only swaps the host-side
    # block-flow table (each direction carries its SHORT-WAY half of the ring —
    # fwd depth N//2, bwd (N-1)//2) and wires the wrap link through
    # ccl_dm_route(.., Ring) (1-hop wrap routes, probed green under FABRIC_1D).
    "topology": [_Topology.Linear, _Topology.Ring],
    # Scatter dims, POSITIVE convention. Negative aliases are canonicalized BEFORE
    # the membership test (-1 ≡ 3, -2 ≡ 2). dim=2 landed in Refinement 1: the reduce
    # reader walks the dim=2 slice as dense per-(batch, channel) tile-row runs.
    "dim": [3, 2],
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


def _canonicalize_dim(dim: int) -> int:
    """Positive-convention canonicalization, rank pinned to 4: -1 ≡ 3, -2 ≡ 2."""
    return dim % _RANK


def validate(input_tensor, *, dim, topology, output_tensor):
    """Runtime gate. Structural input errors raise ValueError; axis refusals raise
    the registry-model UnsupportedAxisValue / ExcludedCell.

    Ordering (op_design.md "Validation & Registry Contract"): universal structural
    checks (needed to even form the axes dict) raise ValueError first; then the
    AXIS GATE (typed refusals); then the axis-value-DEPENDENT structural checks
    (slice divisibility, output spec, L1 accumulator budget) — so an
    out-of-SUPPORTED axis value always yields the typed refusal, never a
    shape-derived ValueError computed under the wrong axis.

    Returns ``(num_devices, canonical_dim)``.
    """
    # --- Universal structural checks ---
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("reduce_scatter_average: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] != 1:
        raise ValueError(f"reduce_scatter_average: expected a 1-D line mesh view (1, N), got shape {mesh_shape}")

    num_devices = mesh_shape[0] * mesh_shape[1]
    if num_devices < 2:
        raise ValueError("reduce_scatter_average: requires at least 2 mesh devices on the line")

    shape = list(input_tensor.shape)
    if len(shape) != _RANK:
        raise ValueError(f"reduce_scatter_average: expected a rank-4 input shard, got rank {len(shape)}")

    if not (-_RANK <= dim < _RANK):
        raise ValueError(f"reduce_scatter_average: dim={dim} out of range for a rank-4 tensor")
    canonical_dim = _canonicalize_dim(dim)

    if input_tensor.is_sharded():
        raise ValueError("reduce_scatter_average: sharded input not supported (interleaved only)")

    if shape[2] % _TILE != 0 or shape[3] % _TILE != 0:
        raise ValueError(
            f"reduce_scatter_average: shard H and W must be tile-aligned (multiples of {_TILE}); "
            f"got H={shape[2]}, W={shape[3]}"
        )

    # --- Axis gate (registry model). dim uses the canonical POSITIVE convention. ---
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
            raise UnsupportedAxisValue(f"reduce_scatter_average: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"reduce_scatter_average: unsupported combination (refinement candidate): {exc}")

    # --- Axis-value-dependent structural checks (all axes are in SUPPORTED here) ---
    # Scatter constraint: shape[dim] divisible by N AND the per-device slice
    # tile-aligned. Reject loudly — never pad silently.
    if shape[canonical_dim] % (num_devices * _TILE) != 0:
        raise ValueError(
            f"reduce_scatter_average: shape[{canonical_dim}]={shape[canonical_dim]} must be divisible by "
            f"num_devices*{_TILE} = {num_devices * _TILE} so each device's slice is tile-aligned"
        )

    # Load-bearing: the fabric writer sends align(page_size, l1_alignment) bytes per
    # page while the gather_buffer accessor spaces pages by the raw page_size; a
    # non-16-aligned page would overrun into the next page. TILE pages are already
    # aligned; keep the guard explicit (reference precedent).
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"reduce_scatter_average: per-shard page size ({page} B) must be 16-byte aligned")

    # L1 growth cliff: the reduce core keeps the whole S-tile running sum resident.
    slice_tiles = (shape[0] * shape[1] * (shape[2] // _TILE) * (shape[3] // _TILE)) // num_devices
    if slice_tiles > _MAX_SLICE_TILES:
        raise ValueError(
            f"reduce_scatter_average: per-device slice of {slice_tiles} tiles exceeds the resident "
            f"accumulator budget ({_MAX_SLICE_TILES} tiles); larger shards are a refinement candidate"
        )

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
                "reduce_scatter_average: output_tensor spec must equal the derived output spec "
                f"(shape {expected_shape}, input dtype/layout/buffer_type)"
            )

    return num_devices, canonical_dim


def reduce_scatter_average(
    input_tensor: ttnn.Tensor,
    dim: int = 3,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Element-wise mean of every device's shard across the line, scattered: device
    i's output is slice i (of N equal slices along ``dim``) of the mean.

    Args:
        input_tensor: sharded across a MeshDevice line; each device holds one
            SAME-shape shard (distinct values). TILE_LAYOUT, interleaved.
        dim: scatter dimension (3 or 2; negative aliases -1 / -2 accepted).
        topology: Linear (line relay) or Ring (short-way relay over the wrap
            link; Refinement 2).
        output_tensor: optional pre-allocated output (shape = shard with
            ``[dim] / N``); written into and the SAME handle returned when supplied.
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

    # Op-internal gather_buffer: N shard-blocks stacked on dim 0, allocated FRESH per
    # call (R14) and passed in io_tensors so dispatch resolves/keeps it alive.
    # Mesh-allocated interleaved => uniform buffer address across devices, which is
    # what lets a fabric write_page target the neighbour's block through the LOCAL
    # accessor routed one hop. Block my_chip_id is never written (the reduce reader
    # takes the own contribution directly from the input tensor).
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
        canonical_dim,
        topology,
        sem_fwd_addr,
        sem_bwd_addr,
    )
    if hasattr(mesh_pd, "semaphores"):
        mesh_pd.semaphores = [sem_fwd, sem_bwd]

    # ONE dispatch: gather, arrival-overlapped reduce, and 1/N scale in a single
    # program per device. Output preallocated and LAST.
    ttnn.generic_op([input_tensor, gather_buffer, output_tensor], mesh_pd)

    return output_tensor
