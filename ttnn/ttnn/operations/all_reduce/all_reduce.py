# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — self-contained Python CCL + compute op (generic_op + MeshProgramDescriptor).

Sums each device's shard element-wise across all N devices of a 1-D MeshDevice
line and leaves the IDENTICAL sum on every device::

    output[d][i] = sum_{k=0..N-1} input[k][i]      for every device d

Unlike the pure-movement CCLs (point_to_point identity, all_gather concat) the
element values CHANGE, so this op combines fabric dataflow (cross-device
movement) with a compute (TRISC) reduction.

Algorithm — **broadcast-all then local N-way sum** (op_design.md "Dataflow
Strategy"). Every device chip-level-MULTICASTs its own shard to every peer on the
line, driving BOTH fabric directions from ONE worker core via the CCL dataflow
helper's DUPLEX tier; the payload lands in slot ``sender_id`` of an op-internal
gathered buffer, and the LAST packet of each direction is a fused write+atomic-inc
so a peer's arrival costs no extra packet. Each device then locally sums its own
shard (read straight from ``input_tensor``) plus the N-1 received slots with a
single streaming compute pass (pairwise ``add_tiles`` folded in one DEST register).

This op does NOT import, wrap, re-export or dispatch to any existing
all_reduce / reduce_scatter / all_gather.

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


from .all_reduce_program_descriptor import create_mesh_program_descriptor, line_direction_slots


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# `alignment` is shape-derived from the per-device shard's last two dims. It is
# load-bearing (not cosmetic): the gathered landing buffer scales dim 0 by N, and
# "slot k == pages [k*P, (k+1)*P)" only survives that scaling when each shard
# occupies WHOLE tile-rows. See op_design.md Risk 13.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # The reduction is an FPU tile add; both float dtypes are handled, with
    # fp32_dest_acc_en tracking the dtype so the accumulator is never narrower
    # than the operands (op_design.md Risk 9).
    "dtype": [ttnn.bfloat16, ttnn.float32],
    # TILE only: the reduction is a tile compute, and a page IS a tile end to end
    # (no tilize/untilize anywhere in the pipeline).
    "layout": [ttnn.TILE_LAYOUT],
    # Ring needs the alternating target-count math plus an explicit range_hops==0
    # guard (op_design.md Risk 3); Linear is the proven primary topology.
    "topology": [_Topology.Linear],
    "alignment": ["tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS: list = []


# ---------------------------------------------------------------------------
# Op-internal cross-device GlobalSemaphore: created ONCE per mesh_device (+ one
# synchronize_device), reused across program-cache hits, never recreated.
#
# The cache is stored as an attribute ON the mesh_device object rather than in a
# module-level `{id(mesh_device): sem}` dict: a MeshDevice does not support weak
# references (`MeshDevice.__weakrefoffset__ == 0`), so an id-keyed module dict
# outlives the device it was created for, and CPython freely re-uses the freed
# address for the NEXT MeshDevice — which would hand a fresh device a
# GlobalSemaphore that belongs to a closed one. Binding the lifetime to the
# device object (MeshDevice carries a Python __dict__ — the root conftest already
# attaches `cache_entries_counter` the same way) makes that class of stale reuse
# impossible and releases the semaphore with the device.
# ---------------------------------------------------------------------------
_SEM_ATTR = "_ttnn_all_reduce_recv_semaphore"


def _get_or_create_semaphore(mesh_device):
    sem = getattr(mesh_device, _SEM_ATTR, None)
    if sem is None:
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        worker_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        ttnn.synchronize_device(mesh_device)
        setattr(mesh_device, _SEM_ATTR, sem)
    return sem


def _gathered_shape(shard_shape, num_devices):
    """Landing-buffer shape: dim 0 scaled by N so slot k == pages [k*P, (k+1)*P).

    For a TILE interleaved tensor the page order is row-major over (..., Ht, Wt),
    so prepending N independent copies of the trailing block makes device k's
    shard the contiguous page run [k*P, (k+1)*P). Holds for rank >= 3 (dim 0 is a
    batch dim) and for rank 2 when H % 32 == 0.
    """
    out = list(shard_shape)
    out[0] = out[0] * num_devices
    return out


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, topology, output_tensor):
    """Runtime gate.

    Structural input errors raise ``ValueError``; axis refusals raise the
    registry-model ``UnsupportedAxisValue`` / ``ExcludedCell``.

    Returns ``(num_devices, direction_slots)`` — the per-device fabric
    FORWARD/BACKWARD neighbour slotting, computed here (it is also validated
    here: the two neighbours must not report the same fabric direction) and
    handed to the program descriptor so the routes are derived exactly once.
    """
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

    rank = len(input_tensor.shape)
    if rank < 2:
        raise ValueError(f"all_reduce: input rank must be >= 2, got {rank}")

    # Axis gate (registry model). It runs BEFORE the dtype/page-size-dependent
    # framing gates below: those would otherwise be able to reject an
    # out-of-SUPPORTED dtype with a ValueError, and the golden harness expects a
    # NotImplementedError subclass for every out-of-SUPPORTED cell (a ValueError
    # there shows up as xfail_wrong_mode). Placement/shape checks stay above —
    # they are independent of the axis universe, and `rank >= 2` is a
    # precondition of the shape-derived tagger.
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

    page_size = input_tensor.buffer_page_size()
    l1_alignment = ttnn.get_l1_alignment()
    if page_size % l1_alignment != 0:
        raise ValueError(
            f"all_reduce: per-shard page size ({page_size} B) must be a multiple of the "
            f"L1 alignment ({l1_alignment} B) — the fabric sends align(page_size, 1) bytes "
            f"into a page-spaced destination"
        )

    # The op deliberately sends ONE tile page per fabric packet and never
    # coalesces or segments, so a page that does not fit in a single packet would
    # silently corrupt. Gate on it (op_design.md Risk 14).
    dims = ttnn._ttnn.fabric.ccl_packet_dims(
        input_tensor.dtype, page_size, input_tensor.buffer_num_pages(), l1_alignment
    )
    if dims.page_segments != 1:
        raise ValueError(
            f"all_reduce: one tile page ({page_size} B) must fit in one fabric packet "
            f"(got page_segments={dims.page_segments})"
        )

    if output_tensor is not None:
        if (
            list(output_tensor.shape) != list(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("all_reduce: output_tensor spec must equal the input shard spec exactly")

    # Fabric direction slotting. `is_forward` is NOT "toward increasing index" —
    # ccl_dm_route owns a deliberate sign reversal — so never assume it; query it,
    # and assert the two neighbours land in DIFFERENT slots (op_design.md Risk 4).
    direction_slots = [line_direction_slots(device, i, num_devices, topology) for i in range(num_devices)]

    return num_devices, direction_slots


def all_reduce(
    input_tensor: ttnn.Tensor,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Element-wise SUM of every device's shard, left identically on every device.

    Args:
        input_tensor: shard on a ``(1, N)`` MeshDevice line; every device holds a
            shard of the SAME shape. TILE_LAYOUT, interleaved.
        topology: fabric topology (``Linear``).
        output_tensor: optional pre-allocated output with the input shard's spec;
            written in place and returned.

    Returns:
        A tensor with the input shard's shape whose per-device value is the
        element-wise sum of all N shards (identical on every device).
    """
    num_devices, direction_slots = validate(input_tensor, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()

    # The op overwrites every output page, so no clone / zero-seed is needed.
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            input_tensor.dtype,
            input_tensor.layout,
            mesh_device,
            input_tensor.memory_config(),
        )

    # Op-internal landing buffer for the fabric multicast. ONE mesh allocation, so
    # it sits at the IDENTICAL address on every device — which is what makes a
    # noc0-encoded destination resolve to the right DRAM bank on every receiving
    # chip (op_design.md Risk 7). Slot my_id is written by nobody and read by
    # nobody: the local contribution comes straight from input_tensor (Risk 8).
    gathered_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_gathered_shape(list(input_tensor.shape), num_devices)),
        input_tensor.dtype,
        input_tensor.layout,
        mesh_device,
        input_tensor.memory_config(),
    )

    sem = _get_or_create_semaphore(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(sem)

    mesh_program_descriptor = create_mesh_program_descriptor(
        input_tensor, gathered_tensor, output_tensor, topology, sem_addr, direction_slots
    )
    # Park the semaphore so the framework keeps its L1 alive across cache hits.
    # Do NOT add a post-dispatch synchronize_device for that purpose.
    mesh_program_descriptor.semaphores = [sem]

    ttnn.generic_op([input_tensor, gathered_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
