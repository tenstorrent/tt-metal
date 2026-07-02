# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — self-contained Python CCL op (generic_op + MeshProgramDescriptor).

Copies one mesh device's interleaved shard of a mesh-sharded tensor to another
device over the Tenstorrent fabric. It is PURE data movement (identity byte copy,
no arithmetic): AFTER the op the receiver device's output shard equals the sender
device's input shard bit-for-bit, and every other device's output shard equals its
own input shard (unchanged).

Newly authored fabric dataflow kernels under ``kernels/`` are assembled by a
``ttnn.generic_op`` over a ``ttnn.MeshProgramDescriptor`` holding a SEND program on
the sender coord and a RECEIVE program on the receiver coord. This op does NOT wrap,
import, or dispatch to the bound C++ ``ttnn.point_to_point`` — the C++ op and the
``all_gather_async`` kernels were read as a correctness reference only.

Because only the receiver device runs a writing program, the op first seeds the
output tensor = input on EVERY device (``ttnn.clone`` / ``ttnn.copy``), then the
fabric program overwrites only the receiver's shard.

Verified on the 8-chip Blackhole ``(2, 4)`` sim mesh with ``FABRIC_1D`` (the
mandated acceptance topology); adjacent ``(0,0) -> (0,1)`` transfer, Linear + Ring.
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


from .point_to_point_program_descriptor import create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# Registry-model declarations
# ---------------------------------------------------------------------------
# point_to_point is pure byte movement (never tilizes/untilizes), so it is
# format-agnostic in principle. `alignment` IS a shape-derived axis tagged from the
# per-device shard's last two dims: the op copies the physical pages (padded tiles
# for TILE, last-dim sticks for ROW_MAJOR) verbatim, so non-tile-aligned shards
# transfer just like aligned ones.


def tag_alignment(inputs, axes):
    """Both of the per-device shard's last two dims divisible by 32 -> tile_aligned."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS: dict = {"alignment": tag_alignment}

SUPPORTED = {
    # Pure data movement: every fixed-width dtype is correct in principle. The proven
    # primary set is the acceptance-test dtypes (bf16 / f32 / bf8b). Integer dtypes in
    # the golden TARGET are refinement candidates.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    # Both topologies are supported; Ring routes the short way (identical to Linear
    # for the adjacent pair used in the acceptance test).
    "topology": [_Topology.Linear, _Topology.Ring],
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


def _check_in_mesh(coord, mesh_shape, name):
    c = tuple(coord)
    if len(c) != len(mesh_shape) or any(not (0 <= c[i] < mesh_shape[i]) for i in range(len(c))):
        raise ValueError(f"point_to_point: {name} {c} outside mesh bounds {mesh_shape}")


def _allocate_intermediate(input_tensor, mesh_device):
    """Allocate the (replicated) fabric landing buffer.

    Shape ``(total_packets, packet_size_bytes // element_size)`` — one page per fabric
    packet, sized from ``ccl_packet_dims``. The buffer is only ever addressed as
    ``total_packets`` packet-size pages via the kernels' page-size override, so its
    dtype/layout is internal raw-byte staging detail, independent of the input.

    For most dtypes we mirror the input dtype/layout. Block-float formats (bfloat8_b)
    have no per-element byte size (``element_size()`` raises) and no ROW_MAJOR
    representation, so the landing buffer uses a raw-byte proxy (uint8 ROW_MAJOR); the
    bytes copied through it are format-agnostic, so the receiver still reconstructs the
    sender's bfloat8_b tile bit-for-bit.
    """
    l1_alignment = ttnn.get_l1_alignment()
    page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()
    pd = ttnn._ttnn.fabric.ccl_packet_dims(input_tensor.dtype, page_size, num_pages, l1_alignment)

    if input_tensor.dtype == ttnn.bfloat8_b:
        inter_dtype = ttnn.uint8  # 1 byte/element raw-byte proxy
        inter_layout = ttnn.ROW_MAJOR_LAYOUT
        element_size = 1
    else:
        inter_dtype = input_tensor.dtype
        inter_layout = input_tensor.layout
        element_size = input_tensor.element_size()

    packet_page_dim = pd.packet_size_bytes // element_size
    inter_shape = [pd.total_packets, packet_page_dim]
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(inter_shape),
        inter_dtype,
        inter_layout,
        mesh_device,
        input_tensor.memory_config(),
    )


def validate(input_tensor, sender_coord, receiver_coord, *, topology, output_tensor=None):
    """Runtime gate. Structural errors raise ValueError; axis refusals raise the
    registry-model UnsupportedAxisValue / ExcludedCell."""
    device = input_tensor.device()
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError("point_to_point: input_tensor must be on a MeshDevice")

    mesh_shape = tuple(device.shape)
    if sender_coord == receiver_coord:
        raise ValueError("point_to_point: sender_coord and receiver_coord must differ (cannot send to self)")
    _check_in_mesh(sender_coord, mesh_shape, "sender_coord")
    _check_in_mesh(receiver_coord, mesh_shape, "receiver_coord")

    if input_tensor.is_sharded():
        raise ValueError("point_to_point: sharded (non-interleaved) input not supported")

    # Load-bearing: the fabric writer sends align(page_size, l1_alignment) bytes per page
    # (ccl_helpers_dataflow.inl:34). Requiring 16B-aligned pages makes that round-up a no-op
    # so the on-wire payload never overruns the next intermediate page.
    page = input_tensor.buffer_page_size()
    if page % 16 != 0:
        raise ValueError(f"point_to_point: per-shard page size ({page} B) must be 16-byte aligned")

    if output_tensor is not None:
        if (
            list(output_tensor.shape) != list(input_tensor.shape)
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != input_tensor.layout
            or output_tensor.memory_config().buffer_type != input_tensor.memory_config().buffer_type
        ):
            raise ValueError("point_to_point: output_tensor spec must equal the input spec")

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
            raise UnsupportedAxisValue(f"point_to_point: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"point_to_point: unsupported combination (refinement candidate): {exc}")


def point_to_point(
    input_tensor: ttnn.Tensor,
    sender_coord,
    receiver_coord,
    *,
    topology: ttnn.Topology = _Topology.Linear,
    output_tensor: ttnn.Tensor = None,
    intermediate_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Copy the sender device's shard to the receiver device over the fabric.

    Returns a tensor whose receiver-device shard equals the sender-device input
    shard; every other device's shard is unchanged.
    """
    validate(input_tensor, sender_coord, receiver_coord, topology=topology, output_tensor=output_tensor)

    mesh_device = input_tensor.device()

    # Seed output = input on EVERY device: only the receiver runs a writing program, so
    # the sender + non-participating devices must retain their own input shard (identity).
    if output_tensor is None:
        output_tensor = ttnn.clone(input_tensor)
    else:
        ttnn.copy(input_tensor, output_tensor)

    if intermediate_tensor is None:
        intermediate_tensor = _allocate_intermediate(input_tensor, mesh_device)

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
    # Park the semaphore so the framework keeps its L1 alive across program-cache hits.
    mesh_program_descriptor.semaphores = [sem]

    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)
    return output_tensor
