# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
point_to_point — self-contained Python CCL data-movement op.

Copies one mesh device's shard of an interleaved tensor to another device over the
Tenstorrent fabric. It performs NO arithmetic: after the call the receiver device's shard
is bit-identical to the sender device's input shard, and every non-participating device's
shard is untouched.

Built on ``ttnn.generic_op`` + ``ttnn.MeshProgramDescriptor`` with newly-authored
sender/receiver dataflow kernels under ``kernels/dataflow/``. It does NOT wrap or dispatch
to the bound C++ op ``ttnn.point_to_point`` — that op is a correctness reference only.

The registry-style declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS) mirror the golden
feature_spec axes (dtype / layout / topology / alignment). Because the transfer is a pure
page copy, every dtype/layout/topology/alignment combination is supported; INVALID
(bfloat8_b + ROW_MAJOR) lives in the golden feature_spec and is skipped before reaching the
op. validate() also enforces the structural fabric/mesh preconditions from op_design.md.
"""

from __future__ import annotations

import ttnn

from .point_to_point_program_descriptor import create_mesh_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — shape-derived categorical axes
# ---------------------------------------------------------------------------
def _tag_alignment(inputs, axes=None):
    """Last two dims tile-aligned (32-multiple)."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS = {
    "alignment": _tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — per-axis accepted values
# ---------------------------------------------------------------------------
# Built lazily: this module is imported eagerly during ttnn init (before
# ttnn.Topology is registered), so the ttnn.* enum references cannot be evaluated
# at module-import time. `SUPPORTED` is exposed via module __getattr__ below.
def _supported():
    return {
        "dtype": [
            ttnn.bfloat16,
            ttnn.float32,
            ttnn.bfloat8_b,
            ttnn.uint16,
            ttnn.int32,
            ttnn.uint32,
        ],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "alignment": ["tile_aligned", "non_tile_aligned"],
    }


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells refused for now (none: pure byte-copy supports all of SUPPORTED)
# ---------------------------------------------------------------------------
EXCLUSIONS = []


def __getattr__(name):
    # Lazy module attribute so `point_to_point.SUPPORTED` resolves the registry axes
    # without evaluating ttnn enums at import time (circular-import safe).
    if name == "SUPPORTED":
        return _supported()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 4. validate() — runtime gate (registry axes + structural fabric/mesh rules)
# ---------------------------------------------------------------------------
def validate(input_tensor, sender_coord, receiver_coord, topology, output_tensor=None, intermediate_tensor=None):
    # --- structural rules (op_design.md "Validation Rules") ---
    # Rule 8: rank >= 2.
    if len(input_tensor.shape) < 2:
        raise ValueError("point_to_point: input rank must be >= 2")

    # Rule 1: input must live on a MeshDevice.
    mesh_device = input_tensor.device()
    if mesh_device is None or not hasattr(mesh_device, "shape"):
        raise ValueError("point_to_point: input must be on a MeshDevice")

    # Rule 2: cannot send to self.
    if sender_coord == receiver_coord:
        raise ValueError("point_to_point: cannot send to self (sender_coord == receiver_coord)")

    # Rule 5: sharded (non-interleaved) memory layout rejected.
    if input_tensor.is_sharded():
        raise ValueError("point_to_point: sharded configs not yet supported")

    # Rule 6: page size must be 16-byte (L1) aligned.
    l1_alignment = ttnn.get_l1_alignment()
    page_size_bytes = input_tensor.buffer_page_size()
    if not (page_size_bytes % l1_alignment == 0 or page_size_bytes == l1_alignment):
        raise ValueError(f"point_to_point: page size {page_size_bytes} must be {l1_alignment}-byte aligned")

    # Rules 3 & 4: coordinates inside the mesh and 1-D routable (share a row or column).
    mesh_shape = tuple(mesh_device.shape)
    for c in (sender_coord, receiver_coord):
        for d in range(len(mesh_shape)):
            if not (0 <= c[d] < mesh_shape[d]):
                raise ValueError(
                    f"point_to_point: coordinate {tuple(c[i] for i in range(len(mesh_shape)))} outside mesh {mesh_shape}"
                )
    if not (sender_coord[0] == receiver_coord[0] or sender_coord[1] == receiver_coord[1]):
        raise ValueError("point_to_point: sender/receiver must share a row or column (1-D fabric routable)")

    # Rule 7: supplied output_tensor spec must equal the resolved output spec (== input spec).
    # TensorSpec has no __eq__ binding, so compare the spec fields component-wise.
    if output_tensor is not None:
        in_spec, out_spec = input_tensor.spec, output_tensor.spec
        in_mc, out_mc = in_spec.memory_config, out_spec.memory_config
        spec_matches = (
            tuple(out_spec.shape) == tuple(in_spec.shape)
            and out_spec.dtype == in_spec.dtype
            and out_spec.layout == in_spec.layout
            and out_mc.buffer_type == in_mc.buffer_type
            and out_mc.memory_layout == in_mc.memory_layout
        )
        if not spec_matches:
            raise ValueError("point_to_point: output_tensor spec must equal resolved output spec")

    # --- registry axis checks ---
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "topology": topology,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in _supported().items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"point_to_point: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"point_to_point: unsupported combination (refinement candidate): {exc}")


def _intermediate_spec(input_tensor, packet_dims):
    """Resolved intermediate (fabric landing buffer) spec: 2-D, same dtype/layout/memory as input.

    Shape is (total_packets, packet_page_dim). packet_page_dim is the per-packet element
    count. For block-quantized bfloat8_b (TILE-only), tt::datum_size throws, so derive the
    element count from the tile (1024 logical elements per 32x32 tile).
    """
    dtype = input_tensor.dtype
    layout = input_tensor.layout
    packet_size_bytes = packet_dims.packet_size_bytes

    if dtype == ttnn.bfloat8_b:
        tile_bytes = ttnn.tile_size(dtype)
        packet_page_dim = (packet_size_bytes // tile_bytes) * (32 * 32)
    else:
        packet_page_dim = packet_size_bytes // ttnn.element_size(dtype)

    intermediate_shape = ttnn.Shape([packet_dims.total_packets, packet_page_dim])
    buffer_type = input_tensor.memory_config().buffer_type
    return ttnn.TensorSpec(intermediate_shape, dtype, layout, buffer_type)


def point_to_point(
    input_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology = None,
    output_tensor: ttnn.Tensor = None,
    intermediate_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Send ``input_tensor``'s shard at ``sender_coord`` to ``receiver_coord`` over the fabric.

    Returns the output tensor whose receiver-device shard is bit-identical to the sender's
    input shard; all other devices' shards are preserved (when output_tensor is supplied) or
    uninitialized (fresh allocation).
    """
    if topology is None:
        topology = ttnn.Topology.Linear

    validate(input_tensor, sender_coord, receiver_coord, topology, output_tensor, intermediate_tensor)

    mesh_device = input_tensor.device()
    l1_alignment = ttnn.get_l1_alignment()

    input_page_size_bytes = input_tensor.buffer_page_size()
    input_num_pages = input_tensor.buffer_num_pages()

    # Fabric packet framing — host and kernels agree on the exact framing via this helper.
    packet_dims = ttnn._ttnn.fabric.ccl_packet_dims(
        input_tensor.dtype, input_page_size_bytes, input_num_pages, l1_alignment
    )

    # Allocate the transient fabric landing buffer mesh-wide (same address on every device).
    if intermediate_tensor is None:
        intermediate_tensor = ttnn.allocate_tensor_on_device(_intermediate_spec(input_tensor, packet_dims), mesh_device)

    # Allocate the output (per-device shard shape == input shard shape) if not supplied.
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    # Fresh GlobalSemaphore each call (over the full worker grid so its address is valid on
    # the worker core). Kept alive until after the post-dispatch barrier below.
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 0)
    ttnn.synchronize_device(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(semaphore)

    mesh_program_descriptor = create_mesh_program_descriptor(
        mesh_device,
        input_tensor,
        intermediate_tensor,
        output_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
        packet_dims,
    )

    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)

    # Barrier BEFORE returning so the device finishes consuming the semaphore before the
    # GlobalSemaphore object (and its L1 allocation) is freed at function return.
    ttnn.synchronize_device(mesh_device)

    return output_tensor
