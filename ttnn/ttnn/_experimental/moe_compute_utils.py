# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Python helper utilities for ttnn.experimental.moe_compute.

This module provides reference implementations for preparing weight tensors
for the fused MoE compute operation. The functions here are "executable
specifications" that produce the exact byte layout the kernels expect.

**Weight tensor layout overview**

``ttnn.experimental.moe_compute`` takes **two packed weight tensor arguments**
that contain **three logical expert weight matrices**:

- ``matmul_w0_w1_tensor``: W0 (gate) and W1 (up) weights interleaved/packed.
- ``matmul_w2_tensor``: W2 (down projection) weights.

The layout is highly specific to the MoE ring-all-to-all kernel implementation
and is derived from the ``(hidden_size, intermediate_size)`` pair.  Per-core
tile counts are computed by ``_shard_tiles()`` (Euclidean-rhythm / Bresenham
distribution) and ``_w2_shard_tiles()`` (complementary pattern for load
balancing). These formulas must stay in sync with the constexpr equivalents
in ``moe_ring_common.h``.

**Tile-block constants (must match moe_ring_common.h)**

- ``W0_W1_BLOCK_TILES_W = 4``     — W0/W1 read block width in tiles
- ``W2_TILES_PER_A2A_ITER_W = 4`` — W2 a2a-iter width in tiles
- ``BLOCK_TILES_H = 7``           — tiles per DRAM read transaction (height)

Other tile counts (W0/W1 shard sizes, W2 groups per core, etc.) are now
computed from ``hidden_size`` and ``intermediate_size`` via the shard formulas
rather than being hardcoded per model.

**Bias support (``has_bias=True``)**

When bias is enabled, callers must pack bias values into the weight tensors
in a kernel-specific format and set ``has_bias=True`` on the ``moe_compute`` call:

- **W0/W1 bias (b0, b1)**: PyTorch format is ``(L, E, N)`` where L=layers, E=experts,
  N=intermediate dim. This is expanded to tile format ``(L, E, 32, N)`` with only
  row 0 populated, concatenated **after** W0/W1 along the K (input) dimension, then
  K is padded to a multiple of BLOCK_TILES_H (7) tiles.

- **W2 bias (b2)**: PyTorch format is ``(L, E, K)``. Expanded to tile format
  ``(L, E, 32, K)`` with row 0 populated, K-column-sharded like W2, appended along
  the N (intermediate) axis **without** ring-rotation. N+32 is then padded to a
  multiple of BLOCK_TILES_H tiles.

**Output shapes**

- ``prepare_w0_w1_tensor_for_moe_compute``: ``(num_cores, L, E, groups_per_core, K_padded, 4*TILE_SIZE)``
- ``prepare_w0_w1_tensor_with_bias``: K_padded includes the bias tile row plus padding
- ``prepare_w2_tensor_for_moe_compute``: ``(num_cores, L, E, w2_groups_per_core, N_padded, 4*TILE_SIZE)``
- ``prepare_w2_tensor_with_bias``: N_padded includes the bias tile row plus padding

The leading ``num_cores`` dimension (typically 12) corresponds to DRAM bank layout.

**DRAM sharding**

Callers must create DRAM-sharded memory configs with heights derived from the
padded dimensions above. Use ``get_weight_mem_configs(...)`` to compute the
memory configs, or see ``test_moe_compute_6U.py`` for the full flow.

**Available functions**

- Shard formulas: ``_shard_tiles``, ``_w2_shard_tiles``, ``auto_output_width_shard_dim``,
  ``effective_matmul_ring_size``
- Shard maps: ``get_weight_core_shard_maps(mesh_device, hidden_size, intermediate_size)``
- Memory configs: ``get_weight_mem_configs(...)``
- Non-bias: ``prepare_w0_w1_tensor_for_moe_compute``, ``prepare_w2_tensor_for_moe_compute``
- With bias: ``prepare_w0_w1_tensor_with_bias``, ``prepare_w2_tensor_with_bias``
- Helpers: ``cluster_distance``, ``map_shared_experts``, ``add_shared_expert_weights``

See individual function docstrings for argument details and layout invariants.
"""

from __future__ import annotations

import math
from typing import Sequence

import ttnn


def cluster_distance(d0: int, d1: int, mesh_shape: tuple[int, int], cluster_axis: int) -> int | None:
    """Calculate Manhattan distance between two devices along the cluster axis.

    Returns None if devices are not on the same cluster line, otherwise returns
    the distance along the cluster axis.
    """
    c0 = (d0 // mesh_shape[1], d0 % mesh_shape[1])
    c1 = (d1 // mesh_shape[1], d1 % mesh_shape[1])

    return None if c0[1 - cluster_axis] != c1[1 - cluster_axis] else abs(c0[cluster_axis] - c1[cluster_axis])


def map_shared_experts(
    expert_mapping_tensor: "torch.Tensor",
    shared_expert_ids_to_devices: dict[int, list[int]],
    mesh_shape: Sequence[int],
    cluster_axis: int,
) -> "torch.Tensor":
    """
    Map shared experts to their nearest on-axis device for dispatch operations.

    This function extends the expert mapping tensor to include shared experts by determining
    the optimal device assignment for each shared expert based on cluster topology. For each
    dispatching device, it selects the nearest receiving device on the same cluster axis
    that has the shared expert.

    Args:
        expert_mapping_tensor: 2D tensor of shape [devices, routed_experts] containing
            linearized mesh coordinates of the device owning each expert.
        shared_expert_ids_to_devices: Dictionary mapping shared expert IDs to lists of
            device IDs where they are replicated. Expert IDs must be contiguous
            continuations of routed expert IDs.
        mesh_shape: Tuple/list representing the dimensions of the device mesh (e.g., (4, 4)).
        cluster_axis: Axis along which devices are clustered (0 or 1). Determines the
            direction of nearest-neighbor search for shared experts.

    Returns:
        torch.Tensor: Extended mapping tensor of shape [devices, routed_experts + shared_experts]
            where each entry [d, e] contains the device ID that device d should dispatch
            expert e to. For shared experts, this is the nearest device on the same
            cluster axis that has the expert.

    Raises:
        RuntimeError: If shared experts are not distributed evenly across devices.
        RuntimeError: If shared expert IDs are not contiguous with routed expert IDs.

    Notes:
        - The function uses Manhattan distance along the cluster axis to find nearest devices.
        - If no device with the shared expert is on the same cluster axis, a default
          device is selected (the first in the list).
        - This mapping is critical for efficient MoE dispatch operations in distributed systems.
    """
    import torch

    # assuming [devices, experts] -> linearized mesh coordinate of owning device
    if len(expert_mapping_tensor.shape) != 2:
        raise RuntimeError(f"Invalid shape of mapping tensor. Expected: 2. Got: {len(expert_mapping_tensor.shape)}")

    devices = expert_mapping_tensor.shape[0]
    routed_experts = expert_mapping_tensor.shape[1]

    shared_experts = len(shared_expert_ids_to_devices)

    shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_devices, devices)

    if not len(set(shared_experts_per_device)) == 1:
        raise RuntimeError("Shared Experts must be distributed such that all devices have an equal number of experts")

    if list(range(routed_experts)) + sorted([se for se in shared_expert_ids_to_devices]) != list(
        range(routed_experts + shared_experts)
    ):
        raise RuntimeError("Shared expert IDs should be a contiguous continuation of routed expert IDs ")

    routed_and_shared_expert_mapping = torch.cat(
        [expert_mapping_tensor, torch.zeros((devices, shared_experts), dtype=expert_mapping_tensor.dtype)], dim=1
    )
    for disp_d in range(devices):
        for se, rec_ds in shared_expert_ids_to_devices.items():
            min_distance = mesh_shape[cluster_axis] + 1

            # just pick one as the default case. If none of the device assignments are on the same cluster axis as
            # disp_d then this expert will also get skipped by dispatch on disp_d
            routed_and_shared_expert_mapping[disp_d, se] = rec_ds[0]
            for rec_d in rec_ds:
                distance = cluster_distance(disp_d, rec_d, mesh_shape, cluster_axis)
                if distance is not None and distance < min_distance:
                    routed_and_shared_expert_mapping[disp_d, se] = rec_d
                    min_distance = distance

    return routed_and_shared_expert_mapping


def get_shared_experts_per_device(shared_expert_ids_to_devices: dict[int, list[int]], devices: int) -> list[int]:
    """
    Calculate the number of shared experts assigned to each device.

    This function counts how many shared experts are assigned to each device based on the
    provided mapping. It's used to verify even distribution of shared experts and to
    determine memory requirements for each device.

    Args:
        shared_expert_ids_to_devices: Dictionary mapping shared expert IDs to lists of
            device IDs where they are replicated.
        devices: Total number of devices in the system.

    Returns:
        List[int]: A list of length 'devices' where each element represents the number
            of shared experts assigned to that device index.

    Example:
        >>> mapping = {0: [0, 2], 1: [1, 3]}  # Expert 0 on devices 0,2; Expert 1 on devices 1,3
        >>> get_shared_experts_per_device(mapping, 4)
        [1, 1, 1, 1]  # Each device has 1 shared expert
    """
    shared_experts_per_device = [0] * devices
    for ds in shared_expert_ids_to_devices.values():
        for d in ds:
            shared_experts_per_device[d] += 1
    return shared_experts_per_device


def add_shared_expert_weights(
    routed_w0: "torch.Tensor",  # (layers, routed experts, hidden, matmul N)
    routed_w1: "torch.Tensor",  # (layers, routed experts, hidden, matmul N)
    routed_w2: "torch.Tensor",  # (layers, routed experts, matmul N, hidden)
    shared_w0: dict[int, "torch.Tensor"],  # id: (layers, 1, hidden, matmul N)
    shared_w1: dict[int, "torch.Tensor"],  # id: (layers, 1, hidden, matmul N)
    shared_w2: dict[int, "torch.Tensor"],  # id: (layers, 1, matmul N, hidden)
    shared_expert_ids_to_device: dict[int, list[int]],
    num_devices: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Combine routed and shared expert weights into a unified tensor format for MoE computation.

    This function reorganizes MoE expert weights by combining routed experts (unique to each
    device) with shared experts (replicated across multiple devices) into a single tensor
    format. It ensures proper weight distribution across devices for efficient MoE dispatch.

    Args:
        routed_w0: First layer weights for routed experts.
            Shape: [layers, routed_experts, hidden_dim, matmul_n]
        routed_w1: Second layer weights for routed experts.
            Shape: [layers, routed_experts, hidden_dim, matmul_n]
        routed_w2: Third layer weights for routed experts.
            Shape: [layers, routed_experts, matmul_n, hidden_dim]
        shared_w0: Dictionary mapping expert IDs to first layer weights for shared experts.
            Values have shape: [layers, 1, hidden_dim, matmul_n]
        shared_w1: Dictionary mapping expert IDs to second layer weights for shared experts.
            Values have shape: [layers, 1, hidden_dim, matmul_n]
        shared_w2: Dictionary mapping expert IDs to third layer weights for shared experts.
            Values have shape: [layers, 1, matmul_n, hidden_dim]
        shared_expert_ids_to_device: Dictionary mapping shared expert IDs to lists of
            device IDs where they should be replicated.
        num_devices: Total number of devices in the system.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors containing the
            combined weights (w0, w1, w2) with both routed and shared experts arranged
            for each device. Each output tensor has shape:
            [layers, num_devices * total_experts_per_device, ...]

    Raises:
        RuntimeError: If shared experts are not evenly distributed across devices
            (validated in get_shared_experts_per_device).

    Notes:
        - Routed experts are distributed sequentially across devices.
        - Shared experts are appended after routed experts for each device.
        - The function assumes contiguous expert IDs and even distribution.
        - Better validation of the shared_expert_ids_to_device mapping occurs in
          map_shared_experts function (also generally required for shared expert usage).

    Example:
        >>> # 4 devices, 8 routed experts (2 per device), 2 shared experts
        >>> routed_w0 = torch.randn(1, 8, 256, 512)
        >>> shared_w0 = {8: torch.randn(1, 1, 256, 512), 9: torch.randn(1, 1, 256, 512)}
        >>> mapping = {8: [0, 2], 9: [1, 3]}  # Each shared expert on 2 devices
        >>> result_w0, _, _ = add_shared_expert_weights(
        ...     routed_w0, routed_w1, routed_w2,
        ...     shared_w0, shared_w1, shared_w2,
        ...     mapping, 4)
        >>> result_w0.shape
        torch.Size([1, 12, 256, 512])  # 4 devices * 3 experts/device
    """
    import torch

    num_routed_experts = routed_w0.shape[1]
    num_routed_experts_per_device = num_routed_experts // num_devices
    num_shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_device, num_devices)[0]
    total_experts_per_device = num_routed_experts_per_device + num_shared_experts_per_device

    device_to_shared_experts = [[] for _ in range(num_devices)]
    sorted_shared_ids = sorted(shared_expert_ids_to_device.keys())

    for shared_id in sorted_shared_ids:
        for device in shared_expert_ids_to_device[shared_id]:
            device_to_shared_experts[device].append(shared_id)

    # Get tensor dimensions for pre-allocation
    layers = routed_w0.shape[0]
    hidden_dim = routed_w0.shape[2]
    matmul_n = routed_w0.shape[3]

    # Pre-allocate output tensors
    total_experts = num_devices * total_experts_per_device
    output_w0 = torch.empty((layers, total_experts, hidden_dim, matmul_n), dtype=routed_w0.dtype)
    output_w1 = torch.empty((layers, total_experts, hidden_dim, matmul_n), dtype=routed_w1.dtype)
    output_w2 = torch.empty((layers, total_experts, matmul_n, hidden_dim), dtype=routed_w2.dtype)

    # Fill output tensors using direct indexing
    for d in range(num_devices):
        # Calculate output indices for this device
        start_idx = d * total_experts_per_device
        routed_end_idx = start_idx + num_routed_experts_per_device

        # Copy routed experts for this device using slice assignment
        routed_start = d * num_routed_experts_per_device
        routed_end = (d + 1) * num_routed_experts_per_device

        output_w0[:, start_idx:routed_end_idx, :, :] = routed_w0[:, routed_start:routed_end, :, :]
        output_w1[:, start_idx:routed_end_idx, :, :] = routed_w1[:, routed_start:routed_end, :, :]
        output_w2[:, start_idx:routed_end_idx, :, :] = routed_w2[:, routed_start:routed_end, :, :]

        # Copy shared experts for this device
        for i, shared_id in enumerate(device_to_shared_experts[d]):
            shared_idx = routed_end_idx + i
            output_w0[:, shared_idx : shared_idx + 1, :, :] = shared_w0[shared_id]
            output_w1[:, shared_idx : shared_idx + 1, :, :] = shared_w1[shared_id]
            output_w2[:, shared_idx : shared_idx + 1, :, :] = shared_w2[shared_id]

    return output_w0, output_w1, output_w2


####################################################################################################
# Global constants — must be consistent with moe_ring_common.h
# W0_W1_BLOCK_TILES_W and W2_TILES_PER_A2A_ITER_W happen to share the same value today
# but represent distinct kernel quantities (W0/W1 read block width vs W2 a2a iter width).
# Kept as separate names to mirror the kernel; do not collapse.
W0_W1_BLOCK_TILES_W = 4  # matches moe_ring_common.h:W0_W1_BLOCK_TILES_W
W2_TILES_PER_A2A_ITER_W = 4  # matches moe_ring_common.h:W2_TILES_PER_A2A_ITER_W
BLOCK_TILES_H = 7  # block height for the default 14-tile transaction; see _block_tiles_h
# Half block-column (the odd gate/up column of a ring core with an odd column count): 2 tiles wide
# (W0 c, W1 c) x 14 K rows per 28-tile block. Matches moe_ring_common.h:W0_W1_HALF_BLOCK_TILES_{W,H}.
W0_W1_HALF_BLOCK_TILES_W = 2
W0_W1_HALF_BLOCK_TILES_H = 14
# DRAM transaction geometry (moe_ring_common.h): a block is 2 transactions; the transaction size is per shape
# (_tiles_per_txn): 14 tiles, or 20 for the Qwen3.8-Flash-Next expert. Half-width last W2 a2a iteration (20-tile
# transactions only): 2 output tiles wide (W2_HALF_A2A_ITER_TILES_W).
W0_W1_TXNS_PER_BLOCK = 2
DEFAULT_TILES_PER_TXN = 14
ALT_TILES_PER_TXN = 20
W2_HALF_A2A_ITER_TILES_W = 2

# Historical model-specific shard constants — superseded by the generalized
# _shard_tiles() / _w2_shard_tiles() formulas below.  Kept as commented-out
# reference so reviewers can compare old vs new distributions.
#
# DeepSeek (hidden=7168, intermediate=2048): the generalized formulas produce
# the SAME distribution as the old hardcoded config.
#   DS_PAD_CORES = {1, 2, 4, 5, 7, 8, 10, 11}
#   DS_W0_W1_SHARD_VALS = [6, 5]                       # -> [6,5,5,6,5,5,6,5,5,6,5,5]
#   DS_W2_SHARD_VALS = {False: (2, 2), True: (3, 1)}
#
# GPT-OSS (hidden=2880, intermediate=2880): the generalized formulas produce a
# DIFFERENT per-core assignment (same totals — 6×8 + 6×7 = 90 — but the
# Bresenham pattern interleaves big/small cores differently).
#   Old:  [8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7]   pad_cores = {2,3,6,7,10,11}
#   New:  [8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7]   (Euclidean rhythm)
#   GPT_PAD_CORES = {2, 3, 6, 7, 10, 11}
#   GPT_W0_W1_SHARD_VALS = [8, 7]
#   GPT_W2_SHARD_VALS = {False: (4, 0), True: (3, 1)}
####################################################################################################


####################################################################################################
# Generalized shard distribution formulas (added for shape generalization)
# MUST stay in sync with constexpr equivalents in:
#   ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h
####################################################################################################


def _shard_tiles(n_tiles: int, core_id: int, n_cores: int) -> int:
    """Euclidean rhythm (Bresenham) distribution: tiles owned by ring position core_id."""
    n_big = n_tiles % n_cores
    small = n_tiles // n_cores
    is_big = n_big > 0 and (core_id * n_big) % n_cores < n_big
    return small + (1 if is_big else 0)


def _w2_shard_tiles(Ht: int, core_id: int, Nt: int, n_cores: int) -> int:
    """W2 hidden-tile distribution per ring position.

    Uses complementary pattern when Nt%n_cores + Ht%n_cores == n_cores:
    big W2 cores = small W0/W1 cores (balances DRAM load).
    Otherwise falls back to _shard_tiles(Ht, core_id, n_cores).
    """
    n_big_nt = Nt % n_cores
    n_big_ht = Ht % n_cores
    small_ht = Ht // n_cores
    if n_big_nt + n_big_ht == n_cores:
        is_big_nt = n_big_nt > 0 and (core_id * n_big_nt) % n_cores < n_big_nt
        return small_ht if is_big_nt else small_ht + 1
    return _shard_tiles(Ht, core_id, n_cores)


def _even_stride_at_least_a2a_width(tiles: int) -> int:
    even_tiles = tiles + (tiles % 2)
    return max(even_tiles, W2_TILES_PER_A2A_ITER_W)


def _block_tiles_h(tiles_per_txn: int) -> int:
    """K rows of a 4-wide block (W0/W1 block-column, W2 a2a iteration): 7 for 14-tile transactions, 10 for 20."""
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn // W0_W1_BLOCK_TILES_W


def _half_block_tiles_h(tiles_per_txn: int) -> int:
    """K rows of a 2-wide block (half block-column, half a2a iteration): 14 for 14-tile transactions, 20 for 20."""
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn // W0_W1_HALF_BLOCK_TILES_W


def _w2_num_a2a_iters(Ht: int, num_cores: int) -> int:
    return math.ceil(math.ceil(Ht / num_cores) / W2_TILES_PER_A2A_ITER_W)


def _w2_last_a2a_iter_half(Ht: int, num_cores: int, tiles_per_txn: int) -> bool:
    """With 20-tile transactions the last W2 a2a iteration is half width when every core has at most 2 output tiles
    left for it; 14-tile layouts keep 4-wide iterations only (moe_ring_common.h:w2_last_a2a_iter_half)."""
    last = math.ceil(Ht / num_cores) % W2_TILES_PER_A2A_ITER_W
    return tiles_per_txn != DEFAULT_TILES_PER_TXN and 0 < last <= W2_HALF_A2A_ITER_TILES_W


def _w2_core_blocks_per_expert(Ht: int, k_w2_tiles: int, num_cores: int, tiles_per_txn: int) -> int:
    """W2 blocks one ring core stores per (layer, expert) (moe_ring_common.h:w2_core_blocks_per_expert)."""
    iters = _w2_num_a2a_iters(Ht, num_cores)
    half = 1 if _w2_last_a2a_iter_half(Ht, num_cores, tiles_per_txn) else 0
    return (iters - half) * math.ceil(k_w2_tiles / _block_tiles_h(tiles_per_txn)) + half * math.ceil(
        k_w2_tiles / _half_block_tiles_h(tiles_per_txn)
    )


def _tiles_per_txn(Ht: int, Nt: int, has_bias: bool) -> int:
    """The per-shape DRAM transaction size in tiles of both weight streams (moe_ring_common.h:tiles_per_txn_for_shape):
    20 for the Qwen3.8-Flash-Next expert (hidden 2560 = 80 tiles, intermediate 640 = 20 tiles, no bias), whose layout
    then has no padding at all on 8 banks; 14 for every other shape (DeepSeek, GPT-OSS, ...)."""
    return ALT_TILES_PER_TXN if (Ht, Nt, has_bias) == (80, 20, False) else DEFAULT_TILES_PER_TXN


def _w0_w1_compact_layout(
    k_dram_tiles: int,
    Nt: int,
    num_cores: int,
    num_banks: int | None = None,
    tiles_per_txn: int = DEFAULT_TILES_PER_TXN,
) -> dict:
    """Geometry of the compact W0/W1 layout (mirrors ``MoeRingConfig`` in moe_ring_common.h).

    ``k_dram_tiles`` is the stored K height in tiles (hidden tiles, plus one with bias). A block is 2 transactions
    of ``tiles_per_txn`` tiles, i.e. ``block_rows`` stored rows of 4 tiles. Ring core c stores only its
    ``_shard_tiles(Nt, c, num_cores)`` gate/up columns: ``cols // 2`` block-columns of ``blocks_per_col`` blocks
    (4 tiles wide x ``_block_tiles_h`` K rows: W0 c, W1 c, W0 c+1, W1 c+1) and, for an odd count, one half
    block-column of ``blocks_per_half_col`` blocks (2 tiles wide x ``_half_block_tiles_h`` K rows, two consecutive
    K rows per stored 4-tile row). Per (layer, expert) the cores' slices are laid back to back
    (``core_block_offsets``) and that stream is cut into ``num_banks`` equal pieces of ``bank_blocks_per_expert``
    blocks (zero-padded at the end); piece b is stored in bank b.
    """
    num_banks = num_cores if num_banks is None else num_banks
    blocks_per_col = math.ceil(k_dram_tiles / _block_tiles_h(tiles_per_txn))
    blocks_per_half_col = math.ceil(k_dram_tiles / _half_block_tiles_h(tiles_per_txn))
    cols = [_shard_tiles(Nt, c, num_cores) for c in range(num_cores)]
    core_blocks = [(n // 2) * blocks_per_col + (n % 2) * blocks_per_half_col for n in cols]
    core_block_offsets = [sum(core_blocks[:c]) for c in range(num_cores)]
    expert_blocks = sum(core_blocks)
    return {
        "tiles_per_txn": tiles_per_txn,
        "block_rows": _block_tiles_h(tiles_per_txn) * ttnn.TILE_SIZE,
        "blocks_per_col": blocks_per_col,
        "blocks_per_half_col": blocks_per_half_col,
        "cols": cols,
        "core_blocks": core_blocks,
        "core_block_offsets": core_block_offsets,
        "expert_blocks": expert_blocks,
        "num_banks": num_banks,
        "bank_blocks_per_expert": math.ceil(expert_blocks / num_banks),
        # Today's per-core stride tensor shape is kept where the compact layout is byte-identical to it:
        # 14-tile transactions, every core owning the same even column count, one core per bank.
        "uniform": tiles_per_txn == DEFAULT_TILES_PER_TXN
        and len(set(cols)) == 1
        and cols[0] % 2 == 0
        and num_banks == num_cores,
    }


def w0_w1_bank_rows_per_expert(
    k_dram_tiles: int,
    Nt: int,
    num_cores: int,
    num_banks: int | None = None,
    tiles_per_txn: int = DEFAULT_TILES_PER_TXN,
) -> int:
    """Rows (of the 4-tile-wide packed W0/W1 tensor) one DRAM bank holds per (layer, expert)."""
    layout = _w0_w1_compact_layout(k_dram_tiles, Nt, num_cores, num_banks, tiles_per_txn)
    return layout["bank_blocks_per_expert"] * layout["block_rows"]


def w2_core_rows_per_expert(Ht: int, k_w2_tiles: int, num_cores: int, tiles_per_txn: int) -> int:
    """Rows (of the 4-tile-wide packed W2 tensor) one ring core holds per (layer, expert)."""
    return _w2_core_blocks_per_expert(Ht, k_w2_tiles, num_cores, tiles_per_txn) * (
        _block_tiles_h(tiles_per_txn) * ttnn.TILE_SIZE
    )


def _w2_blocks_from_groups(N_reordered: "torch.Tensor", k_tiles: int, Ht: int, tiles_per_txn: int) -> "torch.Tensor":
    """Lay the ring-rotated W2 groups (num_cores, L, E, groups, k_tiles*TILE, 4*TILE) out in DRAM blocks.

    Full a2a iterations: 4 wide, K padded to whole ``_block_tiles_h`` blocks. A half-width last iteration
    (``_w2_last_a2a_iter_half``): its 2 valid columns only, K padded to whole ``_half_block_tiles_h`` blocks, two
    consecutive K tile rows side by side per stored row. Without a half iteration the result keeps the grouped shape
    (num_cores, L, E, groups, K_padded, 4*TILE) -- the old layout for 14-tile transactions; with one it is
    (num_cores, L, E, blocks, block_rows, 4*TILE).
    """
    import torch

    num_cores, L, E, groups, rows, width = N_reordered.shape
    tile = ttnn.TILE_SIZE
    full_rows = math.ceil(k_tiles / _block_tiles_h(tiles_per_txn)) * _block_tiles_h(tiles_per_txn) * tile
    half = _w2_last_a2a_iter_half(Ht, num_cores, tiles_per_txn)
    full_groups = groups - (1 if half else 0)
    full = N_reordered[:, :, :, :full_groups]
    if full_rows > rows:
        pad = torch.zeros(num_cores, L, E, full_groups, full_rows - rows, width, dtype=N_reordered.dtype)
        full = torch.cat([full, pad], dim=4)
    if not half:
        return full
    half_rows = math.ceil(k_tiles / _half_block_tiles_h(tiles_per_txn)) * _half_block_tiles_h(tiles_per_txn) * tile
    last = N_reordered[:, :, :, groups - 1, :, : W2_HALF_A2A_ITER_TILES_W * tile]
    if half_rows > rows:
        pad = torch.zeros(num_cores, L, E, half_rows - rows, W2_HALF_A2A_ITER_TILES_W * tile, dtype=N_reordered.dtype)
        last = torch.cat([last, pad], dim=3)
    last = last.reshape(num_cores, L, E, half_rows // (2 * tile), 2, tile, 2 * tile).permute(0, 1, 2, 3, 5, 4, 6)
    last = last.reshape(num_cores, L, E, half_rows // 2, width)
    stream = torch.cat([full.reshape(num_cores, L, E, full_groups * full_rows, width), last], dim=3)
    block_rows = _block_tiles_h(tiles_per_txn) * tile
    return stream.reshape(num_cores, L, E, -1, block_rows, width)


def effective_matmul_ring_size(mesh_device, bh_ring_size: int = 8) -> int:
    """Matmul ring N used by ``moe_compute`` on this device.

    One matmul core is used per DRAM-bank-adjacent worker. On Blackhole up to one DRAM bank
    can be fused off, so the live bank count is 7 or 8. On Wormhole DRAM banks are never
    harvested, so the ring is always 12.

    The public ``ttnn.experimental.moe_compute`` op auto-detects the ring from the device and
    no longer exposes a ``bh_ring_size`` knob, so the returned value is the ring the op will
    actually use. Pass the result to the ``prepare_*`` / ``get_weight_*`` helpers so host weight
    layout matches the op.
    """
    if mesh_device.arch() == ttnn.Arch.BLACKHOLE:
        return len(ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0))
    return 12


def auto_output_width_shard_dim(
    hidden_size: int,
    tile_size: int = 32,
    max_dim: int = 4,
    matmul_ring_size: int | None = None,
) -> int:
    """Largest divisor d of (hidden_size // tile_size) with d <= max_dim.

    When ``matmul_ring_size`` is set, also require ``matmul_ring_size % d == 0`` so the
    chosen width parallelism divides the matmul ring evenly. This matches the op's
    ring-aware auto-derivation in ``moe_compute_device_operation.cpp::invoke()``.

    Use ``effective_matmul_ring_size(mesh_device, bh_ring_size)`` for ``matmul_ring_size``
    when preparing test tensors so host layout matches the device op.
    """
    hidden_tiles = hidden_size // tile_size
    for d in range(max_dim, 0, -1):
        if hidden_tiles % d == 0 and (matmul_ring_size is None or matmul_ring_size % d == 0):
            return d
    return 1


def prepare_w0_w1_tensor_for_moe_compute(
    torch_w0: "torch.Tensor",
    torch_w1: "torch.Tensor",
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
    num_banks: int | None = None,
    tiles_per_txn: int | None = None,
):
    """
    Prepare the w0_w1 tensor input for moe_compute by interleaving chunks of w0 and w1 width-wise.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        shard_map: List of logical shard sizes (one per ring core).
        num_banks: DRAM banks the tensor is HEIGHT_SHARDED over (default: one per ring core, which is what
            the op requires).
        tiles_per_txn: DRAM transaction size in tiles (default: the op's per-shape choice, ``_tiles_per_txn``
            for K and N without bias; ``prepare_w0_w1_tensor_with_bias`` passes the with-bias choice).

    Returns:
        torch_w0_w1_paired: tensor of shape (num_banks, L, E, bank_blocks_per_expert, block_rows, 4*TILE_SIZE)
        with block_rows = 7*TILE_SIZE (14-tile transactions) or 10*TILE_SIZE (20); for 14-tile transactions with
        every core owning the same even column count (one core per bank) this is byte-identical to, and keeps the
        shape of, the per-core stride layout (num_cores, L, E, groups_per_core, K_padded, 4*TILE_SIZE).

    Compact layout (see ``_w0_w1_compact_layout``): ring core c stores only its ``shard_map[c]`` columns --
    full 4-wide block-columns (W0 c, W1 c, W0 c+1, W1 c+1) over K padded to whole blocks (7 K rows for 14-tile
    transactions, 10 for 20), then, for an odd count, the last column as a 2-wide half block-column over K padded to
    whole half blocks (14 or 20 K rows) with two consecutive K tile rows side by side per stored tile row. Per (layer, expert) the cores' slices are laid back to back
    and cut into num_banks equal pieces (zero-padded); piece b is the per-(layer, expert) unit of bank b's
    shard. ttnn.from_torch with HEIGHT_SHARDED over num_banks shards then puts piece b in bank b.
    """
    import torch

    # Check that K and N are divisible by ttnn.TILE_SIZE
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Nt = N // ttnn.TILE_SIZE
    num_cores = len(shard_map)
    if num_cores == 0:
        raise ValueError("shard_map must contain one entry per ring core")
    expected_shard_map = [_shard_tiles(Nt, core_id, num_cores) for core_id in range(num_cores)]
    if shard_map != expected_shard_map:
        raise RuntimeError(f"W0W1 shard map must match the kernel distribution {expected_shard_map}, got: {shard_map}")

    if tiles_per_txn is None:
        tiles_per_txn = _tiles_per_txn(K // ttnn.TILE_SIZE, Nt, has_bias=False)
    layout = _w0_w1_compact_layout(K // ttnn.TILE_SIZE, Nt, num_cores, num_banks, tiles_per_txn)
    num_banks = layout["num_banks"]
    # K padded to whole blocks of the block-column height and of the half block-column height.
    Kp_full = layout["blocks_per_col"] * _block_tiles_h(tiles_per_txn) * ttnn.TILE_SIZE
    Kp_half = layout["blocks_per_half_col"] * _half_block_tiles_h(tiles_per_txn) * ttnn.TILE_SIZE
    Kp = max(Kp_full, Kp_half)

    if K < Kp:
        padding = torch.zeros((L, E, Kp - K, N), dtype=torch_w0.dtype)
        working_torch_w0 = torch.concat([torch_w0, padding], dim=2)
        working_torch_w1 = torch.concat([torch_w1, padding], dim=2)
    else:
        working_torch_w0 = torch_w0
        working_torch_w1 = torch_w1

    # Reshape to expose chunks: (L, E, K, N) -> (L, E, K, Nt, ttnn.TILE_SIZE)
    w0_chunks = working_torch_w0.view(L, E, Kp, Nt, ttnn.TILE_SIZE)
    w1_chunks = working_torch_w1.view(L, E, Kp, Nt, ttnn.TILE_SIZE)

    # Stack w0 and w1 chunks together: (L, E, K, Nt, 2, ttnn.TILE_SIZE)
    # This puts w0_chunk_i and w1_chunk_i adjacent to each other
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)

    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE)
    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
    torch_w0_w1_interleaved = stacked.view(L, E, Kp, Nt, 2 * ttnn.TILE_SIZE)

    # Permute to move Nt before K: (L, E, K, Nt, 2*TILE) -> (L, E, Nt, K, 2*TILE)
    torch_w0_w1_permuted = torch_w0_w1_interleaved.permute(0, 1, 3, 2, 4)

    # Each core's compact slice as stored 4-tile rows: (L, E, rows, 4*TILE).
    each_slice = []
    start_tile = 0
    for num_tiles in shard_map:
        pairs = num_tiles // 2
        if pairs > 0:
            # (L, E, 2*pairs, Kp_full, 2*TILE) -> (L, E, pairs, Kp_full, 4*TILE): row k = W0 c, W1 c, W0 c+1, W1 c+1
            pair_cols = torch_w0_w1_permuted[:, :, start_tile : start_tile + 2 * pairs, :Kp_full, :]
            pair_cols = pair_cols.reshape(L, E, pairs, 2, Kp_full, 2 * ttnn.TILE_SIZE).permute(0, 1, 2, 4, 3, 5)
            each_slice.append(pair_cols.reshape(L, E, pairs * Kp_full, 4 * ttnn.TILE_SIZE))
        if num_tiles % 2:
            # (L, E, Kp_half, 2*TILE) -> (L, E, Kp_half / 2, 4*TILE): tile row j = K tile rows 2j and 2j+1 side by side
            half_col = torch_w0_w1_permuted[:, :, start_tile + 2 * pairs, :Kp_half, :]
            half_col = half_col.reshape(L, E, Kp_half // (2 * ttnn.TILE_SIZE), 2, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE)
            each_slice.append(half_col.permute(0, 1, 2, 4, 3, 5).reshape(L, E, Kp_half // 2, 4 * ttnn.TILE_SIZE))
        start_tile += num_tiles

    block_rows = layout["block_rows"]  # every block (full or half) is 2 * tiles_per_txn / 4 stored tile rows
    bank_blocks = layout["bank_blocks_per_expert"]
    stream_pad_blocks = num_banks * bank_blocks - layout["expert_blocks"]
    if stream_pad_blocks > 0:
        each_slice.append(
            torch.zeros(L, E, stream_pad_blocks * block_rows, 4 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype)
        )
    stream = torch.cat(each_slice, dim=2)

    # (L, E, num_banks * bank_blocks * block_rows, 4*TILE) -> (num_banks, L, E, bank_blocks, block_rows, 4*TILE)
    torch_w0_w1_paired = stream.view(L, E, num_banks, bank_blocks, block_rows, 4 * ttnn.TILE_SIZE)
    torch_w0_w1_paired = torch_w0_w1_paired.permute(2, 0, 1, 3, 4, 5)
    if layout["uniform"]:
        groups_per_core = shard_map[0] // 2
        return torch_w0_w1_paired.reshape(num_cores, L, E, groups_per_core, Kp_full, 4 * ttnn.TILE_SIZE)
    return torch_w0_w1_paired.contiguous()


def prepare_w2_tensor_for_moe_compute(
    torch_w2: "torch.Tensor",
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
    tiles_per_txn: int | None = None,
) -> "torch.Tensor":
    """
    Prepare the w2 tensor input for moe_compute by padding and reordering tiles.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        w2_shard_map: List of tuples (last_group_tiles, last_group_pad_tiles) for each core
        w0_w1_shard_map: List of shard sizes from w0_w1 preparation
        tiles_per_txn: DRAM transaction size in tiles (default: the op's per-shape choice without bias).

    Returns:
        torch_w2_paired: tensor of shape (num_cores, L, E, w2_groups_per_core, N_padded, 4*ttnn.TILE_SIZE), or,
        when the last a2a iteration is half width (see ``_w2_blocks_from_groups``),
        (num_cores, L, E, w2_blocks_per_core, block_rows, 4*ttnn.TILE_SIZE).

    See :func:`prepare_w0_w1_tensor_for_moe_compute` for the layout/HEIGHT_SHARDED note.
    """
    import torch

    # Check that N and K are divisible by ttnn.TILE_SIZE
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    each_shard = []

    start_col = 0
    # groups are always 4 tiles wide in K, and full N
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE])
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        if last_group_pad_tiles > 0:
            each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, num_cores, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, 5, 128) -> (12, L, E, 5, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles first
    N_grouped = all_groups_per_bank.view(
        num_cores, L, E, w2_groups_per_core, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, num groups, 64, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)

    # Figure out the starting position for each chunk
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(w0_w1_shard_map, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(num_cores, L, E, w2_groups_per_core, -1, 4 * ttnn.TILE_SIZE)

    # Pad "N" to whole DRAM blocks (and lay a half-width last iteration out 2 wide).
    Nt = N // ttnn.TILE_SIZE
    if tiles_per_txn is None:
        tiles_per_txn = _tiles_per_txn(Kt, Nt, has_bias=False)
    return _w2_blocks_from_groups(N_reordered, Nt, Kt, tiles_per_txn)


def prepare_w0_w1_tensor_with_bias(
    torch_w0: "torch.Tensor",
    torch_w1: "torch.Tensor",
    torch_b0: "torch.Tensor",
    torch_b1: "torch.Tensor",
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
):
    """
    Prepare the w0_w1 tensor with bias by concatenating bias rows along K dimension,
    padding to transaction-aligned height, then delegating to prepare_w0_w1_tensor_for_moe_compute.

    Converts true PyTorch bias format (L, E, N) to kernel tile format (L, E, 32, N) with
    only the first row populated, then concatenates to weights along K dimension.

    The kernel reads W0/W1 in blocks of 2 * tiles_per_txn tiles (the per-shape transaction size, _tiles_per_txn).
    With bias, K goes from K/32 tiles to (K/32 + 1) tiles, padded to whole blocks; the weight tensor must contain
    those padding tiles (zeros) so the DRAM reads don't overrun the expert boundary.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        torch_b0: Bias tensor of shape (L, E, N) -- true PyTorch format
        torch_b1: Bias tensor of shape (L, E, N) -- true PyTorch format
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        shard_map: List of shard sizes for each core

    Returns:
        torch_w0_w1_paired: Prepared tensor with bias of shape (num_cores, L, E, groups_per_core, K_padded, 4*ttnn.TILE_SIZE)

    See also:
        Module docstring for full layout contract and constants that must match
        moe_ring_common.h; ``prepare_w0_w1_tensor_for_moe_compute`` for the non-bias path.
    """
    import torch

    # Check that K and N are divisible by ttnn.TILE_SIZE
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    # This constant must match moe_ring_common.h — it determines the DRAM read block alignment.
    K_tiles = K // ttnn.TILE_SIZE
    K_tiles_with_bias = K_tiles + 1
    K_with_bias = K_tiles_with_bias * ttnn.TILE_SIZE

    # Convert true PyTorch bias (L, E, N) to kernel tile format (L, E, 32, N) with only row 0 populated.
    torch_b0_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, N, dtype=torch_b0.dtype)
    torch_b0_tiled[:, :, 0, :] = torch_b0
    torch_b1_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, N, dtype=torch_b1.dtype)
    torch_b1_tiled[:, :, 0, :] = torch_b1

    torch_w0_b0 = torch.cat([torch_w0, torch_b0_tiled], dim=2)  # (L, E, K+32, N)
    torch_w1_b1 = torch.cat([torch_w1, torch_b1_tiled], dim=2)  # (L, E, K+32, N)

    tiles_per_txn = _tiles_per_txn(K_tiles, N // ttnn.TILE_SIZE, has_bias=True)
    return prepare_w0_w1_tensor_for_moe_compute(
        torch_w0_b0, torch_w1_b1, L, E, K_with_bias, N, shard_map, tiles_per_txn=tiles_per_txn
    )


def prepare_w2_tensor_with_bias(
    torch_w2: "torch.Tensor",
    torch_b2: "torch.Tensor",
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> "torch.Tensor":
    """
    Prepare the w2 tensor with bias. The bias tile row is concatenated along N,
    but only the weight tiles are ring-rotated — the bias tile stays fixed at
    position N/32 for all cores (matching GPT-OSS behavior).

    Converts true PyTorch bias format (L, E, K) to kernel tile format (L, E, 32, K)
    with only the first row populated, then performs K-column sharding.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        torch_b2: Bias tensor of shape (L, E, K) -- true PyTorch format
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        w2_shard_map: List of tuples (last_group_tiles, last_group_pad_tiles) for each core
        w0_w1_shard_map: List of shard sizes from w0_w1 preparation

    Returns:
        N_with_bias: Prepared tensor of shape (num_cores, L, E, groups_per_core, N_target, 4*ttnn.TILE_SIZE)

    See also:
        Module docstring for full layout contract and constants that must match
        moe_ring_common.h; ``prepare_w2_tensor_for_moe_compute`` for the non-bias path.
    """
    import torch

    # Check that N and K are divisible by ttnn.TILE_SIZE
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    # Convert true PyTorch bias (L, E, K) to kernel tile format (L, E, 32, K) with only row 0 populated.
    torch_b2_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, K, dtype=torch_b2.dtype)
    torch_b2_tiled[:, :, 0, :] = torch_b2

    # Column-shard K dimension for weights
    each_shard = []
    start_col = 0
    # groups are always 4 tiles wide in K, and full N
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE])
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        if last_group_pad_tiles > 0:
            each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, num_cores, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, groups_per_core, 128) -> (12, L, E, groups_per_core, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles (weight tiles only, no bias yet)
    N_grouped = all_groups_per_bank.view(
        num_cores, L, E, w2_groups_per_core, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, num groups, Nt, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)

    # Figure out the starting position for each chunk
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(w0_w1_shard_map, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(num_cores, L, E, w2_groups_per_core, -1, 4 * ttnn.TILE_SIZE)

    # Now prepare bias tile row with the same K-column sharding
    b2_each_shard = []
    start_col = 0
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        b2_each_shard.append(
            torch_b2_tiled[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE]
        )
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        b2_each_shard.append(torch_b2_tiled[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        if last_group_pad_tiles > 0:
            b2_each_shard.append(
                torch.zeros(L, E, ttnn.TILE_SIZE, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_b2_tiled.dtype)
            )

    torch_b2_reordered = torch.cat(b2_each_shard, dim=-1)
    b2_groups_per_bank = torch_b2_reordered.view(L, E, ttnn.TILE_SIZE, num_cores, -1, 4 * ttnn.TILE_SIZE)
    b2_groups_per_bank = b2_groups_per_bank.permute(3, 0, 1, 4, 2, 5)  # (12, L, E, groups_per_core, 32, 128)

    # Concatenate bias tile row after weight tiles (NOT ring-rotated)
    N_with_bias = torch.cat([N_reordered, b2_groups_per_bank], dim=4)  # (12, L, E, groups_per_core, N+32, 128)

    # Pad "N+32" to whole DRAM blocks (and lay a half-width last iteration out 2 wide).
    tiles_per_txn = _tiles_per_txn(Kt, Nt, has_bias=True)
    return _w2_blocks_from_groups(N_with_bias, Nt + 1, Kt, tiles_per_txn)


def get_weight_core_shard_maps(mesh_device, hidden_size: int, intermediate_size: int):
    """Compute per-ring-position shard maps for W0/W1 and W2 weight tensors.

    Uses _shard_tiles() (Euclidean rhythm) for W0/W1 and _w2_shard_tiles()
    (complementary when Nt%n_cores + Ht%n_cores == n_cores) for W2.
    Ring ordering: DRAM bank logical coords sorted by (y, x) descending.

    The matmul ring size is auto-detected from the device via ``effective_matmul_ring_size``
    (12 on Wormhole, 7/8 on Blackhole) so the packed weights always line up with the op. The
    weights are still HEIGHT_SHARDED across the live DRAM banks, so ``dram_core_range_set``
    has exactly ``n_dram_banks`` entries while the shard maps have ``target_ring_size`` entries.
    """
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    n_dram_banks = len(in0_core_coords)
    target_ring_size = effective_matmul_ring_size(mesh_device)

    core2dram = {cc: dram_bank_id for dram_bank_id, cc in enumerate(in0_core_coords)}
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    Nt = intermediate_size // ttnn.TILE_SIZE
    Ht = hidden_size // ttnn.TILE_SIZE
    # groups_per_core is fixed across ring positions (max W2 tiles owned by any ring slot,
    # rounded up to W2_TILES_PER_A2A_ITER_W). Must use target_ring_size so it matches the
    # kernel's num_a2a_iters * W2_TILES_PER_A2A_ITER_W / W2_TILES_PER_TXN.
    max_w2_tiles = (Ht + target_ring_size - 1) // target_ring_size
    groups_per_core = (max_w2_tiles + W2_TILES_PER_A2A_ITER_W - 1) // W2_TILES_PER_A2A_ITER_W

    sorted_dram_core_coords = []
    w0_w1_shard_map = []
    w2_shard_map_list = []

    for ring_pos in range(target_ring_size):
        # First n_dram_banks ring positions own actual DRAM-bank-adjacent cores; positions
        # beyond that are synthetic (their data lives in the leading dim of the prepared
        # tensor, which HEIGHT_SHARDED regroups onto the same n_dram_banks physical shards).
        if ring_pos < n_dram_banks:
            sorted_dram_core_coords.append(core2dram[in0_core_coords_sorted[ring_pos]])

        w0_w1_tiles = _shard_tiles(Nt, ring_pos, target_ring_size)
        w0_w1_shard_map.append(w0_w1_tiles)

        w2_tiles = _w2_shard_tiles(Ht, ring_pos, Nt, target_ring_size)
        last_group_tiles = w2_tiles - (groups_per_core - 1) * W2_TILES_PER_A2A_ITER_W
        last_group_pad_tiles = groups_per_core * W2_TILES_PER_A2A_ITER_W - w2_tiles
        w2_shard_map_list.append((last_group_tiles, last_group_pad_tiles))

    dram_core_coords = [ttnn.CoreCoord(c, 0) for c in sorted_dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc) for cc in dram_core_coords])

    return w0_w1_shard_map, w2_shard_map_list, dram_core_range_set


def get_weight_mem_configs(
    num_layers,
    experts_per_device,
    hidden_size,
    intermediate_size,
    w0_w1_shard_map,
    w2_shard_map,
    dram_core_range_set,
    has_bias=False,
):
    """
    Get memory configurations for W0/W1 and W2 weight tensors.

    When has_bias=True:
    - W0/W1: K dimension grows by 1 tile (for bias) and is padded to transaction boundary
    - W2: N dimension grows by 1 tile (for bias) and is padded to align with 7-tile reads

    Memory layout: always HEIGHT_SHARDED with leading dim = num_banks. The ring size equals
    the live DRAM-bank count (12 on Wormhole, 7/8 on Blackhole), so num_cores == num_banks
    (1:1). Each ring core's slice therefore covers exactly one bank; the "bank-run" loop in
    dm0.cpp is retained for correctness with direct prim callers that may use a different ring size.

    `dram_core_range_set` is constructed by `get_weight_core_shard_maps` and has exactly
    num_banks entries (placement target).

    Returns:
        tuple: (w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total)
            - K_for_shard: The padded K dimension for W0/W1
            - w2_N_total: The padded N dimension for W2
    """

    # Check that hidden_size and intermediate_size are divisible by ttnn.TILE_SIZE
    if hidden_size % ttnn.TILE_SIZE != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if intermediate_size % ttnn.TILE_SIZE != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})"
        )

    # Per-shape DRAM transaction size (both streams) and the block height it gives.
    Ht = hidden_size // ttnn.TILE_SIZE
    Nt = intermediate_size // ttnn.TILE_SIZE
    num_cores = len(w0_w1_shard_map)
    tiles_per_txn = _tiles_per_txn(Ht, Nt, has_bias)
    block_h = _block_tiles_h(tiles_per_txn)

    # Calculate K dimension for W0/W1 (stored K tiles: hidden, plus one bias tile row)
    k_dram_tiles = Ht + (1 if has_bias else 0)
    # K padded to whole blocks (the height of a 4-wide block-column)
    K_for_shard = math.ceil(k_dram_tiles / block_h) * block_h * ttnn.TILE_SIZE

    # Calculate N dimension for W2 (stored K tiles of W2: intermediate, plus one bias tile row), padded to whole
    # blocks for the full a2a iterations
    k_w2_tiles = Nt + (1 if has_bias else 0)
    w2_N_total = math.ceil(k_w2_tiles / block_h) * block_h * ttnn.TILE_SIZE

    # HEIGHT_SHARDED with num_banks shards. Shard height is computed from the LOGICAL view
    # (per-ring-core view): num_cores * groups_per_core * K_for_shard rows total flat,
    # which redistributes evenly into num_banks shards because the prepare functions enforce
    # divisibility (see prepare_w0_w1_tensor_for_moe_compute / prepare_w2_tensor_for_moe_compute).
    num_banks = dram_core_range_set.num_cores()

    # W0/W1 memory config: the compact layout (prepare_w0_w1_tensor_for_moe_compute) gives every bank the same
    # rows per (layer, expert), so the shard height is that times layers * experts.
    if w0_w1_shard_map != [_shard_tiles(Nt, c, num_cores) for c in range(num_cores)]:
        raise RuntimeError(f"get_weight_mem_configs: w0_w1 shard map {w0_w1_shard_map} does not match the kernel's")
    w0_w1_shard_height = (
        num_layers
        * experts_per_device
        * w0_w1_bank_rows_per_expert(k_dram_tiles, Nt, num_cores, num_banks, tiles_per_txn)
    )
    w0_w1_shard_width = W0_W1_BLOCK_TILES_W * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # W2 memory config
    w2_core_rows = w2_core_rows_per_expert(Ht, k_w2_tiles, num_cores, tiles_per_txn)
    w2_total_rows = num_layers * experts_per_device * num_cores * w2_core_rows
    if w2_total_rows % num_banks != 0:
        raise RuntimeError(
            f"get_weight_mem_configs: w2 total rows {w2_total_rows} not divisible by num_banks {num_banks} "
            f"(num_cores={num_cores}, w2_core_rows={w2_core_rows})"
        )
    w2_shard_height = w2_total_rows // num_banks
    w2_shard_width = W2_TILES_PER_A2A_ITER_W * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    return w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total
