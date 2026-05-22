# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""ttnn-tensor versions of the weight-prep helpers in ``moe_compute_utils.py``.

The torch versions in the sibling module operate on host (CPU) tensors and
the result then gets pushed to device via ``ttnn.from_torch``. The functions
here accept ``ttnn.Tensor`` inputs and produce ``ttnn.Tensor`` outputs using
``ttnn.concat`` / ``ttnn.reshape`` / ``ttnn.permute`` / ``ttnn.slice`` /
``ttnn.zeros``, so callers that already have weights on device can skip the
host round-trip.

Only the tensor-manipulating functions are duplicated — pure-logic helpers
(``cluster_distance``, ``get_shared_experts_per_device``, ``_shard_tiles``,
``_w2_shard_tiles``, ``auto_output_width_shard_dim``, ``get_weight_core_shard_maps``,
``get_weight_mem_configs``) live in the original module and are imported by
callers as-is.

The block-size constants (``BLOCK_TILES_H``, ``BLOCK_TILES_W``) and shape
invariants are the same as in the torch version; see the module docstring of
``moe_compute_utils`` for the full kernel layout contract.
"""

from __future__ import annotations

import math

import ttnn

from ttnn._experimental.moe_compute_utils import BLOCK_TILES_H


def _stack(tensors: list[ttnn.Tensor], dim: int) -> ttnn.Tensor:
    """`torch.stack` equivalent built from `ttnn.unsqueeze` + `ttnn.concat`."""
    return ttnn.concat([ttnn.unsqueeze(t, dim) for t in tensors], dim=dim)


def _zeros_like_dtype(shape: list[int], reference: ttnn.Tensor) -> ttnn.Tensor:
    """Allocate a zero tensor whose dtype / device / layout match `reference`."""
    return ttnn.zeros(
        shape,
        dtype=reference.dtype,
        layout=reference.layout,
        device=reference.device(),
        memory_config=reference.memory_config(),
    )


def add_shared_expert_weights(
    routed_w0: ttnn.Tensor,  # sharded on dim 1, local (L, E_per_dev, H, N)
    routed_w1: ttnn.Tensor,
    routed_w2: ttnn.Tensor,  # local (L, E_per_dev, N, H)
    shared_w0: ttnn.Tensor,  # sharded on dim 1, local (L, S_per_dev, H, N)
    shared_w1: ttnn.Tensor,
    shared_w2: ttnn.Tensor,  # local (L, S_per_dev, N, H)
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Append per-device shared experts after routed experts along the experts dim.

    Inputs are multi-device tensors sharded on dim 1 (experts). Each device's
    routed shard holds its assigned routed experts; each device's shared shard
    holds its assigned shared experts, already in the correct slot order. The
    helper just concatenates the two along dim 1.

    Callers own the device → shared-expert mapping (see
    ``shared_expert_ids_to_device`` semantics in the torch reference module)
    and are responsible for producing the pre-arranged ``shared_w*`` tensors.
    """
    output_w0 = ttnn.concat([routed_w0, shared_w0], dim=1)
    output_w1 = ttnn.concat([routed_w1, shared_w1], dim=1)
    output_w2 = ttnn.concat([routed_w2, shared_w2], dim=1)
    return output_w0, output_w1, output_w2


def prepare_w0_w1_tensor_for_moe_compute(
    tt_w0: ttnn.Tensor,
    tt_w1: ttnn.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
) -> ttnn.Tensor:
    """ttnn port of ``moe_compute_utils.prepare_w0_w1_tensor_for_moe_compute``.

    See the torch version for the full layout contract. Inputs / output are
    ``ttnn.Tensor``; otherwise the byte layout is identical.
    """
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Nt = N // ttnn.TILE_SIZE
    # Pad K up to a multiple of (TILE_SIZE * BLOCK_TILES_H) — matches the DRAM read transaction.
    Kp = math.ceil(K // ttnn.TILE_SIZE / BLOCK_TILES_H) * ttnn.TILE_SIZE * BLOCK_TILES_H
    num_cores = len(shard_map)

    if K < Kp:
        padding = _zeros_like_dtype([L, E, Kp - K, N], tt_w0)
        working_w0 = ttnn.concat([tt_w0, padding], dim=2)
        working_w1 = ttnn.concat([tt_w1, padding], dim=2)
    else:
        working_w0 = tt_w0
        working_w1 = tt_w1

    # (L, E, Kp, N) -> (L, E, Kp, Nt, TILE_SIZE)
    w0_chunks = ttnn.reshape(working_w0, [L, E, Kp, Nt, ttnn.TILE_SIZE])
    w1_chunks = ttnn.reshape(working_w1, [L, E, Kp, Nt, ttnn.TILE_SIZE])

    # Stack w0 / w1 along a new axis 4 so the two ops alternate: (L, E, Kp, Nt, 2, TILE_SIZE).
    stacked = _stack([w0_chunks, w1_chunks], dim=4)
    interleaved = ttnn.reshape(stacked, [L, E, Kp, Nt, 2 * ttnn.TILE_SIZE])

    # Move Nt before Kp so we can slice along the Nt dim per core: (L, E, Nt, Kp, 2*TILE_SIZE).
    permuted = ttnn.permute(interleaved, [0, 1, 3, 2, 4])

    # Validate shard distribution and pick the padded shard size.
    max_shard_size = max(shard_map)
    max_shard_size = max_shard_size + (max_shard_size % 2)  # round up to even
    if any(x not in [max_shard_size, max_shard_size - 1, max_shard_size - 2] for x in shard_map):
        raise RuntimeError(
            f"W0W1 shard sizes must be in [{max_shard_size - 2}, {max_shard_size}] "
            f"(after rounding max to even), got: {shard_map}"
        )

    each_shard: list[ttnn.Tensor] = []
    start_tile = 0
    for num_tiles in shard_map:
        each_shard.append(
            ttnn.slice(permuted, [0, 0, start_tile, 0, 0], [L, E, start_tile + num_tiles, Kp, 2 * ttnn.TILE_SIZE])
        )
        pad_tiles = max_shard_size - num_tiles
        if pad_tiles > 0:
            each_shard.append(_zeros_like_dtype([L, E, pad_tiles, Kp, 2 * ttnn.TILE_SIZE], permuted))
        start_tile += num_tiles

    reordered = ttnn.concat(each_shard, dim=2)

    # Split the Nt axis into (num_cores, max_shard_size) and bring num_cores to the front:
    # (L, E, num_cores * max_shard_size, Kp, 2*TILE_SIZE) -> (num_cores, L, E, max_shard_size, Kp, 2*TILE_SIZE)
    all_groups = ttnn.reshape(reordered, [L, E, num_cores, max_shard_size, Kp, 2 * ttnn.TILE_SIZE])
    all_groups = ttnn.permute(all_groups, [2, 0, 1, 3, 4, 5])

    # Pair adjacent (w0, w1) tiles so each group is 2 along the inner expert axis, 4 tiles wide
    # in the trailing dim. (num_cores, L, E, groups_per_core, 2, Kp, 2*TILE_SIZE)
    groups_per_core = max_shard_size // 2
    paired = ttnn.reshape(all_groups, [num_cores, L, E, groups_per_core, 2, Kp, 2 * ttnn.TILE_SIZE])
    paired = ttnn.permute(paired, [0, 1, 2, 3, 5, 4, 6])
    return ttnn.reshape(paired, [num_cores, L, E, groups_per_core, Kp, 4 * ttnn.TILE_SIZE])


def prepare_w2_tensor_for_moe_compute(
    tt_w2: ttnn.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> ttnn.Tensor:
    """ttnn port of ``moe_compute_utils.prepare_w2_tensor_for_moe_compute``.

    Includes the ring-rotated N-tile reordering. The ``core_chunk_order``
    cycling is handled per-core in Python; each core's reordering is built
    via slice / concat on the ttnn tensor.
    """
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    # K-column shard: take the first (groups_per_core - 1) blocks of 4 tiles, then the last
    # block whose size comes from w2_shard_map, then optional zero padding.
    each_shard: list[ttnn.Tensor] = []
    start_col = 0
    full_block_width = (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        if full_block_width > 0:
            each_shard.append(ttnn.slice(tt_w2, [0, 0, 0, start_col], [L, E, N, start_col + full_block_width]))
            start_col += full_block_width
        each_shard.append(
            ttnn.slice(tt_w2, [0, 0, 0, start_col], [L, E, N, start_col + last_group_tiles * ttnn.TILE_SIZE])
        )
        start_col += last_group_tiles * ttnn.TILE_SIZE
        if last_group_pad_tiles > 0:
            each_shard.append(_zeros_like_dtype([L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE], tt_w2))

    reordered = ttnn.concat(each_shard, dim=-1)
    # (L, E, N, num_cores, groups_per_core, 4*TILE) -> (num_cores, L, E, groups_per_core, N, 4*TILE)
    grouped_per_bank = ttnn.reshape(reordered, [L, E, N, num_cores, w2_groups_per_core, 4 * ttnn.TILE_SIZE])
    grouped_per_bank = ttnn.permute(grouped_per_bank, [3, 0, 1, 4, 2, 5])

    # Split N into (Nt, TILE_SIZE) so we can pick whole-tile chunks per ring position:
    # (num_cores, L, E, groups_per_core, Nt, TILE_SIZE, 4*TILE)
    n_grouped = ttnn.reshape(
        grouped_per_bank, [num_cores, L, E, w2_groups_per_core, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE]
    )

    # Per-core ring rotation of the Nt chunks (matches the torch version's
    # `core_chunk_order = list(reversed(range(num_cores))).roll(1)` pattern).
    chunk_start_positions: list[int] = [0]
    for s in w0_w1_shard_map:
        chunk_start_positions.append(chunk_start_positions[-1] + s)
    base_order = list(reversed(range(num_cores)))
    base_order = base_order[-1:] + base_order[:-1]  # `.roll(1)`

    per_core_shards: list[ttnn.Tensor] = []
    current_order = base_order
    for core_id in range(num_cores):
        # ttnn.slice on the leading axis to grab this core's slab, then assemble Nt chunks.
        core_slab = ttnn.slice(
            n_grouped,
            [core_id, 0, 0, 0, 0, 0, 0],
            [core_id + 1, L, E, w2_groups_per_core, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE],
        )
        chunks: list[ttnn.Tensor] = []
        for chunk_id in current_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            chunks.append(
                ttnn.slice(
                    core_slab,
                    [0, 0, 0, 0, start_pos, 0, 0],
                    [1, L, E, w2_groups_per_core, end_pos, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE],
                )
            )
        per_core_shards.append(ttnn.concat(chunks, dim=4))
        # Rotate: equivalent of `core_chunk_order.roll(1)`.
        current_order = current_order[-1:] + current_order[:-1]

    stacked = ttnn.concat(per_core_shards, dim=0)  # (num_cores, 1, L, E, groups_per_core, Nt, TILE_SIZE, 4*TILE)
    n_reordered = ttnn.reshape(stacked, [num_cores, L, E, w2_groups_per_core, Nt * ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE])

    # Pad N up to a multiple of BLOCK_TILES_H tiles for the 7-tile DRAM reads.
    n_padded_tiles = math.ceil(Nt / BLOCK_TILES_H) * BLOCK_TILES_H
    n_padding = n_padded_tiles * ttnn.TILE_SIZE - N
    if n_padding > 0:
        pad = _zeros_like_dtype([num_cores, L, E, w2_groups_per_core, n_padding, 4 * ttnn.TILE_SIZE], tt_w2)
        n_reordered = ttnn.concat([n_reordered, pad], dim=4)

    return n_reordered


def prepare_w0_w1_tensor_with_bias(
    tt_w0: ttnn.Tensor,
    tt_w1: ttnn.Tensor,
    tt_b0: ttnn.Tensor,
    tt_b1: ttnn.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
) -> ttnn.Tensor:
    """ttnn port of ``moe_compute_utils.prepare_w0_w1_tensor_with_bias``.

    Converts true PyTorch bias (L, E, N) to kernel tile format (L, E, 32, N) with
    only row 0 populated, concatenates with W along K, then delegates to the
    no-bias prepare function.
    """
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    K_with_bias = (K // ttnn.TILE_SIZE + 1) * ttnn.TILE_SIZE

    # Tile-format bias: (L, E, N) -> (L, E, TILE_SIZE, N) with row 0 populated.
    b0_row = ttnn.unsqueeze(tt_b0, 2)  # (L, E, 1, N)
    b1_row = ttnn.unsqueeze(tt_b1, 2)
    pad_rows = _zeros_like_dtype([L, E, ttnn.TILE_SIZE - 1, N], tt_b0)
    b0_tiled = ttnn.concat([b0_row, pad_rows], dim=2)
    b1_tiled = ttnn.concat([b1_row, pad_rows], dim=2)

    w0_b0 = ttnn.concat([tt_w0, b0_tiled], dim=2)  # (L, E, K + TILE_SIZE, N)
    w1_b1 = ttnn.concat([tt_w1, b1_tiled], dim=2)

    return prepare_w0_w1_tensor_for_moe_compute(w0_b0, w1_b1, L, E, K_with_bias, N, shard_map)


def prepare_w2_tensor_with_bias(
    tt_w2: ttnn.Tensor,
    tt_b2: ttnn.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> ttnn.Tensor:
    """ttnn port of ``moe_compute_utils.prepare_w2_tensor_with_bias``.

    The bias tile row stays fixed (not ring-rotated). We reuse the no-bias W2
    preparer for the rotated weight tiles, then column-shard the bias the same
    way, concatenate along N, and pad N+TILE_SIZE up to a BLOCK_TILES_H multiple.
    """
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    # 1) Ring-rotated W2 (without bias) at the kernel's expected layout.
    # We call the unpadded form here — the bias concat below adds back the final pad.
    n_reordered_no_pad = _prepare_w2_no_n_pad(tt_w2, L, E, N, K, w2_shard_map, w0_w1_shard_map)

    # 2) Bias tile row: (L, E, K) -> (L, E, TILE_SIZE, K) with row 0 populated, then column-shard.
    b2_row = ttnn.unsqueeze(tt_b2, 2)  # (L, E, 1, K)
    pad_rows = _zeros_like_dtype([L, E, ttnn.TILE_SIZE - 1, K], tt_b2)
    b2_tiled = ttnn.concat([b2_row, pad_rows], dim=2)

    b2_each_shard: list[ttnn.Tensor] = []
    start_col = 0
    full_block_width = (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        if full_block_width > 0:
            b2_each_shard.append(
                ttnn.slice(b2_tiled, [0, 0, 0, start_col], [L, E, ttnn.TILE_SIZE, start_col + full_block_width])
            )
            start_col += full_block_width
        b2_each_shard.append(
            ttnn.slice(
                b2_tiled, [0, 0, 0, start_col], [L, E, ttnn.TILE_SIZE, start_col + last_group_tiles * ttnn.TILE_SIZE]
            )
        )
        start_col += last_group_tiles * ttnn.TILE_SIZE
        if last_group_pad_tiles > 0:
            b2_each_shard.append(
                _zeros_like_dtype([L, E, ttnn.TILE_SIZE, last_group_pad_tiles * ttnn.TILE_SIZE], tt_b2)
            )

    b2_reordered = ttnn.concat(b2_each_shard, dim=-1)
    b2_grouped = ttnn.reshape(b2_reordered, [L, E, ttnn.TILE_SIZE, num_cores, w2_groups_per_core, 4 * ttnn.TILE_SIZE])
    b2_grouped = ttnn.permute(b2_grouped, [3, 0, 1, 4, 2, 5])  # (num_cores, L, E, groups, TILE_SIZE, 4*TILE)

    # 3) Concat bias row after weight tiles (NOT rotated).
    n_with_bias = ttnn.concat([n_reordered_no_pad, b2_grouped], dim=4)  # adds TILE_SIZE rows along N

    # 4) Pad to BLOCK_TILES_H tile multiple along N.
    n_total_tiles = Nt + 1
    n_target_tiles = math.ceil(n_total_tiles / BLOCK_TILES_H) * BLOCK_TILES_H
    n_padding = (n_target_tiles - n_total_tiles) * ttnn.TILE_SIZE
    if n_padding > 0:
        pad = _zeros_like_dtype([num_cores, L, E, w2_groups_per_core, n_padding, 4 * ttnn.TILE_SIZE], tt_w2)
        n_with_bias = ttnn.concat([n_with_bias, pad], dim=4)

    return n_with_bias


def _prepare_w2_no_n_pad(
    tt_w2: ttnn.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> ttnn.Tensor:
    """Same as `prepare_w2_tensor_for_moe_compute` but skips the trailing N pad.

    Used by the bias-aware path so the bias tile row can be concatenated before
    the alignment pad is applied.
    """
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    each_shard: list[ttnn.Tensor] = []
    start_col = 0
    full_block_width = (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        if full_block_width > 0:
            each_shard.append(ttnn.slice(tt_w2, [0, 0, 0, start_col], [L, E, N, start_col + full_block_width]))
            start_col += full_block_width
        each_shard.append(
            ttnn.slice(tt_w2, [0, 0, 0, start_col], [L, E, N, start_col + last_group_tiles * ttnn.TILE_SIZE])
        )
        start_col += last_group_tiles * ttnn.TILE_SIZE
        if last_group_pad_tiles > 0:
            each_shard.append(_zeros_like_dtype([L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE], tt_w2))

    reordered = ttnn.concat(each_shard, dim=-1)
    grouped_per_bank = ttnn.reshape(reordered, [L, E, N, num_cores, w2_groups_per_core, 4 * ttnn.TILE_SIZE])
    grouped_per_bank = ttnn.permute(grouped_per_bank, [3, 0, 1, 4, 2, 5])
    n_grouped = ttnn.reshape(
        grouped_per_bank, [num_cores, L, E, w2_groups_per_core, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE]
    )

    chunk_start_positions: list[int] = [0]
    for s in w0_w1_shard_map:
        chunk_start_positions.append(chunk_start_positions[-1] + s)
    base_order = list(reversed(range(num_cores)))
    base_order = base_order[-1:] + base_order[:-1]

    per_core_shards: list[ttnn.Tensor] = []
    current_order = base_order
    for core_id in range(num_cores):
        core_slab = ttnn.slice(
            n_grouped,
            [core_id, 0, 0, 0, 0, 0, 0],
            [core_id + 1, L, E, w2_groups_per_core, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE],
        )
        chunks: list[ttnn.Tensor] = []
        for chunk_id in current_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            chunks.append(
                ttnn.slice(
                    core_slab,
                    [0, 0, 0, 0, start_pos, 0, 0],
                    [1, L, E, w2_groups_per_core, end_pos, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE],
                )
            )
        per_core_shards.append(ttnn.concat(chunks, dim=4))
        current_order = current_order[-1:] + current_order[:-1]

    stacked = ttnn.concat(per_core_shards, dim=0)
    return ttnn.reshape(stacked, [num_cores, L, E, w2_groups_per_core, Nt * ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE])
