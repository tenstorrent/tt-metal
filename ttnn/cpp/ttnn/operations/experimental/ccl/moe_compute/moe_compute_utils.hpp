// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/types.hpp"

// C++ port of the ttnn-tensor helpers in
// ``ttnn/_experimental/moe_compute_utils_tt.py``. These produce the exact
// byte layout the ``ttnn.experimental.moe_compute`` kernels expect.
//
// Pure-logic helpers (``cluster_distance``, ``get_shared_experts_per_device``,
// ``_shard_tiles``, ``_w2_shard_tiles``, ``auto_output_width_shard_dim``,
// ``get_weight_core_shard_maps``, ``get_weight_mem_configs``) continue to live
// in the Python module ``ttnn._experimental.moe_compute_utils`` and are
// imported by callers as-is.
//
// All inputs are expected to be multi-device tensors sharded on dim 1 (the
// experts dim); see the Python module docstring for the full layout contract.

namespace ttnn::experimental {

// Append per-device shared experts after routed experts along the experts dim.
//
// Inputs are multi-device tensors sharded on dim 1 (experts). Each device's
// routed shard holds its assigned routed experts; each device's shared shard
// holds its assigned shared experts, already in the correct slot order.
// Callers own the device → shared-expert mapping and produce the pre-arranged
// ``shared_w*`` tensors.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> add_shared_expert_weights(
    const ttnn::Tensor& routed_w0,
    const ttnn::Tensor& routed_w1,
    const ttnn::Tensor& routed_w2,
    const ttnn::Tensor& shared_w0,
    const ttnn::Tensor& shared_w1,
    const ttnn::Tensor& shared_w2);

// Pack W0/W1 into the interleaved, padded, per-core layout the MoE kernel reads.
// Output local shape: ``(num_cores, L, E, groups_per_core, K_padded, 4*TILE)``.
ttnn::Tensor prepare_w0_w1_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w0,
    const ttnn::Tensor& tt_w1,
    uint32_t L,
    uint32_t E,
    uint32_t K,
    uint32_t N,
    const std::vector<uint32_t>& shard_map);

// Pack W2 into the ring-rotated per-core layout the MoE kernel reads.
// Output local shape: ``(num_cores, L, E, w2_groups_per_core, N_padded, 4*TILE)``.
ttnn::Tensor prepare_w2_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w2,
    uint32_t L,
    uint32_t E,
    uint32_t N,
    uint32_t K,
    const std::vector<std::pair<uint32_t, uint32_t>>& w2_shard_map,
    const std::vector<uint32_t>& w0_w1_shard_map);

// Bias-aware W0/W1 packer: appends bias tiles along K, then delegates to the
// no-bias packer. Bias inputs are PyTorch-format ``(L, E, N)``; output matches
// the no-bias version with K_padded sized for ``K + TILE_SIZE`` rounded up.
ttnn::Tensor prepare_w0_w1_tensor_with_bias(
    const ttnn::Tensor& tt_w0,
    const ttnn::Tensor& tt_w1,
    const ttnn::Tensor& tt_b0,
    const ttnn::Tensor& tt_b1,
    uint32_t L,
    uint32_t E,
    uint32_t K,
    uint32_t N,
    const std::vector<uint32_t>& shard_map);

// Bias-aware W2 packer: weight tiles get ring-rotated as usual; bias tile row
// is column-sharded and concatenated along N **without** rotation, then N is
// padded to a multiple of BLOCK_TILES_H tiles.
ttnn::Tensor prepare_w2_tensor_with_bias(
    const ttnn::Tensor& tt_w2,
    const ttnn::Tensor& tt_b2,
    uint32_t L,
    uint32_t E,
    uint32_t N,
    uint32_t K,
    const std::vector<std::pair<uint32_t, uint32_t>>& w2_shard_map,
    const std::vector<uint32_t>& w0_w1_shard_map);

}  // namespace ttnn::experimental
