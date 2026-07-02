// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/types.hpp"

// C++ port of the ttnn-tensor helpers in
// ``ttnn/_experimental/moe_compute_utils.py``. These produce the exact
// byte layout the ``ttnn.experimental.moe_compute`` kernels expect.
//
// Pure-logic helpers (``cluster_distance``, ``get_shared_experts_per_device``,
// ``_shard_tiles``, ``_w2_shard_tiles``, ``auto_output_width_shard_dim``)
// continue to live in the Python module ``ttnn._experimental.moe_compute_utils``
// and are imported by callers as-is.
//
// All inputs are expected to be multi-device tensors sharded on dim 1 (the
// experts dim); see the Python module docstring for the full layout contract.

namespace ttnn::experimental {

// Per-ring-position shard maps for W0/W1 and W2 weight tensors, plus the
// DRAM-bank CoreRangeSet that callers pass to ``ttnn.MemoryConfig`` for the
// packed weight tensors.
struct WeightCoreShardMaps {
    std::vector<uint32_t> w0_w1_shard_map;
    std::vector<std::pair<uint32_t, uint32_t>> w2_shard_map;
    ttnn::CoreRangeSet dram_core_range_set;
};

// Compute the per-ring-position shard maps and DRAM-bank CoreRangeSet the MoE
// compute op expects. Uses ``shard_tiles`` (Euclidean rhythm) for W0/W1 and
// ``w2_shard_tiles`` (complementary when ``Nt%n_cores + Ht%n_cores == n_cores``)
// for W2. Ring ordering: DRAM bank logical coords sorted by ``(y, x)`` descending.
// The matmul ring size is auto-detected from the device arch (8 on Blackhole,
// 12 — the DRAM-bank count — on Wormhole), matching ``ttnn.experimental.moe_compute``.
WeightCoreShardMaps get_weight_core_shard_maps(
    ttnn::MeshDevice* mesh_device, uint32_t hidden_size, uint32_t intermediate_size);

// DRAM-sharded memory configs for the packed W0/W1 and W2 weight tensors. The
// shard maps and DRAM-bank CoreRangeSet are computed internally to match
// ``get_weight_core_shard_maps``.
struct WeightMemoryConfigs {
    ttnn::MemoryConfig w0_w1;
    ttnn::MemoryConfig w2;
};

// Build the DRAM-sharded memory configs the packed W0/W1 and W2 weight tensors
// must live under for the MoE compute kernel. When ``has_bias`` is true, the
// padded K/N dimensions grow by one tile and are re-aligned to BLOCK_TILES_H.
WeightMemoryConfigs get_weight_mem_configs(
    ttnn::MeshDevice* mesh_device,
    uint32_t num_layers,
    uint32_t experts_per_device,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    bool has_bias = false);

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
    const ttnn::Tensor& shared_w2,
    uint32_t cluster_axis);

// Pack W0/W1 into the interleaved, padded, per-core layout the MoE kernel reads.
// Output local shape: ``(num_cores, L, E, groups_per_core, K_padded, 4*TILE)``.
// The per-core shard map is derived internally from ``K`` (hidden_size) and
// ``N`` (intermediate_size) via ``get_weight_core_shard_maps``.
ttnn::Tensor prepare_w0_w1_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w0, const ttnn::Tensor& tt_w1, uint32_t L, uint32_t E, uint32_t K, uint32_t N);

// Pack W2 into the ring-rotated per-core layout the MoE kernel reads.
// Output local shape: ``(num_cores, L, E, w2_groups_per_core, N_padded, 4*TILE)``.
// The per-core shard maps are derived internally from ``K`` (hidden_size) and
// ``N`` (intermediate_size) via ``get_weight_core_shard_maps``.
ttnn::Tensor prepare_w2_tensor_for_moe_compute(
    const ttnn::Tensor& tt_w2, uint32_t L, uint32_t E, uint32_t N, uint32_t K);

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
    uint32_t N);

// Bias-aware W2 packer: weight tiles get ring-rotated as usual; bias tile row
// is column-sharded and concatenated along N **without** rotation, then N is
// padded to a multiple of BLOCK_TILES_H tiles.
ttnn::Tensor prepare_w2_tensor_with_bias(
    const ttnn::Tensor& tt_w2, const ttnn::Tensor& tt_b2, uint32_t L, uint32_t E, uint32_t N, uint32_t K);

// Round-trip a device tensor through host to change its dtype and optionally re-upload
// it under the supplied memory config. Used to quantize the packed weight
// tensors to ``bfloat4_b`` on the DRAM-sharded mem config the kernel consumes.
// Host side quantization is higher quality and maintains PCC with the original implementation
ttnn::Tensor quantize_weights_via_host(
    const ttnn::Tensor& device_tensor, ttnn::DataType dtype, const std::optional<ttnn::MemoryConfig>& memory_config);

}  // namespace ttnn::experimental
