// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "unified_routed_expert_ffn_types.hpp"  // RoutedExpertActivation

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

using unified::RoutedExpertActivation;

struct HybridRoutedExpertFfnParams {
    // Shared shape. Both halves are built for the same per-expert M and the same local expert
    // count; x is the wider shared dispatched buffer and each expert is placed at its region
    // offset, exactly as in either op alone.
    uint32_t m_tiles = 0;
    uint32_t experts_per_chip = 1;
    bool x_is_row_major = false;
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    bool fuse_bias = false;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    // The measured per-model split, read from the device-resident counts vector: an expert with
    // `count <= hybrid_token_threshold` runs on the fused half, the rest on the unified one. This
    // is the whole reason the two implementations are one op -- the choice is per expert and the
    // counts are only known on device, so neither half can be skipped host-side.
    //
    // Zero means no expert reaches the fused half: the unified half owns the layer, the fused
    // pass is not run and its circular buffers are not placed. Both bodies are still compiled in,
    // so the merged binaries are the same shape either way.
    uint32_t hybrid_token_threshold = 0;

    uint32_t origin_y = 2;
    uint32_t grid_x = 11;
    uint32_t grid_y = 8;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "m_tiles",
        "experts_per_chip",
        "x_is_row_major",
        "activation",
        "fuse_bias",
        "compute_kernel_config",
        "hybrid_token_threshold",
        "origin_y",
        "grid_x",
        "grid_y");

    auto attribute_values() const {
        return std::forward_as_tuple(
            m_tiles,
            experts_per_chip,
            x_is_row_major,
            activation,
            fuse_bias,
            compute_kernel_config,
            hybrid_token_threshold,
            origin_y,
            grid_x,
            grid_y);
    }
};

// The union of both halves' inputs, which is just the unified half's list: the fused half consumes
// the same activations, weights, counts, index table, region offsets and biases, and writes into
// the same shared output.
struct HybridRoutedExpertFfnInputs {
    Tensor x;
    std::vector<Tensor> gate_projs;
    std::vector<Tensor> up_projs;
    std::vector<Tensor> down_projs;
    Tensor counts;
    Tensor global_expert_idx_table;
    Tensor output;
    std::optional<Tensor> expert_region_offsets;
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;

    // Per-core L1 scratch both halves' circular buffers are laid over. Required whenever pass A
    // runs: the two halves' buffers sum to more L1 than a core has, and overlaying them -- safe
    // because the passes are ordered, never concurrent -- is what lets both keep the whole grid.
    // Owned by the caller because the program keeps a raw pointer to it that must stay valid
    // across program-cache hits.
    std::optional<Tensor> l1_arena;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
