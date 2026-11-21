// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "sdpa_decode_bh_device_operation_types.hpp"

namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail {

constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS = 6;

struct TreeReductionParams {
    uint32_t num_rounds = 0;
    bool is_root = false;
    uint32_t parent_core_in_group = 0;
    uint32_t send_at_round = 0;
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
};

inline TreeReductionParams get_tree_reduction_params(uint32_t core_id, uint32_t N) {
    TreeReductionParams p{};
    std::fill_n(p.children_per_round, MAX_TREE_REDUCTION_ROUNDS, UINT32_MAX);
    p.parent_core_in_group = p.send_at_round = UINT32_MAX;

    if (N <= 1) {
        p.is_root = true;
        return p;
    }

    uint32_t vid = (N - 1) - core_id;
    p.num_rounds = 32 - __builtin_clz(N - 1);
    p.is_root = (core_id == 0);

    for (uint32_t r = 0; r < p.num_rounds; r++) {
        uint32_t mask = (2u << r) - 1;
        if ((vid & mask) == mask) {
            uint32_t child_vid = vid - (1u << r);
            if (child_vid < N) {
                p.children_per_round[r] = (N - 1) - child_vid;
            }
        }
    }

    if (!p.is_root) {
        uint32_t trailing_ones = __builtin_ctz(~vid);
        uint32_t parent_vid = vid + (1u << trailing_ones);

        p.parent_core_in_group = (parent_vid < N) ? (N - 1) - parent_vid : 0;
        p.send_at_round = trailing_ones;
    }

    if (p.is_root) {
        for (uint32_t c = 1; c < N; c++) {
            uint32_t cv = (N - 1) - c;
            uint32_t t = __builtin_ctz(~cv);
            if (cv + (1u << t) >= N && p.children_per_round[t] == UINT32_MAX) {
                p.children_per_round[t] = c;
            }
        }
    }
    return p;
}

struct SdpaDecodeDeviceOperation {
    using operation_attributes_t = SdpaDecodeParams;
    using tensor_args_t = SdpaDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& page_table_tensor,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& attention_sink,
    bool is_causal,
    bool paged_attention,
    const std::vector<uint32_t>& cur_pos,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size,
    std::optional<bool> share_cache,
    std::optional<bool> use_mla,
    std::optional<uint32_t> head_dim_v,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt,
    std::optional<uint32_t> cache_position_modulo = std::nullopt,
    uint32_t spec_multi_pos_tiles = 0,
    bool multi_pos = false);

}  // namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail
