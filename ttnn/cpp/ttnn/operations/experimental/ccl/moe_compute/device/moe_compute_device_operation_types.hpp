// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeParams {
    // MoE compute attributes
    uint32_t layer_id;
    uint32_t output_height_shard_dim;
    uint32_t output_width_shard_dim;
    SelectiveReduceCombineParams combine_params;

    // Same value as combine_params.axis (single source of truth)
    std::optional<uint32_t> cluster_axis() const { return combine_params.axis; }

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("output_height_shard_dim", output_height_shard_dim);
        attrs.emplace_back("output_width_shard_dim", output_width_shard_dim);
        attrs.emplace_back("combine_params", combine_params);
        return attrs;
    }
};

struct MoEComputeInputs {
    const ttnn::Tensor& tilize_input_tensor;
    const ttnn::Tensor& tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_expert_mapping_tensor;
    const ttnn::Tensor& matmul_w0_w1_tensor;
    const ttnn::Tensor& matmul_w2_tensor;
    const std::optional<ttnn::Tensor>& optional_output_tensor;
};

}  // namespace ttnn::experimental::prim
