// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h"
#include "ttnn/operations/experimental/ccl/moe_compute/device/hostdevcommon/config.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeParams {
    // MoE compute attributes
    uint32_t layer_id;
    uint32_t output_height_shard_dim;
    bool has_bias;
    SelectiveReduceCombineParams combine_params;
    ttnn::experimental::prim::detail::MoEActivationFunction activation_type =
        ttnn::experimental::prim::detail::MoEActivationFunction::SILU;  // Default to SILU

    // Same value as combine_params.axis (single source of truth)
    std::optional<uint32_t> cluster_axis() const { return combine_params.axis; }

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("output_height_shard_dim", output_height_shard_dim);
        attrs.emplace_back("has_bias", has_bias);
        attrs.emplace_back("combine_params", combine_params);
        attrs.emplace_back("activation_type", static_cast<uint32_t>(activation_type));
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
