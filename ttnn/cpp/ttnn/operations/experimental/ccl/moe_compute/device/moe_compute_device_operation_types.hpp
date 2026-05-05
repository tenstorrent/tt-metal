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
    uint32_t layer_id = 0;
    uint32_t output_height_shard_dim = 0;
    bool has_bias = false;

    // Number of token-parallel and data-parallel cores. These govern matmul output shard layout
    // even in compute_only mode, so they live at the top level rather than only on combine_params.
    uint32_t num_token_parallel_cores = 0;
    uint32_t num_data_parallel_cores = 0;

    // When compute_only=true, the fused selective_reduce_combine path is bypassed entirely:
    // - combine_params is std::nullopt
    // - no combine cores allocated, no fabric setup, no global semaphores created
    // - op returns 5 tensors instead of 6 (matmul_output is the final output)
    bool compute_only = false;

    // nullopt iff compute_only=true
    std::optional<SelectiveReduceCombineParams> combine_params;

    ttnn::experimental::prim::detail::MoEActivationFunction activation_type =
        ttnn::experimental::prim::detail::MoEActivationFunction::SILU;  // Default to SILU

    // Same value as combine_params->axis (single source of truth when combine_params is set)
    std::optional<uint32_t> cluster_axis() const {
        return combine_params.has_value() ? combine_params->axis : std::nullopt;
    }

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("output_height_shard_dim", output_height_shard_dim);
        attrs.emplace_back("has_bias", has_bias);
        attrs.emplace_back("num_token_parallel_cores", num_token_parallel_cores);
        attrs.emplace_back("num_data_parallel_cores", num_data_parallel_cores);
        attrs.emplace_back("compute_only", compute_only);
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
