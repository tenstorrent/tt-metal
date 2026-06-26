// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "fused_pre_post_program_factory.hpp"
#include "fused_pre_post_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedPrePostDeviceOperation {
    using operation_attributes_t = FusedPrePostParams;
    using tensor_args_t = FusedPrePostInputs;
    using spec_return_value_t = FusedPrePostSpecReturn;
    using tensor_return_value_t = FusedPrePostTensorReturn;
    using program_factory_t = std::variant<FusedPrePostProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

// Decode-only (T == 1) fused stage. Returns {post, collapsed}:
//   post      = 2 * sigmoid(post_w * post_scale + post_bias)              [1,1,1,H]
//   collapsed = (sigmoid(pre_w * pre_scale + pre_bias) + eps) @ hidden    [1,1,1,D]
std::array<Tensor, 2> fused_hyperconnection_pre_post(
    const Tensor& pre_w,
    const Tensor& post_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& hidden_streams,
    float pre_scale,
    float post_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
