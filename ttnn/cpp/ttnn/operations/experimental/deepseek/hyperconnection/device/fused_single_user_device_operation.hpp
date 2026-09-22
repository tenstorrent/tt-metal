// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "fused_single_user_program_factory.hpp"
#include "fused_single_user_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedSingleUserDeviceOperation {
    using operation_attributes_t = FusedSingleUserParams;
    using tensor_args_t = FusedSingleUserInputs;
    using spec_return_value_t = FusedSingleUserSpecReturn;
    using tensor_return_value_t = FusedSingleUserTensorReturn;
    using program_factory_t = std::variant<FusedSingleUserProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

// Single-user decode implementation. The first eight cores consume width shards of
// hidden_streams and produce collapsed, core 8 computes post, and core 9 computes comb
// plus Sinkhorn. fused_w is broadcast from core 0 to all ten participating cores.
std::array<Tensor, 3> fused_hyperconnection_single_user(
    const Tensor& fused_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    const Tensor& hidden_streams,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
