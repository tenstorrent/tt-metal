// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "fused_single_user_dot_program_factory.hpp"
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
    using program_factory_t = std::variant<FusedSingleUserProgramFactory, FusedSingleUserDotProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

// Single-user decode implementation. The first eight cores consume width shards of
// hidden_streams and produce collapsed, core 8 computes post, and core 9 computes comb
// plus Sinkhorn. fused_w is broadcast from core 0 to all ten participating cores. With pre_mix,
// the collapse cores weight the streams by it instead of the pre derived from fused_w, and core 8
// also computes that pre and returns it as a fourth output.
//
// Given fn_weight in place of fused_w, the same program also computes fused_w: the cores of dot_core_grid each
// dot their K-slice of the RMS-normed streams, read straight out of the collapse cores' shards, with their slab of
// fn_weight, and the partials are summed up a fan-in-reduce_fan_in tree whose root pushes fused_w to the cores
// that read it.
std::vector<Tensor> fused_hyperconnection_single_user(
    const std::optional<Tensor>& fused_w,
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
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& pre_mix = std::nullopt,
    std::optional<Layout> collapsed_layout = std::nullopt,
    const std::optional<Tensor>& fn_weight = std::nullopt,
    const std::optional<CoreRangeSet>& dot_core_grid = std::nullopt,
    uint32_t reduce_fan_in = 2,
    std::optional<float> rms_norm_epsilon = std::nullopt);

}  // namespace ttnn::prim
