// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "gdn_fused_device_operation_types.hpp"
#include "gdn_fused_program_factory.hpp"

namespace ttnn::experimental::prim {

struct GdnFusedDeviceOperation {
    using operation_attributes_t = GdnFusedParams;
    using tensor_args_t = GdnFusedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GdnFusedProgramFactory, GdnFusedMeshWorkloadFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::GdnFusedDeviceOperation::tensor_return_value_t gdn_fused(
    const Tensor& conv_out,
    const Tensor& a_fused,
    const Tensor& b_fused,
    const Tensor& neg_exp_A,
    const Tensor& dt_bias,
    const Tensor& norm_w,
    const Tensor& scale_tt,
    const Tensor& rms_scale_tt,
    const Tensor& rms_eps_tt,
    const Tensor& state,
    const Tensor& output,
    uint32_t num_pairs,
    uint32_t num_cores,
    uint32_t Nv_TP,
    uint32_t Nk_TP,
    uint32_t repeat_factor,
    uint32_t key_dim_tp);

}  // namespace ttnn::prim
