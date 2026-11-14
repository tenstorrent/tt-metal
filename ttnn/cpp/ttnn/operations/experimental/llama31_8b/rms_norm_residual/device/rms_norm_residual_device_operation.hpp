// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

struct RmsNormResidualProgramFactory;

bool rms_norm_residual_fits(const Tensor& input, const Tensor& residual);

struct RmsNormResidualOperation {
    struct operation_attributes_t {
        float eps = 0.0f;
        DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& residual;
        const Tensor& weight;
        const Tensor& residual_out;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RmsNormResidualProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct RmsNormResidualProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RmsNormResidualOperation::operation_attributes_t& operation_attributes,
        const RmsNormResidualOperation::tensor_args_t& tensor_args,
        RmsNormResidualOperation::tensor_return_value_t& output);
};

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_rms_norm_residual(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& weight,
    const Tensor& residual_out,
    float eps,
    const DeviceComputeKernelConfig& compute_kernel_config);
}  // namespace ttnn::prim
