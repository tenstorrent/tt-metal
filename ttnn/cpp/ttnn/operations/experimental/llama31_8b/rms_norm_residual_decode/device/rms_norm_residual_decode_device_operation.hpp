// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::llama31_8b_rms_norm_residual_decode {

struct RmsNormResidualDecodeParams {
    float eps = 0.0f;
    MemoryConfig output_mem_config;
    LayerNormShardedMultiCoreProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct RmsNormResidualDecodeInputs {
    Tensor input;
    Tensor residual;
    Tensor weight;
};

bool in_table(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& weight,
    const LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config);

struct RmsNormResidualDecodeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RmsNormResidualDecodeParams& operation_attributes,
        const RmsNormResidualDecodeInputs& tensor_args,
        Tensor& tensor_return_value);
};

struct RmsNormResidualDecodeDeviceOperation {
    using operation_attributes_t = RmsNormResidualDecodeParams;
    using tensor_args_t = RmsNormResidualDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RmsNormResidualDecodeProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::prim::llama31_8b_rms_norm_residual_decode
