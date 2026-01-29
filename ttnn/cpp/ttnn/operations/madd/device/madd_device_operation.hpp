// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"                                        // Tensor
#include "ttnn/tensor/tensor_spec.hpp"                                   // TensorSpec
#include "ttnn/operations/core/core.hpp"                                 // MemoryConfig, DeviceComputeKernelConfig
#include "ttnn/operations/madd/device/madd_device_operation_types.hpp"   // MAddParams, MAddArgs
#include "ttnn/operations/madd/device/madd_program_factory.hpp"          // MAddProgramFactory
#include "ttnn/operations/madd/device/madd_program_factory_sharded.hpp"  // MAddProgramFactorySharded

namespace ttnn::prim {

struct MAddOperation {
    using operation_attributes_t = MAddParams;
    using tensor_args_t = MAddArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<MAddProgramFactory, MAddProgramFactorySharded>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

ttnn::Tensor madd(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& c,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);
}  // namespace ttnn::prim
