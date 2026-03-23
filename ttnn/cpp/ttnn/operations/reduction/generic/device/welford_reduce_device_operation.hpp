// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "welford_reduce_device_operation_types.hpp"
#include "welford_reduce_program_factory.hpp"

namespace ttnn::prim {

struct WelfordReduceDeviceOperation {
    using operation_attributes_t = WelfordReduceParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<WelfordReduceProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

ttnn::Tensor welford_reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool correction,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim
