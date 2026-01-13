// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "minimal_matmul_split_device_operation_types.hpp"
#include "minimal_matmul_split_program_factory.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct MinimalMatmulSplitDeviceOperation {
    using operation_attributes_t = ttnn::operations::experimental::minimal_matmul::split_operation_attributes_t;
    using tensor_args_t = ttnn::operations::experimental::minimal_matmul::split_tensor_args_t;
    using spec_return_value_t = ttnn::operations::experimental::minimal_matmul::split_spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::experimental::minimal_matmul::split_tensor_return_value_t;

    using program_factory_t = std::variant<program::MinimalMatmulSplitProgramFactory>;

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

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const int32_t chunks,
        const int32_t dim,
        const std::optional<Tensor>& bias_tensor,
        std::optional<unary::UnaryWithParam> fused_activation,
        const std::optional<const MinimalMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::minimal_matmul

namespace ttnn::prim {

operations::experimental::minimal_matmul::MinimalMatmulSplitDeviceOperation::tensor_return_value_t minimal_matmul_split(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const int32_t chunks,
    const int32_t dim,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const operations::experimental::minimal_matmul::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::prim
