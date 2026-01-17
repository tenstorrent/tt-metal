// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "minimal_matmul_device_operation_types.hpp"
#include "minimal_matmul_program_factory.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulDeviceOperation {
    using operation_attributes_t = MinimalMatmulParams;
    using tensor_args_t = MinimalMatmulInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<MinimalMatmulProgramFactory>;

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
        const std::optional<Tensor>& bias_tensor,
        std::optional<operations::unary::UnaryWithParam> fused_activation,
        const std::optional<const MinimalMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor minimal_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::prim
