// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "minimal_matmul_device_operation_types.hpp"
#include "minimal_matmul_program_factory.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct MinimalMatmulDeviceOperation {
    using operation_attributes_t = ttnn::operations::experimental::minimal_matmul::operation_attributes_t;
    using tensor_args_t = ttnn::operations::experimental::minimal_matmul::tensor_args_t;
    using spec_return_value_t = ttnn::operations::experimental::minimal_matmul::spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::experimental::minimal_matmul::tensor_return_value_t;

    using program_factory_t = std::variant<program::MinimalMatmulProgramFactory>;

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
        std::optional<unary::UnaryWithParam> fused_activation,
        const std::optional<const MinimalMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::minimal_matmul

namespace ttnn::prim {
constexpr auto minimal_matmul = ttnn::register_operation<
    "ttnn::prim::minimal_matmul",
    ttnn::operations::experimental::minimal_matmul::MinimalMatmulDeviceOperation>();
}
