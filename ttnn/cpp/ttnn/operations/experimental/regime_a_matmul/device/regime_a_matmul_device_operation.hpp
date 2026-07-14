// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "regime_a_matmul_device_operation_types.hpp"
#include "regime_a_matmul_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RegimeAMatmulDeviceOperation {
    using operation_attributes_t = RegimeAMatmulParams;
    using tensor_args_t = RegimeAMatmulInputs;
    using spec_return_value_t = TensorSpec;  // single output (NOT a vector)
    using tensor_return_value_t = Tensor;

    // Single program factory in the variant. The framework auto-selects it (no custom
    // select_program_factory required — see ttnn/operation_concepts.hpp: a single-alternative
    // program_factory_t is returned automatically).
    using program_factory_t = std::variant<RegimeAMatmulProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const std::optional<const RegimeAMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor regime_a_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::prim
