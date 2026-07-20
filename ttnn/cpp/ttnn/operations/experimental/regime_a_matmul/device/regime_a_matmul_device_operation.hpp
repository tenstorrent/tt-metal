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
    // Vector-valued to support output column-splitting (regime_a_matmul_split). chunks==1 (the default /
    // public regime_a_matmul path) yields a single-element vector, mirroring minimal_matmul.
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

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
        std::optional<DeviceComputeKernelConfig> compute_kernel_config,
        const std::optional<Tensor>& bias_tensor = std::nullopt,
        std::optional<operations::unary::UnaryWithParam> fused_activation = std::nullopt,
        std::optional<float> fused_ternary_scalar = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
        int32_t chunks = 1,
        int32_t dim = -1,
        uint32_t diag_mask = 0);  // test-only ablations (RegimeADiag); public path always 0
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Returns a vector of output tensors (chunks). chunks==1 => single element (public regime_a_matmul).
std::vector<Tensor> regime_a_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& bias_tensor = std::nullopt,
    std::optional<operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
    int32_t chunks = 1,
    int32_t dim = -1);

// Test-only / internal entry point that constructs the primitive with a nonzero diagnostic ablation mask
// (RegimeADiag bits). NOT bound to Python/nanobind; used by the C++ ablation harness only. The public
// regime_a_matmul() above always runs with mask 0. Returns the single (chunks==1) output tensor.
Tensor regime_a_matmul_diag(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t diag_mask);

}  // namespace ttnn::prim
