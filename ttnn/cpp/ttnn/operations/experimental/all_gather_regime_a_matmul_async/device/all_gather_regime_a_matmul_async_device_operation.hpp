// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "all_gather_regime_a_matmul_async_device_operation_types.hpp"
#include "all_gather_regime_a_matmul_async_program_factory.hpp"

namespace ttnn::experimental::prim {

struct AllGatherRegimeAMatmulAsyncDeviceOperation {
    using operation_attributes_t = AllGatherRegimeAMatmulAsyncParams;
    using tensor_args_t = AllGatherRegimeAMatmulAsyncInputs;
    // Vector-valued to support output column-splitting (all_gather_regime_a_matmul_async_split). chunks==1 (the default
    // / public all_gather_regime_a_matmul_async path) yields a single-element vector, mirroring minimal_matmul.
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    // Single program factory in the variant. The framework auto-selects it (no custom
    // select_program_factory required — see ttnn/operation_concepts.hpp: a single-alternative
    // program_factory_t is returned automatically).
    using program_factory_t = std::variant<AllGatherRegimeAMatmulAsyncProgramFactory>;

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
        const std::optional<Tensor>& bias_tensor = std::nullopt,
        std::optional<operations::unary::UnaryWithParam> fused_activation = std::nullopt,
        std::optional<float> fused_ternary_scalar = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
        int32_t chunks = 1);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Returns a vector of output tensors (chunks). chunks==1 => single element (public all_gather_regime_a_matmul_async).
// Numerics are fixed (BF16 in/out, HiFi2, FP32 acc, DRAM-interleaved output) — no dtype/memory_config/
// compute_kernel_config knobs.
std::vector<Tensor> all_gather_regime_a_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<Tensor>& bias_tensor = std::nullopt,
    std::optional<operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
    int32_t chunks = 1);

}  // namespace ttnn::prim
