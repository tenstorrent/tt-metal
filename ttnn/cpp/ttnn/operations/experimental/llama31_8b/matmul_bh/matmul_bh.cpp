// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/matmul_bh.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_shapes.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

namespace {

using ttnn::prim::MatmulInputs;
using ttnn::prim::MatmulParams;

bool batched(const ttnn::Shape& shape) {
    for (int i = 0; i < static_cast<int>(shape.rank()) - 2; ++i) {
        if (shape[i] > 1) {
            return true;
        }
    }
    return false;
}

std::optional<MatmulParams> attributes(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    if (!program_config.has_value() || a.logical_shape().rank() < 2 || b.logical_shape().rank() < 2) {
        return std::nullopt;
    }
    auto params = MatmulParams{
        program_config,
        std::nullopt,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        false,
        std::nullopt,
        std::nullopt,
        batched(b.logical_shape())};
    auto attrs = ttnn::prim::create_matmul_attributes(a, b, params, {std::nullopt});
    ttnn::operations::matmul::normalize_program_config(
        attrs.program_config.value(), a.device()->compute_with_storage_grid_size());
    return attrs;
}

MatmulInputs inputs(const ttnn::Tensor& a, const ttnn::Tensor& b) { return {{a, b}, {std::nullopt}, {std::nullopt}}; }

DeviceComputeKernelConfig minimal_kernel_config(
    const ttnn::Tensor& input_tensor, std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true, true);
}

ttnn::experimental::prim::MinimalMatmulParams minimal_attributes(
    const ttnn::experimental::prim::MinimalMatmulConfig& config,
    std::optional<const DataType> dtype,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    return {
        .config = config,
        .fused_activation = std::nullopt,
        .output_mem_config = std::nullopt,
        .output_dtype = dtype,
        .fused_ternary_scalar = std::nullopt,
        .compute_kernel_config = compute_kernel_config,
        .chunks = 1,
        .dim = -1,
        .fuse_swiglu = false};
}

ttnn::experimental::prim::MinimalMatmulInputs minimal_inputs(const ttnn::Tensor& input, const ttnn::Tensor& weight) {
    return {
        .input_tensor = input,
        .weight_tensor = weight,
        .bias_tensor = std::nullopt,
        .optional_input_tensor = std::nullopt,
        .fused_ternary_input_a = std::nullopt,
        .fused_ternary_input_b = std::nullopt};
}

}  // namespace

bool matmul_bh_covers(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto attrs =
        attributes(input_tensor_a, input_tensor_b, memory_config, dtype, program_config, compute_kernel_config);
    return attrs.has_value() &&
           ttnn::prim::llama31_8b_matmul_bh::covers(attrs.value(), inputs(input_tensor_a, input_tensor_b)) &&
           ttnn::prim::llama31_8b_matmul_bh::fits(attrs.value(), inputs(input_tensor_a, input_tensor_b));
}

ttnn::Tensor matmul_bh(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto attrs =
        attributes(input_tensor_a, input_tensor_b, memory_config, dtype, program_config, compute_kernel_config);
    TT_FATAL(attrs.has_value(), "llama31_8b.matmul_bh needs a program_config and tensors of rank >= 2");
    return ttnn::device_operation::launch<ttnn::prim::llama31_8b_matmul_bh::MatmulBhDeviceOperation>(
               attrs.value(), inputs(input_tensor_a, input_tensor_b))
        .at(0);
}

bool minimal_matmul_bh_covers(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    const auto kernel_config = minimal_kernel_config(input_tensor, compute_kernel_config);
    auto routed = ttnn::prim::llama31_8b_matmul_bh::minimal_route(
        input_tensor, weight_tensor, config, dtype.value_or(input_tensor.dtype()), kernel_config);
    return routed.has_value() &&
           ttnn::experimental::prim::llama31_8b_matmul_bh::minimal_fits(
               minimal_attributes(routed.value(), dtype, kernel_config), minimal_inputs(input_tensor, weight_tensor));
}

ttnn::Tensor minimal_matmul_bh(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto routed = ttnn::prim::llama31_8b_matmul_bh::minimal_route(
        input_tensor,
        weight_tensor,
        config,
        dtype.value_or(input_tensor.dtype()),
        minimal_kernel_config(input_tensor, compute_kernel_config));
    TT_FATAL(
        routed.has_value(),
        "llama31_8b.minimal_matmul_bh: this call is not one of the Llama-3.1-8B Blackhole p100a shapes it covers; "
        "use ttnn.experimental.minimal_matmul");
    using OperationType = ttnn::experimental::prim::llama31_8b_matmul_bh::MinimalBhDeviceOperation;
    ttnn::verify_numerical_configuration(input_tensor.device()->arch(), compute_kernel_config);
    return ttnn::device_operation::launch<OperationType>(
               OperationType::operation_attributes_t{
                   .config = routed.value(),
                   .fused_activation = std::nullopt,
                   .output_mem_config = std::nullopt,
                   .output_dtype = dtype,
                   .fused_ternary_scalar = std::nullopt,
                   .compute_kernel_config = minimal_kernel_config(input_tensor, compute_kernel_config),
                   .chunks = 1,
                   .dim = -1,
                   .fuse_swiglu = false},
               OperationType::tensor_args_t{
                   .input_tensor = input_tensor,
                   .weight_tensor = weight_tensor,
                   .bias_tensor = std::nullopt,
                   .optional_input_tensor = std::nullopt,
                   .fused_ternary_input_a = std::nullopt,
                   .fused_ternary_input_b = std::nullopt})
        .at(0);
}

bool qkv_heads_bh_covers(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_heads,
    uint32_t num_kv_heads) {
    const auto& shape = input_tensor.logical_shape();
    return shape.rank() == 4 && shape[-3] == 1 && input_tensor.padded_shape() == shape &&
           weight_tensor.logical_shape()[-1] == (num_heads + 2 * num_kv_heads) * 128 &&
           minimal_matmul_bh_covers(input_tensor, weight_tensor, config, dtype, compute_kernel_config);
}

std::vector<ttnn::Tensor> qkv_heads_bh(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_heads,
    uint32_t num_kv_heads) {
    TT_FATAL(
        qkv_heads_bh_covers(input_tensor, weight_tensor, config, dtype, compute_kernel_config, num_heads, num_kv_heads),
        "llama31_8b.qkv_heads_bh: this call is not one of the Llama-3.1-8B Blackhole p100a shapes it covers");
    auto routed = ttnn::prim::llama31_8b_matmul_bh::minimal_route(
        input_tensor,
        weight_tensor,
        config,
        dtype.value_or(input_tensor.dtype()),
        minimal_kernel_config(input_tensor, compute_kernel_config));
    using OperationType = ttnn::experimental::prim::llama31_8b_matmul_bh::QkvHeadsDeviceOperation;
    ttnn::verify_numerical_configuration(input_tensor.device()->arch(), compute_kernel_config);
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .matmul =
                {.config = routed.value(),
                 .fused_activation = std::nullopt,
                 .output_mem_config = std::nullopt,
                 .output_dtype = dtype,
                 .fused_ternary_scalar = std::nullopt,
                 .compute_kernel_config = minimal_kernel_config(input_tensor, compute_kernel_config),
                 .chunks = 3,
                 .dim = -1,
                 .fuse_swiglu = false},
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .bias_tensor = std::nullopt,
            .optional_input_tensor = std::nullopt,
            .fused_ternary_input_a = std::nullopt,
            .fused_ternary_input_b = std::nullopt});
}

}  // namespace ttnn::experimental::llama31_8b
