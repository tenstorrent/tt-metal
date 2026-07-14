// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_device_operation.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void RegimeAMatmulDeviceOperation::validate_on_program_cache_miss(
    [[maybe_unused]] const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;

    // Device / storage.
    TT_FATAL(
        act.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "regime_a_matmul operands must be on device");
    TT_FATAL(act.device() == weight.device(), "regime_a_matmul inputs must reside on the same device");
    TT_FATAL(
        act.buffer() != nullptr && weight.buffer() != nullptr,
        "regime_a_matmul inputs must be allocated in device buffers");

    // Layout.
    TT_FATAL(
        act.layout() == Layout::TILE && weight.layout() == Layout::TILE,
        "regime_a_matmul requires TILE layout for input and weight");

    // DType: bf16 in/out only (v1).
    TT_FATAL(
        act.dtype() == DataType::BFLOAT16 && weight.dtype() == DataType::BFLOAT16,
        "regime_a_matmul v1 supports only BFLOAT16 inputs");

    // Shapes: no batching — all leading dims (< -2) must be 1 for both operands.
    const auto& a_logical = act.logical_shape();
    const auto& w_logical = weight.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "regime_a_matmul expects rank >= 2 tensors");
    for (int i = 0; i < static_cast<int>(a_logical.rank()) - 2; ++i) {
        TT_FATAL(a_logical[i] == 1, "regime_a_matmul input must have 1 in all dims < -2 (no batching)");
    }
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(w_logical[i] == 1, "regime_a_matmul weight must have 1 in all dims < -2 (no batching)");
    }

    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    TT_FATAL(K == K_w, "regime_a_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(a_logical[-2] > 0 && K > 0 && w_logical[-1] > 0, "regime_a_matmul dimensions must be positive");

    // in1 must be DRAM width-sharded (built via create_regime_a_weight_memory_config).
    const auto& w_mem = weight.memory_config();
    TT_FATAL(
        w_mem.buffer_type() == BufferType::DRAM && w_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "regime_a_matmul weight must be DRAM WIDTH_SHARDED (use create_regime_a_weight_memory_config)");

    // config is optional: nullopt -> the program factory auto-selects via auto_select_config (ported
    // FLUX/LTX picker). An explicit RegimeAMatmulConfig overrides for reproducibility.
}

RegimeAMatmulDeviceOperation::spec_return_value_t RegimeAMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;
    const uint32_t N = weight.logical_shape()[-1];

    // Output is bf16 by default (v1) and DRAM interleaved unless overridden.
    const auto dtype = operation_attributes.output_dtype.value_or(DataType::BFLOAT16);
    const auto memory_config = operation_attributes.output_mem_config.value_or(MemoryConfig{});

    ttnn::Shape output_shape(act.logical_shape());
    output_shape[-1] = N;
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

RegimeAMatmulDeviceOperation::tensor_return_value_t RegimeAMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor.device());
}

std::tuple<RegimeAMatmulDeviceOperation::operation_attributes_t, RegimeAMatmulDeviceOperation::tensor_args_t>
RegimeAMatmulDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    const auto arch = input_tensor.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    return {
        operation_attributes_t{
            .config = config,
            .output_mem_config = memory_config,
            .output_dtype = dtype,
            .compute_kernel_config = kernel_config_val},
        tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor}};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor regime_a_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = experimental::prim::RegimeAMatmulDeviceOperation;
    const auto arch = input_tensor.device()->arch();
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);

    auto [attributes, tensor_args] =
        OperationType::invoke(input_tensor, weight_tensor, config, memory_config, dtype, compute_kernel_config);
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
