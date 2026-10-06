// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "toy_scaled_add.hpp"

#include "device/toy_scaled_add_device_operation.hpp"

namespace ttnn {

namespace {

using operations::toy_scaled_add::ToyScaledAddParams;

bool is_fp32(const Tensor& t) { return t.dtype() == DataType::FLOAT32; }

MemoryConfig resolve_output_memory_config(
    const Tensor& a, const std::optional<MemoryConfig>& memory_config, const std::optional<Tensor>& output_tensor) {
    if (!memory_config.has_value()) {
        return output_tensor.has_value() ? output_tensor->memory_config() : a.memory_config();
    }
    if (memory_config->memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
        !memory_config->shard_spec().has_value()) {
        TT_FATAL(
            a.memory_config().shard_spec().has_value(),
            "toy_scaled_add: a height-sharded memory_config needs a shard spec when a is interleaved");
        return MemoryConfig(
            memory_config->memory_layout(), memory_config->buffer_type(), a.memory_config().shard_spec());
    }
    return *memory_config;
}

}  // namespace

Tensor toy_scaled_add(
    const Tensor& a,
    const Tensor& b,
    float alpha,
    const std::optional<Tensor>& gamma,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(is_device_tensor(a), "toy_scaled_add: a must be a device tensor");
    const DataType output_dtype = dtype.value_or(output_tensor.has_value() ? output_tensor->dtype() : a.dtype());
    const bool any_fp32 =
        is_fp32(a) || is_fp32(b) || (gamma.has_value() && is_fp32(*gamma)) || output_dtype == DataType::FLOAT32;
    const ToyScaledAddParams params{
        .alpha = alpha,
        .output_dtype = output_dtype,
        .output_memory_config = resolve_output_memory_config(a, memory_config, output_tensor),
        .compute_kernel_config = init_device_compute_kernel_config(
            a.device()->arch(),
            compute_kernel_config,
            MathFidelity::HiFi4,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/any_fp32),
    };
    // The framework allocates the output before it validates (see the device operation's call order), so
    // a refused input is refused here, before anything is allocated.
    operations::toy_scaled_add::check_support(params, {.a = a, .b = b, .gamma = gamma, .output = output_tensor});
    return ttnn::prim::toy_scaled_add(a, b, gamma, output_tensor, params);
}

}  // namespace ttnn
