// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include <optional>

#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/quasar/batch_norm/device/batch_norm_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

namespace {
// HiFi4, approx off, fp32 accumulation for the per-channel scale / shift.
DeviceComputeKernelConfig batch_norm_default_compute_config(tt::ARCH arch) {
    return init_device_compute_kernel_config(
        arch,
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
}
}  // namespace

Tensor batch_norm(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    const float /*momentum*/,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(!training, "Quasar batch_norm only implements inference (training=False).");
    TT_FATAL(!output.has_value(), "Quasar batch_norm does not support a preallocated output tensor.");
    TT_FATAL(
        running_mean.has_value() && running_var.has_value(),
        "Quasar batch_norm (inference) requires running_mean and running_var.");
    TT_FATAL(
        input.logical_shape().rank() == 4,
        "batch_norm not supported for tensors with rank != 4. (rank={})",
        input.logical_shape().rank());

    const auto output_memory_config = memory_config.value_or(input.memory_config());
    const auto arch =
        input.storage_type() == StorageType::DEVICE ? input.device()->arch() : ttnn::GetDefaultDevice()->arch();
    const auto kernel_config_val = compute_kernel_config.value_or(batch_norm_default_compute_config(arch));

    const auto threads = ttnn::prim::qsr::thread_plan_from_env("TTNN_QSR_BATCH_NORM_THREADS", "batch_norm");
    return ttnn::prim::qsr::batch_norm(
        input, *running_mean, *running_var, weight, bias, eps, output_memory_config, kernel_config_val, threads);
}

}  // namespace ttnn::operations::experimental::quasar
