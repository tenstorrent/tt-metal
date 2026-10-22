// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include <optional>

#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/quasar/softmax/device/softmax_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

namespace {
// HiFi4, approx off, fp32 accumulation: the row statistics (max, sum) and the exp intermediates are
// kept in fp32 when the config asks for it.
DeviceComputeKernelConfig softmax_default_compute_config(tt::ARCH arch) {
    return init_device_compute_kernel_config(
        arch,
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
}
}  // namespace

Tensor softmax(
    const Tensor& input_tensor,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    const auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    const int rank = static_cast<int>(input_tensor.logical_shape().rank());
    TT_FATAL(rank > 0, "softmax: 0D tensors are not supported");
    const int dim_calculated = dim < 0 ? rank + dim : dim;
    TT_FATAL(
        dim_calculated >= 0 && dim_calculated < rank,
        "softmax: dimension out of range (expected [{}, {}], got {})",
        -rank,
        rank - 1,
        dim);
    TT_FATAL(
        dim_calculated == rank - 1,
        "Quasar softmax only implements the last dimension (dim={} of rank {}); transpose first.",
        dim,
        rank);

    const auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                         : ttnn::GetDefaultDevice()->arch();
    const auto kernel_config_val = compute_kernel_config.value_or(softmax_default_compute_config(arch));

    const auto threads = ttnn::prim::qsr::thread_plan_from_env("TTNN_QSR_SOFTMAX_THREADS", "softmax");
    return ttnn::prim::qsr::softmax(input_tensor, output_memory_config, kernel_config_val, numeric_stable, threads);
}

}  // namespace ttnn::operations::experimental::quasar
