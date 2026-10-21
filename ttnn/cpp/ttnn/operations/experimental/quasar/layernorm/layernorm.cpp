// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"

#include <optional>

#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/quasar/layernorm/device/layernorm_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

namespace {
// Same defaults as ttnn::layernorm_default_compute_config (HiFi4, approx off, fp32 acc on), inlined so
// this library does not depend on ttnn_op_normalization.
DeviceComputeKernelConfig default_compute_config(tt::ARCH arch) {
    return init_device_compute_kernel_config(
        arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, /*approx_mode=*/false, /*fp32_acc=*/true);
}
}  // namespace

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor) {
    const auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    const auto rank = input_tensor.logical_shape().rank();
    TT_FATAL(rank > 0, "LayerNorm operation not supported for 0D tensors. (rank={})", rank);

    // TODO: 0-volume inputs (original op clones the input). Left for the glue pass.

    const auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                         : ttnn::GetDefaultDevice()->arch();
    const auto kernel_config_val = compute_kernel_config.value_or(default_compute_config(arch));

    // Only the standard Welford path exists here, so the default program config requests it.
    const ttnn::prim::LayerNormProgramConfig program_config_val =
        program_config.value_or(ttnn::prim::LayerNormDefaultProgramConfig{.use_welford = true});

    return ttnn::prim::qsr::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config_val,
        kernel_config_val,
        recip_tensor);
}

}  // namespace ttnn::operations::experimental::quasar
