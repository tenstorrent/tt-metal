// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"
#include <optional>

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/layernorm_device_operation.hpp"
#include "device/layernorm_common.hpp"
#include "ttnn/device.hpp"
namespace ttnn::operations::normalization {}  // namespace ttnn::operations::normalization

namespace ttnn {

DeviceComputeKernelConfig layernorm_default_compute_config(tt::ARCH arch) {
    bool approx_mode = false;
    bool fp32_acc = true;
    return init_device_compute_kernel_config(arch, std::nullopt, MathFidelity::HiFi4, approx_mode, fp32_acc);
}

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto rank = input_tensor.logical_shape().rank();

    // For 0D tensors
    TT_FATAL(rank > 0, "LayerNorm operation not supported for 0D tensors. (rank={})", rank);

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, output_memory_config, compute_kernel_config);
    }

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = compute_kernel_config.value_or(layernorm_default_compute_config(arch));

    return ttnn::prim::layer_norm(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        output_memory_config,
        program_config.value_or(ttnn::prim::create_layernorm_program_config(input_tensor.shard_spec())),
        kernel_config_val,
        std::nullopt,                                      // dtype
        prim::LayerNormType::LAYERNORM,                    // norm_type
        prim::DistributedLayerNormStage::NOT_DISTRIBUTED,  // distributed_norm_stage
        std::nullopt,                                      // stats
        recip_tensor);
}

}  // namespace ttnn
