// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

ttnn::Tensor ExecuteConv3d::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const Conv3dConfig& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               Conv3dOp{
                   .config = config,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .compute_kernel_config = kernel_config_val},
               {input_tensor, weight_tensor},
               {bias_tensor},
               {},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::experimental::conv3d
