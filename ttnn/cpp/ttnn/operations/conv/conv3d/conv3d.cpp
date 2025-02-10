// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::conv {
namespace conv3d {

ttnn::Tensor ExecuteConv3d::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    // const ttnn::Tensor& weight_tensor,
    uint32_t output_channels,
    std::array<uint32_t, 3> kernel_size,
    std::array<uint32_t, 3> stride,
    std::array<uint32_t, 3> padding,
    std::string padding_mode,
    uint32_t groups,
    const std::optional<MemoryConfig>& memory_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();

    return operation::run(
               Conv3dOp{
                   .output_channels = output_channels,
                   .kernel_size = kernel_size,
                   .stride = stride,
                   .padding = padding,
                   .padding_mode = padding_mode,
                   .groups = groups,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG)},
               {input_tensor},
               {},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteConv3d::invoke(
    const ttnn::Tensor& input_tensor,
    // const ttnn::Tensor& weight_tensor,
    uint32_t output_channels,
    std::array<uint32_t, 3> kernel_size,
    std::array<uint32_t, 3> stride,
    std::array<uint32_t, 3> padding,
    std::string padding_mode,
    uint32_t groups,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(
        DefaultQueueId,
        input_tensor,
        // weight_tensor,
        output_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        groups,
        memory_config);
}

}  // namespace conv3d
}  // namespace ttnn::operations::conv
