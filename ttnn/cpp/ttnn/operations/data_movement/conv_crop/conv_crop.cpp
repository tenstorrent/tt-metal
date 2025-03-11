// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/conv_crop/device/conv_crop_op.hpp"
#include "ttnn/run_operation.hpp"
#include "conv_crop.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ConvCropOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const int crop_height,
    const int crop_width,
    const int pre_crop_height,
    const int pre_crop_width) {
    return operation::run(
               ConvCropDeviceOperation{
                   .output_mem_config = memory_config,
                   .crop_height = crop_height,
                   .crop_width = crop_width,
                   .pre_crop_height = pre_crop_height,
                   .pre_crop_width = pre_crop_width},
               {input_tensor},
               {},
               {})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
