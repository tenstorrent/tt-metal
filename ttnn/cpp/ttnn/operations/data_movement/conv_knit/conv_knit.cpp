// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/conv_knit_op.hpp"
#include "ttnn/tensor/types.hpp"
#include "conv_knit.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ConvKnitOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    int kernel_height,
    int num_output_channels,
    int input_width,
    int num_input_channels) {
    log_info(
        tt::LogOp,
        "in ConvKnitOperationInvoke queue_id: {}, kernel_height: {}, num_output_channels: {} input_width: {}",
        *queue_id,
        kernel_height,
        num_output_channels,
        input_width,
        num_input_channels);
    return operation::run(
               ConvKnitDeviceOperation{
                   .kernel_height = kernel_height,
                   .num_output_channels = num_output_channels,
                   input_width = input_width,
                   num_input_channels = num_input_channels},
               {input_tensor})
        .at(0);
}
}  // namespace ttnn::operations::data_movement
