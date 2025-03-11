// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::data_movement {

struct ConvCropOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& memory_config,
        const int crop_height,
        const int crop_width,
        const int pre_crop_height,
        const int pre_crop_width);
};

}  // namespace operations::data_movement

constexpr auto conv_crop = ttnn::
    register_operation_with_auto_launch_op<"ttnn::conv_crop", ttnn::operations::data_movement::ConvCropOperation>();
}  // namespace ttnn
