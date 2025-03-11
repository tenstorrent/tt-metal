// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::data_movement {

struct ConvKnitOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        int kernel_height,
        int num_output_channels,
        int input_width,
        int num_input_channels);
};

}  // namespace operations::data_movement

constexpr auto conv_knit = ttnn::
    register_operation_with_auto_launch_op<"ttnn::conv_knit", ttnn::operations::data_movement::ConvKnitOperation>();
}  // namespace ttnn
