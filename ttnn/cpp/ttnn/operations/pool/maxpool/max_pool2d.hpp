// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "device/max_pool2d_device_op.hpp"

namespace ttnn {
namespace operations::pool {

struct MaxPool2DOp {
    static Tensor invoke(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation);

};

}  // namespace operations::pool

constexpr auto max_pool2d = ttnn::register_operation_with_auto_launch_op<"ttnn::max_pool2d", operations::pool::MaxPool2DOp>();

}  // namespace ttnn
