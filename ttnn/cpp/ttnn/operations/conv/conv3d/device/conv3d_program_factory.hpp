// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::conv::conv3d::detail {

operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    // const Tensor& weight_tensor,
    uint32_t output_channels,
    std::array<uint32_t, 3> kernel_size,
    std::array<uint32_t, 3> stride,
    std::array<uint32_t, 3> padding,
    std::string padding_mode,
    uint32_t groups,
    // const Tensor& bias_tensor,
    const Tensor& output_tensor);

}  // namespace ttnn::operations::conv::conv3d::detail
