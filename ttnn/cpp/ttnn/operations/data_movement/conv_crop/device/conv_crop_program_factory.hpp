// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

struct Stride {
    CoreCoord core;
    uint32_t data;
};

struct PageStride {
    CoreCoord start_core;
    uint32_t start_data;
    uint32_t stride_size;  // number of pages per stride
    Stride stride;
    uint32_t num_strides;
    bool skip;
};

tt::tt_metal::operation::ProgramWithCallbacks conv_crop_multi_core(
    const Tensor& input, Tensor& output, int crop_height, int crop_width, int pre_crop_height, int pre_crop_width);

}  // namespace ttnn::operations::data_movement::detail
