// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv_knit_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t kernel_height,
    uint32_t num_output_channels,
    uint32_t input_width,
    uint32_t num_input_channels);
}
