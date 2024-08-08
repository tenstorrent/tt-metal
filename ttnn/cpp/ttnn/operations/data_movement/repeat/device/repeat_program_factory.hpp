// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks repeat_multi_core(
    const Tensor &input_tensor, const uint32_t repeat_dim, const uint32_t num_repeats, const Tensor &output);

}  // namespace ttnn::operations::data_movement::detail 
