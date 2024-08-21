// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks slice_multi_core(const Tensor& a, Tensor& output, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end); 

}  // namespace ttnn::operations::data_movement::detail 

