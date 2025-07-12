// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

tt::tt_metal::operation::ProgramWithCallbacks move_multi_core_with_overlap(const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement
