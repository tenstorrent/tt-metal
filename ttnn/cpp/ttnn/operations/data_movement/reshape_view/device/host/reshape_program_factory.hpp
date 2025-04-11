// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::reshape {

tt::tt_metal::operation::ProgramWithCallbacks rm_reshape_preparer(const Tensor& input, const Tensor& output);

tt::tt_metal::operation::ProgramWithCallbacks reshape_tiled_program_factory(const Tensor& input, const Tensor& output);

}  // namespace ttnn::operations::data_movement::reshape
