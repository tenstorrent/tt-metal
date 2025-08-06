// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks reshard_multi_core(const std::vector<Tensor>& inputs, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
