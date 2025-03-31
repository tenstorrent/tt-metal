// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks tilize_single_core(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks tilize_multi_core_interleaved(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks tilize_multi_core_sharded(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks tilize_multi_core_block(const Tensor& a, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
