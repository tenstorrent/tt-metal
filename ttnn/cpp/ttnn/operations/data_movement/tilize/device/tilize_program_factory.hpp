// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks tilize_single_core(const Tensor& a, Tensor& output);
operation::ProgramWithCallbacks tilize_multi_core(const Tensor& a, Tensor& output);


}  // namespace ttnn::operations::data_movement::detail
