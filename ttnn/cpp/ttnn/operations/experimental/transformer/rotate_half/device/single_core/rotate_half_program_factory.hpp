// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::transformer::detail {

operation::ProgramWithCallbacks rotate_half_single_core(const Tensor &input_tensor, Tensor &output_tensor);

} // namespace ttnn::operations::experimental::transformer::detail
