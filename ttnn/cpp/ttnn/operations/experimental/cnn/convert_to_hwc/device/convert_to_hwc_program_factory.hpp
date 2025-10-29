// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::cnn::detail {

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output);

}  // namespace ttnn::operations::experimental::cnn::detail
