// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::cnn::detail {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_chw(
    const Tensor& a, Tensor& output, CoreCoord compute_with_storage_grid_size);

}  // namespace ttnn::operations::experimental::cnn::detail
