// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::transformer {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_concat_heads_decode_subcoregrids(
    const Tensor& input_tensor, Tensor& output, CoreCoord compute_with_storage_grid_size);

}  // namespace ttnn::operations::experimental::transformer
