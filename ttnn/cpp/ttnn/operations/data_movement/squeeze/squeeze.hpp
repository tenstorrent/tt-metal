// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt_stl/small_vector.hpp>

namespace ttnn {

// Note: dim is passed by non-const reference because it's convenient to modify it for processing
Tensor squeeze(const Tensor& input_tensor, const SmallVector<int>& dim);
Tensor squeeze(const Tensor& input_tensor, int dim);
Tensor squeeze(const Tensor& input_tensor);

}  // namespace ttnn
