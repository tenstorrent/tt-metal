// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor roll(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int>& shifts, const ttnn::SmallVector<int>& input_dims);

ttnn::Tensor roll(const ttnn::Tensor& input_tensor, int shifts);

ttnn::Tensor roll(const ttnn::Tensor& input_tensor, int shifts, int dim);

}  // namespace ttnn
