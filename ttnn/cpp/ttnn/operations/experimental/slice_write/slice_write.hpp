// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttnn::experimental {

ttnn::Tensor slice_write(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    const ttnn::SmallVector<uint32_t>& begins,
    const ttnn::SmallVector<uint32_t>& ends,
    const ttnn::SmallVector<uint32_t>& step);

}  // namespace ttnn::experimental
