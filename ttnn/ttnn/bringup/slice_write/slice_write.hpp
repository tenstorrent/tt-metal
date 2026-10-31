// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttnn::experimental::bringup {

ttnn::Tensor slice_write(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    const ttsl::SmallVector<uint32_t>& begins,
    const ttsl::SmallVector<uint32_t>& ends,
    const ttsl::SmallVector<uint32_t>& step,
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt);

}  // namespace ttnn::experimental::bringup
