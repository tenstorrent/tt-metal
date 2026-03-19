// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor sharded_to_interleaved_partial(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& cache_tensor,
    int64_t num_slices,
    int64_t slice_index,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace ttnn
