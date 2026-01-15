// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace tt::operations::primary {

using namespace tt_metal;

Tensor prod_nc(
    const Tensor& input,
    const Tensor& output,
    ttnn::SmallVector<int64_t>& dims,
    const MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt::operations::primary
