// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace tt::operations::primary {

tt::tt_metal::Tensor prod_all(
    const tt::tt_metal::Tensor& input,
    const tt::tt_metal::MemoryConfig& mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt::operations::primary
