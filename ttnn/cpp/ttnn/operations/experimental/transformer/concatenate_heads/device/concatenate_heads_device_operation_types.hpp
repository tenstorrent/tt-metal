// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::experimental::transformer {

struct TransformerParams {
    const CoreCoord compute_with_storage_grid_size;
    const tt::tt_metal::MemoryConfig output_mem_config;
};

struct TransformerInputs {
    const Tensor input;
    const std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::experimental::transformer
