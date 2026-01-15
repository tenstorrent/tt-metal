// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads {

struct SplitQueryKeyValueAndSplitHeadsParams {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t num_heads{};
};

struct SplitQueryKeyValueAndSplitHeadsInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> output_tensors;
};

}  // namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads
