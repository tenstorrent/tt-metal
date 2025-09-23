// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../unique_common.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <vector>

namespace ttnn::operations::experimental::unique {

using namespace common;
using namespace ttnn;
using namespace tt::tt_metal;

struct operation_attributes_t {
    const uint32_t single_fetch_subchunk_size;
    const bool sorted;
    const bool return_inverse;
    const bool return_counts;
    const std::optional<int32_t> dim;
    const std::optional<MemoryConfig> memory_config;
};

struct tensor_args_t {
    const Tensor input_tensor;
    const Tensor first_occurrences_tensor;
};

using spec_return_value_t = std::vector<TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::unique
