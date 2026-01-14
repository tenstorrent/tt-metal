// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::reduction::argmax {

struct operation_attributes_t {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    bool use_multicore{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

}  // namespace ttnn::operations::reduction::argmax
