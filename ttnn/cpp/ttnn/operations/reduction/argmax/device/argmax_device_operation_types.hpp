// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::argmax {

struct operation_attributes_t {
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::UINT32;
    std::optional<int> dim;
    bool keepdim = false;
    std::optional<CoreRangeSet> sub_core_grids;
    bool use_multicore = false;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::reduction::argmax
