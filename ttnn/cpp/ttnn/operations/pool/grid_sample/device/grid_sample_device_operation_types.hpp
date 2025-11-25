// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::pool::grid_sample {

struct operation_attributes_t {
    std::string mode = "bilinear";
    std::string padding_mode = "zeros";
    bool align_corners = false;
    bool use_precomputed_grid = false;
    bool batch_output_channels = false;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    Tensor input_tensor;
    Tensor grid;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::pool::grid_sample
