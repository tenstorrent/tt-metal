// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::pool::grid_sample {

struct operation_attributes_t {
    const std::string mode_;
    const std::string padding_mode_;
    const bool align_corners_;
    const bool use_precomputed_grid_;
    const bool batch_output_channels_;
    const tt::tt_metal::MemoryConfig output_mem_config_;
};

struct tensor_args_t {
    const Tensor input_tensor;
    const Tensor grid;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::pool::grid_sample
