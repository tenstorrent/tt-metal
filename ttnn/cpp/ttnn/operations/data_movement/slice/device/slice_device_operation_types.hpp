// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::slice {

struct operation_attributes_t {
    const ttnn::Shape slice_start;
    const ttnn::Shape slice_end;
    const ttnn::Shape step;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const bool use_tensor_args;
    const std::optional<uint32_t> slice_dim;
    const std::optional<uint32_t> num_devices;
    const std::optional<CoreRangeSet> sub_core_grids;
};

struct tensor_args_t {
    const Tensor& input;
    const std::optional<Tensor> start_tensor;
    const std::optional<Tensor> end_tensor;
    const std::optional<Tensor>& preallocated_output;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement::slice
