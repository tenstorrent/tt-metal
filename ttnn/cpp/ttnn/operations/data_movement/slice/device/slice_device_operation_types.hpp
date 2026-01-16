// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::slice {

struct SliceParams {
    ttnn::Shape slice_start;
    ttnn::Shape slice_end;
    ttnn::Shape step;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_tensor_args = false;
    std::optional<uint32_t> slice_dim = std::nullopt;
    std::optional<uint32_t> num_devices = std::nullopt;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct SliceInputs {
    Tensor input;
    std::optional<Tensor> start_tensor;
    std::optional<Tensor> end_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::data_movement::slice
