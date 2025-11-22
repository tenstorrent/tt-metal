// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window::halo {

struct operation_attributes_t {
    SlidingWindowConfig config_;
    ParallelConfig parallel_config_;
    uint32_t pad_val_;
    bool remote_read_;
    bool transpose_mcast_;
    uint32_t max_out_nsticks_per_core_;
    uint32_t in_nsticks_per_core_;
    tt::tt_metal::MemoryConfig output_memory_config_;
    bool is_out_tiled_;
    bool config_tensors_in_dram_;
};

struct tensor_args_t {
    const Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::sliding_window::halo
