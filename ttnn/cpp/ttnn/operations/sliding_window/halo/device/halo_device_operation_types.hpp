// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window::halo {

struct HaloParams {
    SlidingWindowConfig config{};
    ParallelConfig parallel_config{};
    uint32_t pad_val = 0;
    bool remote_read = false;
    bool transpose_mcast = false;
    uint32_t max_out_nsticks_per_core = 0;
    uint32_t in_nsticks_per_core = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    bool is_out_tiled = false;
    bool config_tensors_in_dram = false;
};

struct HaloInputs {
    Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::sliding_window::halo
