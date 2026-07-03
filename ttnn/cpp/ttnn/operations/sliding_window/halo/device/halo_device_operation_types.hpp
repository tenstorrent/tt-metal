// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct HaloParams {
    ttnn::operations::sliding_window::SlidingWindowConfig config{};
    ttnn::operations::sliding_window::ParallelConfig parallel_config{};
    uint32_t pad_val = 0;
    bool remote_read = false;
    bool transpose_mcast = false;
    uint32_t max_out_nsticks_per_core = 0;
    uint32_t in_nsticks_per_core = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    bool is_out_tiled = false;
    bool config_tensors_in_dram = false;
    // Per-caller capability flag: in-place halo (silent output-over-input aliasing) only
    // engages for callers that opt in. Defaults false so conv2d/upsample/fold/etc. never
    // activate it; only the pool caller passes true. See IN_PLACE_HALO_REDO.md sec 10.
    bool allow_in_place = false;
    // Required: caller must pass compute_kernel_config so the halo's untilize
    // compute kernel runs with the correct fp32_dest_acc_en (see issue #43229).
    DeviceComputeKernelConfig compute_kernel_config;
};

}  // namespace ttnn::prim
