// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace operations::pool {

sliding_window::ParallelConfig determine_pool_config_for_auto_shard(
    const Tensor& input_tensor, const sliding_window::SlidingWindowConfig& sliding_window_config, uint32_t channels);

uint32_t calculate_L1_usage(
    const Tensor& input,
    const uint32_t kernel_h,
    const uint32_t kernel_w,
    const uint32_t out_h,
    const uint32_t out_w,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory);
}  // namespace operations::pool
}  // namespace ttnn
