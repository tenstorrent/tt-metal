// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include <vector>
#include <optional>

namespace ttnn::prim {
namespace untilize_helper {
uint32_t get_largest_divisor(uint32_t dividend, uint32_t starting_divisor, uint32_t divisor_factor = 1);
}  // namespace untilize_helper

struct UntilizeTensorArgs {
    Tensor input;
};

struct UntilizeOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_multicore{};
    bool fp32_dest_acc_en{};
    std::optional<CoreRangeSet> sub_core_grids;
    bool enough_space_width{};
    bool enough_space_height{};
    uint32_t pf_type{};
};

using UntilizeTensorReturnValue = Tensor;
using UntilizeSpecReturnValue = tt::tt_metal::TensorSpec;
using UntilizeShapeReturnValue = ttnn::Shape;

}  // namespace ttnn::prim
