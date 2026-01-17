// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <enchantum/enchantum.hpp>
#include "ttnn/operations/core/core.hpp"
#include "embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {
struct CoreSplitResult {
    uint32_t required_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
};

CoreSplitResult split_work_to_cores_aligned(CoreCoord grid_size, uint32_t units_to_divide, uint32_t alignment);
}  // namespace ttnn::prim
