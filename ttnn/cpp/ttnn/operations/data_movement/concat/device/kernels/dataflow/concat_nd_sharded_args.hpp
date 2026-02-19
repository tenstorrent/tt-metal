// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel {

// Maximum number of input tensors supported by the ND sharded concat kernel.
// The kernel is compiled with this many TensorAccessorArgs (output + MAX_NUM_INPUTS).
constexpr std::uint32_t CONCAT_ND_SHARDED_MAX_NUM_INPUTS = 16u;

// Runtime args layout (per core, via SetRuntimeArgs):
//   [0]: scratch L1 buffer address (host-allocated, one page, max page size)
//   [1]: output buffer address
//   [2 .. 17]: input buffer addresses (input0..input15; absent filled from first input)
//   [18]: shard_id (this core's index in the grid)

}  // namespace ttnn::kernel
