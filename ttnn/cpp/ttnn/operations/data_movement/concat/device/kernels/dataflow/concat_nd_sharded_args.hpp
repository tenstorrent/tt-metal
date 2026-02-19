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
//   [0 .. num_input_tensors]: buffer addresses (output at 0, then input0, input1, ...)
//   [num_input_tensors + 1]: shard_id (this core's index in the grid)

}  // namespace ttnn::kernel
