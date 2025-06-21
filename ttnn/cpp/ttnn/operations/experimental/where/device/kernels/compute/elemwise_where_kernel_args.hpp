// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "cpp/kernel/kernel_host_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseComputeKernelArgs {
    uint32_t per_core_block_cnt;
    uint32_t per_core_block_size;
};

VALIDATE_KERNEL_ARGS_STRUCT(ElemwiseComputeKernelArgs)

}  // namespace ttnn::kernel::eltwise::where_args
