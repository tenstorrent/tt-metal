// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseComputeKernelArgs {
    uint32_t per_core_block_cnt;
    uint32_t per_core_block_size;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseComputeKernelArgs>);

}  // namespace ttnn::kernel::eltwise::where_args
