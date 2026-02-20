// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel::eltwise::add_args {

struct ElemwiseComputeKernelArgs {
    uint32_t num_tiles;
    uint32_t tile_ofs;
};

struct CompileTimeComputeKernelArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t output_cb;
    uint32_t num_tiles_per_cycle;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseComputeKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeComputeKernelArgs>);

}  // namespace ttnn::kernel::eltwise::add_args
