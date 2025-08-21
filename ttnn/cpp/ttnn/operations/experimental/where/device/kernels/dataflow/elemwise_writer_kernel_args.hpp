// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseWriterKernelArgs {
    uint32_t dst_base_addr;
    uint32_t num_tiles;
    uint32_t tile_ofs;
};

struct CompileTimeWriterKernelArgs {
    uint32_t cb_dst;
    bool is_dst_dram;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseWriterKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeWriterKernelArgs>);

}  // namespace ttnn::kernel::eltwise::where_args
