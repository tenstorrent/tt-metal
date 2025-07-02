// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel::eltwise::where_args {

struct ElemwiseReaderKernelArgs {
    uint32_t condition_tensor_base_addr;
    uint32_t true_tensor_base_addr;
    uint32_t false_tensor_base_addr;
    uint32_t num_tiles;
    uint32_t tile_ofs;
};

struct CompileTimeReaderKernelArgs {
    uint32_t condition_cb;
    uint32_t true_tensor_cb;
    uint32_t false_tensor_cb;
    bool is_cond_tensor_in_dram;
    bool is_true_tensor_in_dram;
    bool is_false_tensor_in_dram;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseReaderKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeReaderKernelArgs>);

}  // namespace ttnn::kernel::eltwise::where_args
