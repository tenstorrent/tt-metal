// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel::eltwise::add_nd_sharded_args {

struct ElemwiseReaderKernelArgs {
    uint32_t a_tensor_base_addr;
    uint32_t b_tensor_base_addr;
    uint32_t num_shards;
    uint32_t shard_id;
    uint32_t next_shard_offset;
};

struct CompileTimeReaderKernelArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t num_tiles_per_cycle;
};

struct ElemwiseWriterKernelArgs {
    uint32_t dst_base_addr;
    uint32_t num_shards;
    uint32_t shard_id;
    uint32_t next_shard_offset;
    uint32_t num_cycles_per_shard;  // number of cb batches to write per shard (tiles per shard / num_tiles_per_cycle)
};

struct CompileTimeWriterKernelArgs {
    uint32_t cb_dst;
    uint32_t num_tiles_per_cycle;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseWriterKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeWriterKernelArgs>);

static_assert(ttnn::kernel_utils::SerializableKernelArgs<ElemwiseReaderKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeReaderKernelArgs>);

}  // namespace ttnn::kernel::eltwise::add_nd_sharded_args
