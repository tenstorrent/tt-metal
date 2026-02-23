// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace eltwise_nd_dram_optimized {

/* READER KERNEL ARGS */
struct EltwiseReaderArgs {
    uint32_t a_tensor_base_addr;
    uint32_t b_tensor_base_addr;
    uint32_t num_shards;
    uint32_t shard_id;
    uint32_t next_shard_offset;
};

struct EltwiseReaderCTArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t num_tiles_per_cycle;
};

/* COMPUTE KERNEL ARGS */
struct EltwiseComputeArgs {
    uint32_t num_tiles;
};

struct EltwiseComputeCTArgs {
    uint32_t a_tensor_cb;
    uint32_t b_tensor_cb;
    uint32_t output_cb;
    uint32_t num_tiles_per_cycle;
};

/* WRITER KERNEL ARGS */
struct EltwiseWriterArgs {
    uint32_t dst_base_addr;
    uint32_t num_shards;
    uint32_t shard_id;
    uint32_t next_shard_offset;
    uint32_t num_cycles_per_shard;  // number of cb batches to write per shard (tiles per shard / num_tiles_per_cycle)
};

struct EltwiseWriterCTArgs {
    uint32_t cb_dst;
    uint32_t num_tiles_per_cycle;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseComputeArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseComputeCTArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseWriterArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseWriterCTArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseReaderArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<EltwiseReaderCTArgs>);

}  // namespace eltwise_nd_dram_optimized
