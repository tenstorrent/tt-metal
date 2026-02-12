// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/kernel/kernel_common_utils.hpp"

namespace ttnn::kernel {

struct EmbeddingsReaderKernelArgs {
    std::uint32_t input_buffer_src_addr;
    std::uint32_t weight_buffer_src_addr;
    std::uint32_t output_buffer_src_addr;
    std::uint32_t start_shard_id;
    std::uint32_t next_shard_offset;
    std::uint32_t num_shards;  // total shards across all cores
    std::uint32_t index_idx;
};

struct CompileTimeEmbeddingsReaderKernelArgs {
    uint32_t cb_id_index;
    uint32_t input_page_size;
    uint32_t weight_stick_size;
    uint32_t elems_per_page;  // Input elems per block
    uint32_t input_block_size_bytes;
    std::uint32_t input_buf_alignment;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<EmbeddingsReaderKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeEmbeddingsReaderKernelArgs>);

}  // namespace ttnn::kernel
