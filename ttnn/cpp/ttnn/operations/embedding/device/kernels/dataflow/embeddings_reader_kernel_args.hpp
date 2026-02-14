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
    std::uint32_t input_page_id;
    std::uint32_t num_of_pages;
};

struct CompileTimeEmbeddingsReaderKernelArgs {
    uint32_t input_cb_index;
    uint32_t input_page_size;
    uint32_t weight_page_size;
    uint32_t elems_per_page;
    uint32_t input_block_size_bytes;
    uint32_t input_buf_alignment;
    uint32_t output_cb_index;
};

static_assert(ttnn::kernel_utils::SerializableKernelArgs<EmbeddingsReaderKernelArgs>);
static_assert(ttnn::kernel_utils::SerializableKernelArgs<CompileTimeEmbeddingsReaderKernelArgs>);

}  // namespace ttnn::kernel
