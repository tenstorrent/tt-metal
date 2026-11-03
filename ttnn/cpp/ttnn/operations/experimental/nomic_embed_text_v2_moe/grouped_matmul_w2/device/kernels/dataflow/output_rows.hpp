// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One output block of grouped_matmul_w2: compute's row-major 32 x (n_tiles * 32) bfloat16 block, written as each
// row's segment at column_offset of the output's row page, then flushed and popped.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/kernels/dataflow/expert_schedule.hpp"

namespace grouped_matmul_w2 {

template <uint32_t n_tiles, uint32_t segment_bytes, typename Accessor>
FORCE_INLINE void write_output_block(
    const Noc& noc, const Accessor& output, DataflowBuffer& output_cb, uint32_t block, uint32_t column_offset) {
    const uint32_t source = output_cb.get_read_ptr();
    uint32_t row = block * grouped_matmul::kBlockRows;
    for (uint32_t index = 0; index < grouped_matmul::kBlockRows; ++index, ++row) {
        noc.async_write(
            CoreLocalMem<uint32_t>(source + index * segment_bytes),
            output,
            segment_bytes,
            {},
            {.page_id = row, .offset_bytes = column_offset});
    }
    noc.async_writes_flushed();
    output_cb.pop_front(n_tiles);
}

}  // namespace grouped_matmul_w2
