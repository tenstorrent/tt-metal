// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/kernels/riscv_copy.h"

namespace codegen_kernels {

struct SegmentMapData {
    uint32_t input_page_index;
    uint32_t input_page_offset;
    uint32_t output_page_offset;
    uint32_t num_elements;
};

// Segments congruent modulo 16 bytes go through the NoC and are ordered by the async_write_barrier before the tile
// write-out. The rest are RISC copies; a NoC barrier does not order those stores, so the write-out drains the last
// one first. Returns the last RISC store so far.
template <uint32_t tile_bytes>
FORCE_INLINE uint32_t copy_segment(Noc noc, uint32_t dst, uint32_t src, uint32_t bytes, uint32_t last_store) {
    if ((dst & OFFSET_16) == (src & OFFSET_16)) {
        tt::data_movement::common::tt_memmove<false, true, false, tile_bytes>(noc, dst, src, bytes);
        return last_store;
    }
    return riscv_copy_elements(dst, src, bytes);
}

// Copies map entries [first_entry, max_entries) into the scratch tile at output_l1, popping one input tile per change
// of input page to match the reader's pushes. The last input tile stays at the front for the next call or the caller to
// pop; have_input says one is held. Every copy has landed in L1 on return.
template <uint32_t tile_bytes, uint32_t max_entries, uint32_t element_bytes>
FORCE_INLINE void assemble_segments(
    Noc noc,
    CircularBuffer& input_buffer,
    const CoreLocalMem<volatile SegmentMapData>& segments,
    uint32_t first_entry,
    uint32_t output_l1,
    bool& have_input) {
    uint32_t input_l1 = 0;
    uint32_t previous_page = std::numeric_limits<uint32_t>::max();
    uint32_t last_store = 0;
    for (uint32_t index = first_entry; index < max_entries; ++index) {
        if (segments[index].num_elements == 0) {
            continue;
        }
        if (segments[index].input_page_index != previous_page) {
            if (have_input) {
                noc.async_writes_flushed();
                input_buffer.pop_front(1);
            }
            input_buffer.wait_front(1);
            invalidate_l1_cache();
            input_l1 = input_buffer.get_read_ptr();
            previous_page = segments[index].input_page_index;
            have_input = true;
        }
        last_store = copy_segment<tile_bytes>(
            noc,
            output_l1 + segments[index].output_page_offset * element_bytes,
            input_l1 + segments[index].input_page_offset * element_bytes,
            segments[index].num_elements * element_bytes,
            last_store);
    }
    noc.async_write_barrier();
    if (last_store != 0) {
        drain_stores(last_store);
    }
}

}  // namespace codegen_kernels
