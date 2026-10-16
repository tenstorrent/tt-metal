// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Segment-mapped TILE reshape writer. Assembles each output tile in a scratch tile from the map row the reader
// staged, then writes the whole tile. Pops one input tile per change of input page, matching the reader's pushes.
#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/kernels/riscv_copy.h"

using namespace tt::data_movement::common;
using namespace codegen_kernels;

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
        tt_memmove<false, true, false, tile_bytes>(noc, dst, src, bytes);
        return last_store;
    }
    return riscv_copy_elements(dst, src, bytes);
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_output_page = get_arg_val<uint32_t>(0);
    const uint32_t end_output_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t max_entries = get_compile_time_arg_val(1);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t map_cb = get_compile_time_arg_val(3);
    constexpr uint32_t input_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer output_scratch(output_cb);
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);
    output_scratch.reserve_back(1);
    const uint32_t output_l1 = output_scratch.get_write_ptr();
    const CoreLocalMem<uint8_t> output_mem(output_l1);

    bool have_input = false;
    uint32_t input_l1 = 0;
    for (uint32_t output_page = start_output_page; output_page < end_output_page; ++output_page) {
        // Segments cover only logical datums; the tile padding must not inherit the previous tile's bytes.
        noc.async_write_zeros(output_scratch, tile_bytes);
        noc.write_zeros_l1_barrier();

        map_buffer.wait_front(1);
        invalidate_l1_cache();
        CoreLocalMem<volatile SegmentMapData> segments(map_buffer.get_read_ptr());
        uint32_t previous_page = std::numeric_limits<uint32_t>::max();
        uint32_t last_store = 0;

        for (uint32_t index = 0; index < max_entries; ++index) {
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
        noc.async_write<NocOptions::DEFAULT, tile_bytes>(
            output_mem, output_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = output_page, .offset_bytes = 0});
        noc.async_writes_flushed();
        map_buffer.pop_front(1);
    }
    noc.async_write_barrier();
    if (have_input) {
        input_buffer.pop_front(1);
    }
    output_scratch.push_back(1);
}
