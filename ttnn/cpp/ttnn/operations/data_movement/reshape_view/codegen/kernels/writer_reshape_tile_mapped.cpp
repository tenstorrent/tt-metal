// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Segment-mapped TILE reshape writer. Assembles each map row's segments at their natural offsets in a zeroed scratch
// tile, then writes the tile out. Pops one input tile per change of input page, matching the reader's pushes.
//
// A sliced map row covers one byte range of an output tile and leads with a header entry
// {output_page, range_start_bytes, range_bytes, 0} that the reader skips (num_elements == 0); only that range is
// written. The ranges of one output page are disjoint, 16-byte aligned and cover the whole physical tile, so the cores
// sharing a page never write the same byte and every padding byte is zeroed by exactly one of them. The scratch offset
// equals the page offset, so source and destination stay congruent for the NoC write.
#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/kernels/riscv_copy.h"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using ttnn::prim::detail::SegmentMapData;

// Segments congruent modulo 16 bytes go through the NoC and are ordered by the async_write_barrier before the tile
// write-out. The rest are RISC copies; a NoC barrier does not order those stores, so the write-out drains the last
// one first. Returns the last RISC store so far.
template <uint32_t tile_bytes>
FORCE_INLINE uint32_t copy_segment(Noc noc, uint32_t dst, uint32_t src, uint32_t bytes, uint32_t last_store) {
    if ((dst & OFFSET_16) == (src & OFFSET_16)) {
        tt::data_movement::common::tt_memmove<false, true, false, tile_bytes>(noc, dst, src, bytes);
        return last_store;
    }
    return codegen_kernels::riscv_copy_elements(dst, src, bytes);
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t end_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("tile_bytes");
    constexpr uint32_t max_entries = get_named_compile_time_arg_val("max_entries");
    constexpr uint32_t element_bytes = get_named_compile_time_arg_val("element_bytes");
    constexpr uint32_t map_cb = get_named_compile_time_arg_val("map_cb");
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    constexpr bool sliced = get_named_compile_time_arg_val("sliced") != 0;
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr uint32_t first_entry = sliced ? 1 : 0;
    static_assert(max_entries >= first_entry, "sliced map rows lead with a header entry");

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer output_scratch(output_cb);
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);
    // Private scratch that is never pushed, so the write pointer stays at the buffer base.
    const uint32_t output_l1 = output_scratch.get_write_ptr();
    const CoreLocalMem<uint8_t> output_mem(output_l1);

    // The last input tile stays at the front across rows; have_input says one is held.
    bool have_input = false;
    for (uint32_t row = start_row; row < end_row; ++row) {
        // Segments cover only logical datums; the padding must not inherit the previous row's bytes.
        noc.async_write_zeros(output_scratch, tile_bytes);
        noc.write_zeros_l1_barrier();

        map_buffer.wait_front(1);
        // wait_front does not invalidate. With the Blackhole L1 data cache enabled, the RISC loads below could hit
        // lines cached from this CB slot's previous contents instead of the reader's NoC write.
        invalidate_l1_cache();
        CoreLocalMem<volatile SegmentMapData> segments(map_buffer.get_read_ptr());
        uint32_t output_page = row;
        uint32_t range_start = 0;
        uint32_t range_bytes = tile_bytes;
        if constexpr (sliced) {
            output_page = segments[0].input_page_index;
            range_start = segments[0].input_page_offset;
            range_bytes = segments[0].output_page_offset;
        }

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
                // The RISC copies read this tile too, so the same invalidation applies.
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
            codegen_kernels::drain_stores(last_store);
        }

        noc.async_write<NocOptions::DEFAULT, tile_bytes>(
            output_mem,
            output_accessor,
            range_bytes,
            {.offset_bytes = range_start},
            {.page_id = output_page, .offset_bytes = range_start});
        noc.async_writes_flushed();
        map_buffer.pop_front(1);
    }
    noc.async_write_barrier();
    if (have_input) {
        input_buffer.pop_front(1);
    }
}
