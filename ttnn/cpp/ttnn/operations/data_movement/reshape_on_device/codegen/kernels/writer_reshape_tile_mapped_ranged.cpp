// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ranged direct TILE reshape writer (TTDM_RESHAPE_MAPPED_SEGMENT_SPLIT,
// kill-list #11). Gated sibling of writer_reshape_tile_mapped.cpp for SLICED
// map rows: entry 0 of every map row is a host-built HEADER
//   { output_page_id, range_start_bytes, range_len_bytes, num_elements = 0 }
// followed by exactly the segments of one contiguous datum range of that
// output tile. The core assembles its segments into a zero-filled scratch
// tile at their natural offsets, then writes ONLY its
// [range_start, range_start + range_len) bytes to the output page.
//
// Correctness of the shared-page write: the host guarantees that the ranges
// of one output page are disjoint, 16-byte-quantized, and cover the full
// physical tile, so N cores write disjoint byte ranges of the SAME page with
// no race (precedent: writer_reshape_rm.cpp PARTIAL==1) and every pad byte
// is zero-written by exactly one owner (its range's). The L1 source offset
// inside the scratch tile EQUALS the DRAM destination offset inside the
// page, so the NOC write-alignment rule (src/dst congruent modulo
// NOC_DRAM_WRITE_ALIGNMENT_BYTES) holds by construction.
//
// Both mapped readers skip the header untouched (num_elements == 0), so the
// reader<->writer push/pop order contract — wait one input tile per distinct
// ADJACENT input page of the row, pop lazily on the next distinct page — is
// byte-identical to the base writer's.
#include <cstdint>
#include <limits>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

struct SegmentMapData {
    uint32_t input_page_index;
    uint32_t input_page_offset;
    uint32_t output_page_offset;
    uint32_t num_elements;
};

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t max_entries = get_compile_time_arg_val(1);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t map_cb = get_compile_time_arg_val(3);
    constexpr uint32_t input_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();
    static_assert(max_entries >= 1, "sliced map rows lead with a header entry");

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    Noc noc;
    CircularBuffer output_scratch(output_cb);
    CircularBuffer map_buffer(map_cb);
    CircularBuffer input_buffer(input_cb);
    output_scratch.reserve_back(1);
    // Raw L1 base retained: tt_memmove assembles the tile via raw L1 addresses.
    const uint32_t output_l1 = output_scratch.get_write_ptr();
    // Same reserved slot, as a Device 2.0 local-L1 handle for the range write out.
    const CoreLocalMem<uint8_t> output_mem(output_l1);

    bool have_input = false;
    uint32_t input_l1 = 0;
    uint32_t previous_page = std::numeric_limits<uint32_t>::max();
    for (uint32_t row = start_row; row < end_row; ++row) {
        // Only logical elements have mapping segments. Clear the complete
        // physical tile before assembly so the pad bytes of this row's range
        // never inherit stale L1 data from the preceding row.
        noc.async_write_zeros(output_scratch, tile_bytes);
        noc.write_zeros_l1_barrier();

        map_buffer.wait_front(1);
        const uint32_t map_l1 = map_buffer.get_read_ptr();
        CoreLocalMem<volatile SegmentMapData> segments(map_l1);
        // Row header (skipped by the readers: num_elements == 0).
        const uint32_t output_page = segments[0].input_page_index;
        const uint32_t range_start = segments[0].input_page_offset;
        const uint32_t range_bytes = segments[0].output_page_offset;
        previous_page = std::numeric_limits<uint32_t>::max();

        for (uint32_t index = 1; index < max_entries; ++index) {
            if (segments[index].num_elements == 0) {
                continue;
            }
            if (segments[index].input_page_index != previous_page) {
                if (have_input) {
                    noc.async_write_barrier();
                    input_buffer.pop_front(1);
                }
                input_buffer.wait_front(1);
                input_l1 = input_buffer.get_read_ptr();
                previous_page = segments[index].input_page_index;
                have_input = true;
            }
            const uint32_t dst = output_l1 + segments[index].output_page_offset * element_bytes;
            const uint32_t src = input_l1 + segments[index].input_page_offset * element_bytes;
            const uint32_t bytes = segments[index].num_elements * element_bytes;
            tt_memmove<false, true, false, tile_bytes>(noc, dst, src, bytes);
        }
        noc.async_write_barrier();
        noc.async_write<NocOptions::DEFAULT, tile_bytes>(
            output_mem, output_accessor, range_bytes,
            {.offset_bytes = range_start},
            {.page_id = output_page, .offset_bytes = range_start});
        noc.async_write_barrier();
        map_buffer.pop_front(1);
    }
    if (have_input) {
        noc.async_write_barrier();
        input_buffer.pop_front(1);
    }
    output_scratch.push_back(1);
}
