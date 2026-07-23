// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using namespace tt::data_movement::common;
using ttnn::prim::detail::SegmentMapData;

void kernel_main() {
    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);

    const uint32_t start_output_page = get_arg_val<uint32_t>(1);
    const uint32_t end_output_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t Tile_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Max_Map_Entries = get_compile_time_arg_val(1);
    constexpr uint8_t element_sz_bytes = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_mapping = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_input = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_working = get_compile_time_arg_val(5);  // scratch
    constexpr auto output_args = TensorAccessorArgs<6>();

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr);

    Noc noc;
    CircularBuffer cb_mapping(cb_id_mapping);
    CircularBuffer cb_input(cb_id_input);
    CircularBuffer cb_working(cb_id_working);
    // loop over output (reshaped) pages this core is responsible for
    bool first = true;
    cb_working.reserve_back(1);
    const uint32_t working_write_addr = cb_working.get_write_ptr();
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        cb_mapping.wait_front(1);
        const uint32_t map_addr = cb_mapping.get_read_ptr();
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t seg_idx = 0; seg_idx < Max_Map_Entries; ++seg_idx) {
            if (map_ptr[seg_idx].num_elements == 0) {
                continue;
            }

            if (first) {
                cb_input.wait_front(1);
                input_base_addr = cb_input.get_read_ptr();
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
                first = false;

            } else if (map_ptr[seg_idx].input_page_index != previous_input_page_idx) {
                noc.async_write_barrier();
                cb_input.pop_front(1);
                cb_input.wait_front(1);
                input_base_addr = cb_input.get_read_ptr();
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
            }
            // TODO (maybe) pre calculate size and offsets in bytes on host
            const uint32_t output_addr = working_write_addr + map_ptr[seg_idx].output_page_offset * element_sz_bytes;
            const uint32_t input_addr = input_base_addr + map_ptr[seg_idx].input_page_offset * element_sz_bytes;
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;
            tt_memmove<false, true, false, Tile_size_bytes>(noc, output_addr, input_addr, szbytes);
        }
        noc.async_write_barrier();

        const uint64_t output_noc_addr = output_addrgen.get_noc_addr(output_page_idx);
        enhanced_noc_async_write<Tile_size_bytes, true>(noc, working_write_addr, output_noc_addr, Tile_size_bytes);
        noc.async_write_barrier();

        cb_mapping.pop_front(1);
    }
    // The per-transition pop only releases the previous input page's tile, so the final input
    // tile waited inside the loop is still held here. Pop it once to leave the input CB balanced.
    // Gated on `first` so a core that processed no segments does not pop a tile it never waited.
    if (!first) {
        noc.async_write_barrier();
        cb_input.pop_front(1);
    }
    cb_working.push_back(1);
}
