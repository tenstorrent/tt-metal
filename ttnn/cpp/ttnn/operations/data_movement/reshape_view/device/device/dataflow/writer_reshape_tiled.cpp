// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tuple>

#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"
#include "dataflow_api.h"
#include "debug/assert.h"
#include "dprint_pages.h"
#include "dprint.h"

using namespace tt::data_movement::common;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;

void kernel_main() {
    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);

    const uint32_t start_output_page = get_arg_val<uint32_t>(1);
    const uint32_t end_output_page = get_arg_val<uint32_t>(2);

    constexpr bool output_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t Tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t Max_Map_Entries = get_compile_time_arg_val(2);

    constexpr uint8_t element_sz_bytes = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_mapping = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_input = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_working = tt::CBIndex::c_2;  // scratch

    const DataFormat input_data_format = get_dataformat(cb_id_input);

    //  TODO sharded
    const InterleavedAddrGenFast<output_is_dram> output_addrgen = {
        .bank_base_address = output_base_addr, .page_size = Tile_size_bytes, .data_format = input_data_format};

    // loop over output (reshaped) pages this core is responsible for
    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        // DEBUG
        // auto working_write_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(working_write_addr);
        // for(uint32_t i =0; i < Tile_size_bytes/element_sz_bytes; ++i)working_write_ptr[i]=-1;

        // DPRINT << " WRITER OUT PAGE IDX: " << output_page_idx << "\n";
        // DPRINT << "WRITER wait front map"<< "\n";
        cb_wait_front(cb_id_mapping, 1);
        const uint32_t map_addr = get_read_ptr(cb_id_mapping);
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t previous_input_page_idx, input_base_addr;
        bool first = true;
        for (uint32_t seg_idx = 0; seg_idx < Max_Map_Entries; ++seg_idx) {
            if (map_ptr[seg_idx].num_elements == 0) {
                cb_pop_front(cb_id_input, 1);
                break;
            }

            if (first) {
                // DPRINT << "WRITER wait front in 1"<< "\n";
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
                first = false;
            } else if (map_ptr[seg_idx].input_page_index != previous_input_page_idx) {
                // DPRINT << "WRITER wait front in 2. PREV PAGE: "<<previous_input_page_idx << "\n";
                cb_pop_front(cb_id_input, 1);
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
            }

            // DPRINT << " WRITER OUT PAGE: " << output_page_idx <<" IN PAGE: "<< map_ptr[seg_idx].input_page_index<<"
            // IN OFFSET: " <<map_ptr[seg_idx].input_page_offset<<" OUT OFFSET: "<<
            // map_ptr[seg_idx].output_page_offset<<" COUNT: "<<map_ptr[seg_idx].num_elements  <<"\n";
            // tt::data_movement::common::print_bf16_pages(input_base_addr, 1024,1);

            // TODO we could set up the mapping so this always aligned and add an additional mask step
            const uint32_t output_addr = working_write_addr + map_ptr[seg_idx].output_page_offset * element_sz_bytes;
            const uint32_t input_addr = input_base_addr + map_ptr[seg_idx].input_page_offset * element_sz_bytes;

            // TODO pre calculate size and offsets ^ in bytes on host
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;
            tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);

            // DEBUG!
            // noc_async_write_barrier();
        }
        // DPRINT << "WRITER barrier 1"<< "\n";
        noc_async_write_barrier();

        const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
        // DPRINT << "write barrier 1.5 " << "\n";
        enhanced_noc_async_write<Tile_size_bytes, false>(working_write_addr, output_noc_addr, Tile_size_bytes);
        // DPRINT << "WRITER barrier 2"<< "\n";
        noc_async_write_barrier();

        // DPRINT <<"WRITER DONE OUTPUT TILE: "<<output_page_idx<<"\n";
    }
    cb_push_back(cb_id_working, 1);
}
