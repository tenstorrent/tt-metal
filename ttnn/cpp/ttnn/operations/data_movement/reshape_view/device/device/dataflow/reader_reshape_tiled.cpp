// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "dprint.h"

#include "debug/dprint_pages.h"

#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

inline void print_u32_pages(
    uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t idx, uint32_t start = 0) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << idx << " " << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << U32(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t map_addr = get_arg_val<uint32_t>(1);
    uint32_t start_output_page_idx = get_arg_val<uint32_t>(2);
    uint32_t end_output_page_idx = get_arg_val<uint32_t>(3);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Max_Map_Size_Bytes = get_compile_time_arg_val(1);
    constexpr uint32_t Tile_Size_Bytes = get_compile_time_arg_val(2);

    constexpr uint32_t Max_Map_Entries = Max_Map_Size_Bytes / sizeof(SegmentMapData);
    constexpr uint32_t Max_Map_Elements = Max_Map_Entries * SegmentMapData::size;

    constexpr auto mapping_cb_id = tt::CBIndex::c_0;
    constexpr auto input_cb_id = tt::CBIndex::c_1;

    const DataFormat input_data_format = get_dataformat(input_cb_id);
    const DataFormat map_data_format = get_dataformat(mapping_cb_id);

    const InterleavedAddrGenFast<input_is_dram> input_addr_gen = {
        .bank_base_address = input_addr, .page_size = Tile_Size_Bytes, .data_format = input_data_format};

    const InterleavedAddrGen<true> map_addr_gen = {
        .bank_base_address = map_addr, .page_size = Max_Map_Size_Bytes};  //, .data_format = map_data_format};

    bool first = true;
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        // DPRINT << "READER: RESERVE BACK MAP OUT PAGE: " <<  out_page_idx <<"\n";
        cb_reserve_back(mapping_cb_id, One_Tile_Reserve);
        const uint64_t map_noc_addr = get_noc_addr(out_page_idx, map_addr_gen);
        const uint32_t map_addr = get_read_ptr(mapping_cb_id);
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(map_noc_addr, map_addr, Max_Map_Size_Bytes);
        // DPRINT << "READER: BARRIER 1" << "\n";

        noc_async_read_barrier();
        cb_push_back(mapping_cb_id, 1);

        // DPRINT << " READER MAP OUT PAGE IDX: " << out_page_idx << "END OUT PAGE: "<< end_output_page_idx <<"\n";

        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);

        // DEBUG

        if (out_page_idx == 5) {
            print_u32_pages(map_addr, Max_Map_Size_Bytes / sizeof(uint32_t), 1, out_page_idx);
        }

        uint32_t previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t map_idx = 0; map_idx < Max_Map_Entries; ++map_idx) {
            if (map_ptr[map_idx].num_elements == 0) {
                continue;
            }

            const uint32_t input_page_idx = map_ptr[map_idx].input_page_index;
            if (out_page_idx == 4) {
                DPRINT << "INPUT PAGE IDX: " << input_page_idx << "\n";
            }

            if (first) {
                first = false;
            } else {
                // this segment is also in a tile we've already loaded
                if (input_page_idx == previous_input_page_idx) {
                    // DPRINT << "READER SKIPPING LOAD INPUT INDEX: "<< input_page_idx << "\n";
                    continue;
                }
            }

            // DPRINT <<"READE RESERVE BACK INPUT out page: "<< out_page_idx << " in page: "<<input_page_idx <<"\n";

            cb_reserve_back(input_cb_id, One_Tile_Reserve);
            const uint32_t input_write_addr = get_read_ptr(input_cb_id);
            const uint64_t input_page_noc_addr = get_noc_addr(input_page_idx, input_addr_gen);

            // DPRINT << " IN PAGE IDX: " << input_page_idx <<" IN NOC ADDR: "<< input_page_noc_addr<< " " <<
            //             map_ptr[map_idx].input_page_offset<<" "<< map_ptr[map_idx].output_page_offset<<" "
            //             <<map_ptr[map_idx].num_elements  <<"\n";

            enhanced_noc_async_read<Tile_Size_Bytes, true>(input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
            previous_input_page_idx = input_page_idx;

            // DPRINT << "READER: BARRIER 2" << "\n";
            noc_async_read_barrier();

            if (false) {
                if (out_page_idx == 5) {
                    DPRINT << "READER OUT PAGE " << input_page_idx << " prev input page: " << previous_input_page_idx
                           << "\n";
                    // tt::data_movement::common::print_bf16_pages(input_write_addr, 1024,1);
                    DPRINT << "\n";
                }
            }

            cb_push_back(input_cb_id, 1);

            // tt::data_movement::common::print_bf16_pages(input_write_addr, 1024,1);
        }
        // DPRINT << "READER DONE OUT PAGE INDEX: "<<out_page_idx<<"\n";
    }
}
