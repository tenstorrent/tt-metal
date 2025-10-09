// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;
/*
void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t map_addr = get_arg_val<uint32_t>(1);
    uint32_t start_output_page_idx = get_arg_val<uint32_t>(2);
    uint32_t end_output_page_idx = get_arg_val<uint32_t>(3);

    constexpr uint32_t Max_Map_Size_Bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Tile_Size_Bytes = get_compile_time_arg_val(1);

    constexpr uint32_t mapping_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);

    constexpr auto map_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();

    constexpr uint32_t Max_Map_Entries = Max_Map_Size_Bytes / sizeof(SegmentMapData);
    constexpr uint32_t Max_Map_Elements = Max_Map_Entries * SegmentMapData::size;

    const auto input_addr_gen = TensorAccessor(input_args, input_addr, Tile_Size_Bytes);
    const auto map_addr_gen = TensorAccessor(map_args, map_addr, Max_Map_Size_Bytes);

    bool first = true;
    for (uint32_t out_page_idx = start_output_page_idx; out_page_idx < end_output_page_idx; ++out_page_idx) {
        cb_reserve_back(mapping_cb_id, One_Tile_Reserve);
        const uint64_t map_noc_addr = get_noc_addr(out_page_idx, map_addr_gen);
        const uint32_t map_addr = get_read_ptr(mapping_cb_id);
        enhanced_noc_async_read<Max_Map_Size_Bytes, true>(map_noc_addr, map_addr, Max_Map_Size_Bytes);
        noc_async_read_barrier();
        cb_push_back(mapping_cb_id, 1);

        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        DPRINT << "output_page_idx=" << out_page_idx << " map entries: " << Max_Map_Entries
               << " max elements: " << Max_Map_Elements << "\n";
        for (uint32_t map_idx = 0; map_idx < Max_Map_Entries; ++map_idx) {
            if (map_ptr[map_idx].num_elements == 0) {
                continue;
            }

            const uint32_t input_page_idx = map_ptr[map_idx].input_page_index;
            if (first) {
                first = false;
            } else {
                // this segment is also in a tile we've already loaded
                if (input_page_idx == previous_input_page_idx) {
                    continue;
                }
            }

            cb_reserve_back(input_cb_id, One_Tile_Reserve);
            const uint32_t input_write_addr = get_read_ptr(input_cb_id);
            const uint64_t input_page_noc_addr = get_noc_addr(input_page_idx, input_addr_gen);
            DPRINT << "input_page_noc_addr=" << input_page_noc_addr << " input_write_addr=" << input_write_addr
                   << "\n";
            enhanced_noc_async_read<Tile_Size_Bytes, true>(input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
            previous_input_page_idx = input_page_idx;
            noc_async_read_barrier();
            DPRINT << "printing what we just read\n";
            volatile tt_l1_ptr uint16_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_write_addr);
            for (uint16_t value = 0; value < Tile_Size_Bytes/2; value++) {
                DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value]) << ENDL();
            }
            DPRINT << "\n";
            cb_push_back(input_cb_id, 1);
        }
    }
}

*/
void kernel_main() {
    constexpr uint32_t Tile_Size_Bytes = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);
    constexpr uint8_t element_sz_bytes = 2;

    uint32_t num_templates = get_arg_val<uint32_t>(0);
    uint32_t num_runs = get_arg_val<uint32_t>(1);
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(2);

    constexpr auto map_args = TensorAccessorArgs<4>();
    constexpr auto input_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();

    const auto input_addr_gen = TensorAccessor(input_args, input_buffer_addr, Tile_Size_Bytes);

    DPRINT << "all reader rt args: ";
    for (uint32_t i = 0; i < 3 + num_templates * 3 + num_runs * 3; ++i) {
        DPRINT << get_arg_val<uint32_t>(i) << " ";
    }
    DPRINT << "\n";

    DPRINT << "start of reader kernel\n";
    // Unpack templates
    struct PatternTemplate {
        int32_t input_offset_stride;
        int32_t output_offset_stride;
        uint32_t num_elements;
    };
    PatternTemplate templates[num_templates];
    uint32_t tmpl_base = 3;
    for (uint32_t i = 0; i < num_templates; ++i) {
        templates[i].input_offset_stride = get_arg_val<uint32_t>(tmpl_base + i * 3 + 0);
        templates[i].output_offset_stride = get_arg_val<uint32_t>(tmpl_base + i * 3 + 1);
        templates[i].num_elements = get_arg_val<uint32_t>(tmpl_base + i * 3 + 2);
    }

    uint32_t runs_base = tmpl_base + num_templates * 3;
    uint32_t previous_input_page_idx = std::numeric_limits<uint32_t>::max();
    for (uint32_t run_idx = 0; run_idx < num_runs; ++run_idx) {
        uint32_t packed1 = get_arg_val<uint32_t>(runs_base + run_idx * 4 + 0);
        uint32_t input_offset_start = get_arg_val<uint32_t>(runs_base + run_idx * 4 + 1);
        uint32_t output_offset_start = get_arg_val<uint32_t>(runs_base + run_idx * 4 + 2);
        uint32_t packed2 = get_arg_val<uint32_t>(runs_base + run_idx * 4 + 3);

        uint8_t out_page_start = (packed1 >> 24) & 0xFF;
        uint8_t out_page_end = (packed1 >> 16) & 0xFF;
        uint8_t in_page_start = (packed1 >> 8) & 0xFF;
        uint8_t template_idx = packed1 & 0xFF;

        uint32_t in_offset_start = input_offset_start;
        uint32_t out_offset_start = output_offset_start;
        uint8_t run_length = (packed2 >> 24) & 0xFF;
        int8_t in_page_stride = (packed2 >> 16) & 0xFF;
        int8_t in_offset_stride = (packed2 >> 8) & 0xFF;
        int8_t out_offset_stride = packed2 & 0xFF;

        const auto& tmpl = templates[template_idx];

        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start + (out_page_idx - out_page_start) * in_page_stride;
            uint32_t input_offset = in_offset_start + (out_page_idx - out_page_start) * in_offset_stride;

            // Use run_length from the run, not tmpl.num_elements
            DPRINT << "run length: " << (uint32_t)run_length << " tmpl.num_elements: " << tmpl.num_elements << "\n";
            for (uint32_t seg = 0; seg < run_length; ++seg) {
                uint32_t seg_input_offset = input_offset + seg * tmpl.input_offset_stride;

                DPRINT << "run_idx=" << run_idx << " out_page_idx=" << out_page_idx << " seg=" << seg << "\n";
                DPRINT << "input_page_idx=" << input_page_idx << " seg_input_offset=" << seg_input_offset << "\n";

                if (input_page_idx != previous_input_page_idx) {
                    cb_reserve_back(input_cb_id, 1);
                    DPRINT << "after cb_reserve_back\n";
                    const uint32_t input_write_addr = get_read_ptr(input_cb_id);
                    DPRINT << "after get_read_ptr\n";
                    const uint64_t input_page_noc_addr = get_noc_addr(input_page_idx, input_addr_gen);
                    // const uint64_t input_page_noc_addr = input_buffer_addr + seg_input_offset * element_sz_bytes;
                    DPRINT << "input_page_noc_addr=" << input_page_noc_addr << " input_write_addr=" << input_write_addr
                           << "\n";
                    enhanced_noc_async_read<Tile_Size_Bytes, true>(
                        input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
                    DPRINT << "after enhanced_noc_async_read\n";
                    noc_async_read_barrier();
                    DPRINT << "print the data we jjust read\n";
                    volatile tt_l1_ptr uint16_t* dst_noc2 =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_write_addr);
                    for (uint16_t value = 0; value < Tile_Size_Bytes / 2; value++) {
                        DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value])
                               << ENDL();
                    }
                    DPRINT << "\n";
                    DPRINT << "after noc_async_read_barrier\n";
                    cb_push_back(input_cb_id, 1);
                    previous_input_page_idx = input_page_idx;
                }
            }
            DPRINT << "after seg loop\n";
        }
        DPRINT << "after out_page loop\n";
    }
    DPRINT << "end of reader kernel\n";
}
