// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using namespace tt::data_movement::common;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;

/*
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

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, Tile_size_bytes);

    // loop over output (reshaped) pages this core is responsible for
    bool first = true;
    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        cb_wait_front(cb_id_mapping, 1);
        const uint32_t map_addr = get_read_ptr(cb_id_mapping);
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        for (uint32_t seg_idx = 0; seg_idx < Max_Map_Entries; ++seg_idx) {
            if (map_ptr[seg_idx].num_elements == 0) {
                if (output_page_idx == end_output_page - 1 && seg_idx == Max_Map_Entries - 1) {
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_input, 1);
                }
                continue;
            }

            if (first) {
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
                first = false;

            } else if (map_ptr[seg_idx].input_page_index != previous_input_page_idx) {
                noc_async_write_barrier();
                cb_pop_front(cb_id_input, 1);
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = map_ptr[seg_idx].input_page_index;
            }
            // TODO (maybe) pre calculate size and offsets in bytes on host
            const uint32_t output_addr = working_write_addr + map_ptr[seg_idx].output_page_offset * element_sz_bytes;
            const uint32_t input_addr = input_base_addr + map_ptr[seg_idx].input_page_offset * element_sz_bytes;
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;
            tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);
        }
        noc_async_write_barrier();

        const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
        enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_mapping, 1);
    }
    cb_push_back(cb_id_working, 1);
}
*/

void kernel_main() {
    uint32_t num_templates = get_arg_val<uint32_t>(0);
    const uint32_t output_base_addr = get_arg_val<uint32_t>(2);
    uint32_t num_runs = get_arg_val<uint32_t>(1);

    DPRINT << "all writer rt args: ";
    for (uint32_t i = 0; i < 3 + num_templates * 3 + num_runs * 3; ++i) {
        DPRINT << get_arg_val<uint32_t>(i) << " ";
    }
    DPRINT << "\n";

    DPRINT << "start of writer kernel\n";
    constexpr uint32_t Tile_size_bytes = get_compile_time_arg_val(0);
    constexpr uint8_t element_sz_bytes = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_input = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_working = get_compile_time_arg_val(5);  // scratch
    constexpr auto output_args = TensorAccessorArgs<6>();
    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, Tile_size_bytes);

    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);

    // Unpack pattern templates
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
    uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
    DPRINT << "num of runs: " << num_runs << "\n";
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

        DPRINT << "in_offset_start=" << in_offset_start << " out_offset_start=" << out_offset_start << "\n";
        const auto& tmpl = templates[template_idx];
        // For each output page in the run
        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start + (out_page_idx - out_page_start) * in_page_stride;
            uint32_t input_offset = in_offset_start + (out_page_idx - out_page_start) * in_offset_stride;
            uint32_t output_offset = out_offset_start + (out_page_idx - out_page_start) * out_offset_stride;

            DPRINT << "run_idx=" << run_idx << " out_page_idx=" << out_page_idx << "\n";
            DPRINT << "input_page_idx=" << input_page_idx << " input_offset=" << input_offset
                   << " output_offset=" << output_offset << "\n";
            if (input_page_idx != previous_input_page_idx) {
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = input_page_idx;
            }
            for (uint32_t seg = 0; seg < run_length; ++seg) {
                uint32_t seg_input_offset = input_offset + seg * tmpl.input_offset_stride;
                uint32_t seg_output_offset = output_offset + seg * tmpl.output_offset_stride;
                const uint32_t output_addr = working_write_addr + seg_output_offset * element_sz_bytes;
                DPRINT << "seg=" << seg << " seg_input_offset=" << seg_input_offset
                       << " seg_output_offset=" << seg_output_offset << "\n";
                DPRINT << "OUTPUT addr: " << output_addr << "\n";
                const uint32_t input_addr = input_base_addr + seg_input_offset * element_sz_bytes;

                uint32_t szbytes = tmpl.num_elements * element_sz_bytes;
                tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);
            }
            DPRINT << "after seg loop\n";

            noc_async_write_barrier();
            const uint64_t output_noc_addr = get_noc_addr(out_page_idx, output_addrgen);
            enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
            noc_async_write_barrier();

            cb_pop_front(cb_id_input, 1);
        }
        DPRINT << "after out_page loop\n";
    }
    cb_push_back(cb_id_working, 1);
    DPRINT << "end of writer kernel\n";
}
