// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_struct_common.hpp"

using tt::data_movement::common::enhanced_noc_async_read;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;
constexpr uint32_t One_Tile_Reserve = 1;

void kernel_main() {
    constexpr uint32_t Tile_Size_Bytes = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(1);

    uint32_t num_templates = get_arg_val<uint32_t>(0);
    uint32_t num_short_runs = get_arg_val<uint32_t>(1);
    uint32_t num_long_runs = get_arg_val<uint32_t>(2);
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(3);

    constexpr auto input_args = TensorAccessorArgs<2>();
    const auto input_addr_gen = TensorAccessor(input_args, input_buffer_addr, Tile_Size_Bytes);

    DPRINT << "num_templates=" << num_templates << " num_short_runs=" << num_short_runs
           << " num_long_runs=" << num_long_runs << "\n";

    // Unpack pattern templates
    struct PatternTemplate {
        int32_t input_page_stride;
        int32_t input_offset_stride;
        int32_t output_offset_stride;
        uint32_t num_elements;
    };
    PatternTemplate templates[num_templates];
    uint32_t tmpl_base = 4;
    for (uint32_t i = 0; i < num_templates; ++i) {
        templates[i].input_page_stride = get_arg_val<uint32_t>(tmpl_base + i * 4 + 0);
        templates[i].input_offset_stride = get_arg_val<uint32_t>(tmpl_base + i * 4 + 1);
        templates[i].output_offset_stride = get_arg_val<uint32_t>(tmpl_base + i * 4 + 2);
        templates[i].num_elements = get_arg_val<uint32_t>(tmpl_base + i * 4 + 3);
    }

    uint32_t short_runs_base = tmpl_base + num_templates * 4;
    uint32_t long_runs_base = short_runs_base + num_short_runs * 6;

    uint32_t previous_input_page_idx = std::numeric_limits<uint32_t>::max();
    bool first = true;

    // Process short runs (run_length = 1)
    for (uint32_t i = 0; i < num_short_runs; ++i) {
        uint32_t out_page_start = get_arg_val<uint32_t>(short_runs_base + i * 6 + 0);
        uint32_t out_page_end = get_arg_val<uint32_t>(short_runs_base + i * 6 + 1);
        uint32_t in_page_start = get_arg_val<uint32_t>(short_runs_base + i * 6 + 2);
        uint32_t pattern_template_index = get_arg_val<uint32_t>(short_runs_base + i * 6 + 3);
        uint32_t in_offset_start = get_arg_val<uint32_t>(short_runs_base + i * 6 + 4);
        uint32_t out_offset_start = get_arg_val<uint32_t>(short_runs_base + i * 6 + 5);

        const auto& tmpl = templates[pattern_template_index];

        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start;  // No stride for short runs
            uint32_t input_offset = in_offset_start;

            if (tmpl.num_elements == 0) {
                continue;
            }

            DPRINT << "short_run i=" << i << " out_page_idx=" << out_page_idx << "\n";
            DPRINT << "input_page_idx=" << input_page_idx << " input_offset=" << input_offset << "\n";

            // For short runs, run_length is always 1
            for (uint32_t seg = 0; seg < 1; ++seg) {
                if (first) {
                    first = false;
                } else if (input_page_idx == previous_input_page_idx) {
                    continue;
                }

                cb_reserve_back(input_cb_id, 1);
                const uint32_t input_write_addr = get_read_ptr(input_cb_id);
                const uint64_t input_page_noc_addr = get_noc_addr(input_page_idx, input_addr_gen);

                enhanced_noc_async_read<Tile_Size_Bytes, true>(input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
                noc_async_read_barrier();
                cb_push_back(input_cb_id, 1);
                previous_input_page_idx = input_page_idx;
            }
        }
    }

    // Process long runs (run_length > 1)
    for (uint32_t i = 0; i < num_long_runs; ++i) {
        uint32_t out_page_start = get_arg_val<uint32_t>(long_runs_base + i * 10 + 0);
        uint32_t out_page_end = get_arg_val<uint32_t>(long_runs_base + i * 10 + 1);
        uint32_t in_page_start = get_arg_val<uint32_t>(long_runs_base + i * 10 + 2);
        uint32_t pattern_template_index = get_arg_val<uint32_t>(long_runs_base + i * 10 + 3);
        uint32_t in_offset_start = get_arg_val<uint32_t>(long_runs_base + i * 10 + 4);
        uint32_t out_offset_start = get_arg_val<uint32_t>(long_runs_base + i * 10 + 5);
        uint32_t run_length = get_arg_val<uint32_t>(long_runs_base + i * 10 + 6);
        int32_t in_page_stride = get_arg_val<uint32_t>(long_runs_base + i * 10 + 7);
        int32_t in_offset_stride = get_arg_val<uint32_t>(long_runs_base + i * 10 + 8);
        int32_t out_offset_stride = get_arg_val<uint32_t>(long_runs_base + i * 10 + 9);

        const auto& tmpl = templates[pattern_template_index];

        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start + (out_page_idx - out_page_start) * in_page_stride;
            uint32_t input_offset = in_offset_start + (out_page_idx - out_page_start) * in_offset_stride;

            for (uint32_t seg = 0; seg < run_length; ++seg) {
                if (tmpl.num_elements == 0) {
                    continue;
                }
                uint32_t seg_input_page_idx = input_page_idx + seg * tmpl.input_page_stride;

                DPRINT << "long_run i=" << i << " out_page_idx=" << out_page_idx << " seg=" << seg << "\n";
                DPRINT << "input_page_idx=" << input_page_idx << " input_offset=" << input_offset << "\n";

                if (first) {
                    first = false;
                } else if (seg_input_page_idx == previous_input_page_idx) {
                    continue;
                }

                cb_reserve_back(input_cb_id, 1);
                const uint32_t input_write_addr = get_read_ptr(input_cb_id);
                const uint64_t input_page_noc_addr = get_noc_addr(seg_input_page_idx, input_addr_gen);

                enhanced_noc_async_read<Tile_Size_Bytes, true>(input_page_noc_addr, input_write_addr, Tile_Size_Bytes);
                noc_async_read_barrier();
                cb_push_back(input_cb_id, 1);
                previous_input_page_idx = seg_input_page_idx;
            }
        }
    }

    DPRINT << "end of reader kernel\n";
}
