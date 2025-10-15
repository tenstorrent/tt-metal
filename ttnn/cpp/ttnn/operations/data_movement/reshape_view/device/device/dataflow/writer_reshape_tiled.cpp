// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_kernel_common.hpp"

using namespace tt::data_movement::common;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;
using ttnn::operations::data_movement::reshape::detail::unpack_rt_short;
using ttnn::operations::data_movement::reshape::detail::unpack_short_run_ultra;

void kernel_main() {
    uint32_t num_templates = get_arg_val<uint32_t>(0);
    uint32_t num_short_runs = get_arg_val<uint32_t>(1);
    uint32_t num_long_runs = get_arg_val<uint32_t>(2);
    const uint32_t output_base_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t Tile_size_bytes = get_compile_time_arg_val(0);
    constexpr uint8_t element_sz_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_input = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_working = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, Tile_size_bytes);

    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);

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
    uint32_t long_runs_base = short_runs_base + num_short_runs * 3;

    uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
    bool first = true;

    // Process short runs (run_length = 1)
    for (uint32_t i = 0; i < num_short_runs; ++i) {
        auto [out_page_start, out_page_end] = unpack_rt_short(get_arg_val<uint32_t>(short_runs_base + i * 3 + 0));
        auto packed_template = get_arg_val<uint32_t>(short_runs_base + i * 3 + 1);
        auto [in_offset_start, out_offset_start] = unpack_rt_short(get_arg_val<uint32_t>(short_runs_base + i * 3 + 2));
        uint32_t in_page_start = packed_template >> 16;
        uint32_t pattern_template_index = (packed_template >> 8) & 0xFF;
        const auto& tmpl = templates[pattern_template_index];

        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start;
            uint32_t input_offset = in_offset_start;
            uint32_t output_offset = out_offset_start;

            if (tmpl.num_elements == 0) {
                continue;
            }

            if (first) {
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = input_page_idx;
                first = false;
            } else if (input_page_idx != previous_input_page_idx) {
                noc_async_write_barrier();
                cb_pop_front(cb_id_input, 1);
                cb_wait_front(cb_id_input, 1);
                input_base_addr = get_read_ptr(cb_id_input);
                previous_input_page_idx = input_page_idx;
            }

            // For short runs, run_length is always 1
            for (uint32_t seg = 0; seg < 1; ++seg) {
                const uint32_t output_addr = working_write_addr + output_offset * element_sz_bytes;
                const uint32_t input_addr = input_base_addr + input_offset * element_sz_bytes;
                uint32_t szbytes = tmpl.num_elements * element_sz_bytes;
                tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);
            }

            noc_async_write_barrier();
            const uint64_t output_noc_addr = get_noc_addr(out_page_idx, output_addrgen);
            enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
            noc_async_write_barrier();
        }
    }

    for (uint32_t i = 0; i < num_long_runs; ++i) {
        auto [out_page_start, out_page_end] = unpack_rt_short(get_arg_val<uint32_t>(long_runs_base + i * 5 + 0));
        auto [in_page_start, pattern_template_index] =
            unpack_rt_short(get_arg_val<uint32_t>(long_runs_base + i * 5 + 1));
        auto [in_offset_start, out_offset_start] = unpack_rt_short(get_arg_val<uint32_t>(long_runs_base + i * 5 + 2));
        auto [run_length, in_page_stride] = unpack_rt_short(get_arg_val<uint32_t>(long_runs_base + i * 5 + 3));
        auto [in_offset_stride, out_offset_stride] = unpack_rt_short(get_arg_val<uint32_t>(long_runs_base + i * 5 + 4));

        const auto& tmpl = templates[pattern_template_index];

        for (uint32_t out_page_idx = out_page_start; out_page_idx <= out_page_end; ++out_page_idx) {
            uint32_t input_page_idx = in_page_start + (out_page_idx - out_page_start) * in_page_stride;
            uint32_t input_offset = in_offset_start + (out_page_idx - out_page_start) * in_offset_stride;
            uint32_t output_offset = out_offset_start + (out_page_idx - out_page_start) * out_offset_stride;

            for (uint32_t seg = 0; seg < run_length; ++seg) {
                if (tmpl.num_elements == 0) {
                    continue;
                }

                uint32_t seg_input_page_idx = input_page_idx + seg * tmpl.input_page_stride;
                uint32_t seg_input_offset = input_offset + seg * tmpl.input_offset_stride;
                uint32_t seg_output_offset = output_offset + seg * tmpl.output_offset_stride;

                // Handle input page switching
                if (first) {
                    cb_wait_front(cb_id_input, 1);
                    input_base_addr = get_read_ptr(cb_id_input);
                    previous_input_page_idx = seg_input_page_idx;
                    first = false;
                } else if (seg_input_page_idx != previous_input_page_idx) {
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_input, 1);
                    cb_wait_front(cb_id_input, 1);
                    input_base_addr = get_read_ptr(cb_id_input);
                    previous_input_page_idx = seg_input_page_idx;
                }

                const uint32_t output_addr = working_write_addr + seg_output_offset * element_sz_bytes;
                const uint32_t input_addr = input_base_addr + seg_input_offset * element_sz_bytes;
                uint32_t szbytes = tmpl.num_elements * element_sz_bytes;

                tt_memmove<false, true, false, Tile_size_bytes>(output_addr, input_addr, szbytes);
            }

            // Write output page
            noc_async_write_barrier();
            const uint64_t output_noc_addr = get_noc_addr(out_page_idx, output_addrgen);
            enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
            noc_async_write_barrier();
        }
    }
    cb_push_back(cb_id_working, 1);
}
