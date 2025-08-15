// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/hostdevcommon/common.hpp"

using namespace tt::data_movement::common;
using ttnn::operations::data_movement::reshape::detail::SegmentMapData;

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
    constexpr bool is_bfp8 = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(7);
    constexpr bool last_is_two_facelines = get_compile_time_arg_val(8);
    constexpr uint32_t faceline_row = get_compile_time_arg_val(9);
    constexpr uint32_t faceline_size = get_compile_time_arg_val(10);
    constexpr uint32_t exp_cb_idx = get_compile_time_arg_val(11);
    constexpr uint32_t exponents_size = get_compile_time_arg_val(12);
    constexpr auto output_args = TensorAccessorArgs<13>();

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, Tile_size_bytes);

    // loop over output (reshaped) pages this core is responsible for
    bool first = true;

    cb_reserve_back(cb_id_working, 1);
    const uint32_t working_write_addr = get_write_ptr(cb_id_working);
    for (uint32_t output_page_idx = start_output_page; output_page_idx < end_output_page; ++output_page_idx) {
        const uint32_t exp_ptr_addr = get_write_ptr(exp_cb_idx);
        bool last_tile = ((output_page_idx + 1) % tiles_per_row == 0);
        bool two_facelines = (last_tile && last_is_two_facelines) || !last_tile;
        cb_wait_front(cb_id_mapping, 1);
        const uint32_t map_addr = get_read_ptr(cb_id_mapping);
        auto map_ptr = reinterpret_cast<volatile tt_l1_ptr SegmentMapData*>(map_addr);
        uint32_t input_base_addr, previous_input_page_idx = std::numeric_limits<uint32_t>::max();
        uint32_t faceline_ctr = 0;
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
            uint32_t input_page_offset = map_ptr[seg_idx].input_page_offset * element_sz_bytes;
            uint32_t output_page_offset = map_ptr[seg_idx].output_page_offset * element_sz_bytes;

            const uint32_t output_addr = working_write_addr + output_page_offset;
            const uint32_t input_addr = input_base_addr + input_page_offset;
            const uint32_t szbytes = map_ptr[seg_idx].num_elements * element_sz_bytes;

            // for bfp8, the exponents are stored in the first 64 bytes of the tile in faceline order
            // this kernel stores the exponents corresponsing to the output shape in a separate buffer
            // then copies them to the appropriate address at the end
            if constexpr (is_bfp8) {
                volatile tt_l1_ptr uint8_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_base_addr);
                volatile tt_l1_ptr uint8_t* exp_buffer = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(exp_ptr_addr);
                uint32_t subtile_id = input_page_offset / faceline_size;
                uint32_t row_id = (input_page_offset % faceline_size) / faceline_row;
                uint32_t output_row_id = (faceline_ctr / 2) % faceline_row;
                uint32_t output_subtile_id = output_page_offset / faceline_size;
                uint32_t exp_idx = faceline_row * output_subtile_id + output_row_id;
                exp_buffer[exp_idx] = input_ptr[faceline_row * subtile_id + row_id];

                if (map_ptr[seg_idx].num_elements > faceline_row) {  // if reading more than one faceline at a time
                    for (uint32_t f = 1; f < map_ptr[seg_idx].num_elements / faceline_row; f++) {
                        exp_buffer[exp_idx + f] = input_ptr[faceline_row * subtile_id + row_id + f];
                        faceline_ctr = two_facelines ? faceline_ctr + 1 : faceline_ctr + 2;
                    }
                }
                faceline_ctr = two_facelines ? faceline_ctr + 1 : faceline_ctr + 2;
            }
            // skip exponents for bfloat8_b when moving data
            tt_memmove<false, true, false, Tile_size_bytes>(
                output_addr + exponents_size * is_bfp8, input_addr + exponents_size * is_bfp8, szbytes);
        }
        noc_async_write_barrier();

        if constexpr (is_bfp8) {
            // copy exponents for bfloat8_b
            memmove((void*)working_write_addr, (const void*)exp_ptr_addr, exponents_size);
        }

        const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
        enhanced_noc_async_write<Tile_size_bytes, true>(working_write_addr, output_noc_addr, Tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_mapping, 1);
    }
    cb_push_back(cb_id_working, 1);
}
