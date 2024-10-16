// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/common.hpp"

void kernel_main() {
    uint32_t i = 0;
    // buffers
    uint32_t dst_addr = get_arg_val<uint32_t>(i++);

    // output
    uint32_t output_size_c_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_size_d_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_size_h_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_size_w_without_padding = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_n = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_c = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_d = get_arg_val<uint32_t>(i++);
    uint32_t output_noc_id_stride_h = get_arg_val<uint32_t>(i++);
    uint32_t output_num_stick_width = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);
    uint32_t stick_size = get_arg_val<uint32_t>(i++);
    uint32_t element_size = get_arg_val<uint32_t>(i++);
    uint32_t num_elements_per_alignment = get_arg_val<uint32_t>(i++);
    uint32_t num_alignment_width = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_out0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_out1 = tt::CB::c_out1;

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGen<dst_is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = 1024 * element_size,
    };

#define NOC_MINIMUM_READ_SIZE 32

    uint32_t l1_read_addr0 = get_read_ptr(cb_id_out0);
    uint32_t l1_read_addr1 = get_read_ptr(cb_id_out1);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t output_stick_w = i % num_alignment_width;
        uint32_t w_off = output_stick_w * num_elements_per_alignment;
        uint32_t w_start = w_off;
        uint32_t w_end = min(w_off + num_elements_per_alignment, output_size_w_without_padding);

        uint32_t stick_y = (i / num_alignment_width);
        uint32_t stick_x = w_start / FACE_WIDTH;
        uint32_t stick_idx = stick_y * output_num_stick_width + stick_x;

        Idx5d stick_index_5d = get_stick_indices(stick_idx,
                                                 output_size_c_without_padding,
                                                 output_size_d_without_padding,
                                                 output_size_h_without_padding,
                                                 output_num_stick_width);
        Idx5d tile_index_5d = get_tile_indices(stick_index_5d);

        uint32_t noc_id = tile_index_5d.n * output_noc_id_stride_n + tile_index_5d.c * output_noc_id_stride_c +
                          tile_index_5d.d * output_noc_id_stride_d + tile_index_5d.h * output_noc_id_stride_h +
                          tile_index_5d.w;

        uint32_t noc_offset = get_noc_offset_in_tile(stick_index_5d.h, stick_index_5d.w, tile_index_5d.h, element_size);

        if (num_elements_per_alignment == 8) {
            noc_offset += ((w_start / 8) % 2) * NOC_MINIMUM_READ_SIZE;
        }

        uint32_t j = 0;
        for (uint32_t w = w_start; w < w_end; w++, j++) {
            cb_wait_front(cb_id_out0, 1);

            if (element_size == 4) {
                volatile tt_l1_ptr uint32_t* index_l1_ptr0 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr0);
                volatile tt_l1_ptr uint32_t* index_l1_ptr1 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr1);

                index_l1_ptr1[j] = index_l1_ptr0[0];
            } else if (element_size == 2) {
                volatile tt_l1_ptr uint16_t* index_l1_ptr0 =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr0);
                volatile tt_l1_ptr uint16_t* index_l1_ptr1 =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr1);

                index_l1_ptr1[j] = index_l1_ptr0[0];
            }

            cb_pop_front(cb_id_out0, 1);
        }

        uint64_t dst_noc_addr = get_noc_addr(noc_id, s0, noc_offset);
        noc_async_write(l1_read_addr1, dst_noc_addr, NOC_MINIMUM_READ_SIZE);
        noc_async_write_barrier();
    }
}
