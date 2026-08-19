// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // output
    uint32_t output_size_c_without_padding = get_arg(args::output_size_c_without_padding);
    uint32_t output_size_d_without_padding = get_arg(args::output_size_d_without_padding);
    uint32_t output_size_h_without_padding = get_arg(args::output_size_h_without_padding);
    uint32_t output_size_w_without_padding = get_arg(args::output_size_w_without_padding);
    uint32_t output_noc_id_stride_n = get_arg(args::output_noc_id_stride_n);
    uint32_t output_noc_id_stride_c = get_arg(args::output_noc_id_stride_c);
    uint32_t output_noc_id_stride_d = get_arg(args::output_noc_id_stride_d);
    uint32_t output_noc_id_stride_h = get_arg(args::output_noc_id_stride_h);
    uint32_t output_num_stick_width = get_arg(args::output_num_stick_width);

    // etc
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_sticks = get_arg(args::num_sticks);
    uint32_t stick_size = get_arg(args::stick_size);
    uint32_t element_size = get_arg(args::element_size);
    uint32_t num_elements_per_alignment = get_arg(args::num_elements_per_alignment);
    uint32_t num_alignment_width = get_arg(args::num_alignment_width);

    const auto s0 = TensorAccessor(tensor::s0);

#define NOC_MINIMUM_READ_SIZE 32

    Noc noc;
    // out0 is the buffer the reader staged each selected element in; out1 is where this kernel
    // assembles one aligned run of them before writing that run out to the output tensor.
    DataflowBuffer dfb_out0_obj(dfb::out0);
    DataflowBuffer dfb_out1_obj(dfb::out1);

    uint32_t l1_read_addr0 = dfb_out0_obj.get_read_ptr();
    uint32_t l1_read_addr1 = dfb_out1_obj.get_read_ptr();

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t output_stick_w = i % num_alignment_width;
        uint32_t w_off = output_stick_w * num_elements_per_alignment;
        uint32_t w_start = w_off;
        uint32_t w_end = std::min(w_off + num_elements_per_alignment, output_size_w_without_padding);

        uint32_t stick_y = (i / num_alignment_width);
        uint32_t stick_x = w_start / FACE_WIDTH;
        uint32_t stick_idx = stick_y * output_num_stick_width + stick_x;

        Idx5d stick_index_5d = get_stick_indices(
            stick_idx,
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
            dfb_out0_obj.wait_front(1);

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

            dfb_out0_obj.pop_front(1);
        }

        noc.async_write(
            dfb_out1_obj,
            s0,
            NOC_MINIMUM_READ_SIZE,
            {.offset_bytes = 0},
            {.page_id = noc_id, .offset_bytes = noc_offset});
        noc.async_write_barrier();
    }
}
