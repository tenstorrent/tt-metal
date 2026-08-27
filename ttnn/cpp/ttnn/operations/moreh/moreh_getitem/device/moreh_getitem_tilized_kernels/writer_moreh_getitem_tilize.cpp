// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
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

    const auto s0 = TensorAccessor(tensor::s0);

    Noc noc;
    // The output stick is drained straight out of the buffer the reader staged it in.
    DataflowBuffer dfb_out_obj(dfb::out);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out_obj.wait_front(1);

        uint32_t stick_idx = i;

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

        noc.async_write(
            dfb_out_obj, s0, stick_size, {.offset_bytes = 0}, {.page_id = noc_id, .offset_bytes = noc_offset});
        noc.async_write_barrier();
        dfb_out_obj.pop_front(1);
    }
}
