// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // batch args
    uint32_t MtNt = get_arg_val<uint32_t>(11);  // if 0
    uint32_t batch = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_id_out0 = get_named_compile_time_arg_val("cb_out");

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, out_tensor_addr, single_tile_size_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_id_out0);

    bool one_time_profile = true;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                cb_out.wait_front(out_subblock_tile_count);
                uint32_t out_read_offset = 0;

                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc.async_write(
                            experimental::use<experimental::CircularBuffer::AddrSelector::READ_PTR>(cb_out),
                            s,
                            single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});
                        out_read_offset += single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc.async_write_barrier();
                cb_out.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
}
