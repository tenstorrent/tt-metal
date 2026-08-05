// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(4);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(5);
    uint32_t kv_out_tensor_tile_id = get_arg_val<uint32_t>(6);

    // COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t kv_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(4);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(5);
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);
    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    CircularBuffer cb_out0(cb_id_out0);

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    uint32_t l1_read_addr;
    uint32_t out_num_tiles_read;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id;
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        l1_read_addr = cb_out0.get_read_ptr();
        out_num_tiles_read = 0;

        out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
            q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                out_num_tiles_read += block_size;
                cb_out0.wait_front(out_num_tiles_read);

                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sq,
                    single_tile_size_bytes,
                    {},
                    {.page_id = q_out_tensor_current_tile_id});
                l1_read_addr += single_tile_size_bytes;
                q_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            cb_out0.wait_front(out_num_tiles_read);

            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                sk,
                single_tile_size_bytes,
                {},
                {.page_id = out_tensor_current_tile_id});
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            cb_out0.wait_front(out_num_tiles_read);

            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                sv,
                single_tile_size_bytes,
                {},
                {.page_id = out_tensor_current_tile_id});
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tensor_tile_id += q_out_w_tiles;
        } else {
            q_out_tensor_tile_id = q_out_tensor_current_tile_id;
            q_out_h_dim = 0;
        }

        kv_out_tensor_tile_id += kv_num_tiles;

        noc.async_write_barrier();
        cb_out0.pop_front(out_num_tiles_read);
    }
}
