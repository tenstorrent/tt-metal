// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "dataflow_api.h"

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(4);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(5);
    uint32_t kv_out_tensor_tile_id = get_arg_val<uint32_t>(6);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t kv_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(5);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(6);

    constexpr uint32_t cb_id_out0 = 0;  // same as cb_id_in0
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
    const InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<out_is_dram_bool> sv = {
        .bank_base_address = v_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    uint32_t l1_read_addr;
    uint32_t out_num_tiles_read;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id;
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        l1_read_addr = get_read_ptr(cb_id_out0);
        out_num_tiles_read = 0;

        // q + create q head --> outputs: [B, num_heads, S, head_dim]
        out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
            q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                out_num_tiles_read += block_size;
                cb_wait_front(cb_id_out0, out_num_tiles_read);

                noc_async_write_tile(q_out_tensor_current_tile_id, sq, l1_read_addr);
                l1_read_addr += single_tile_size_bytes;
                q_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        // k
        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            cb_wait_front(cb_id_out0, out_num_tiles_read);

            noc_async_write_tile(out_tensor_current_tile_id, sk, l1_read_addr);
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        // v
        out_tensor_current_tile_id = kv_out_tensor_tile_id;
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            out_num_tiles_read += block_size;
            cb_wait_front(cb_id_out0, out_num_tiles_read);

            noc_async_write_tile(out_tensor_current_tile_id, sv, l1_read_addr);
            l1_read_addr += single_tile_size_bytes;
            out_tensor_current_tile_id++;
        }

        // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tensor_tile_id += q_out_w_tiles;
        } else {
            q_out_tensor_tile_id = q_out_tensor_current_tile_id;
            q_out_h_dim = 0;
        }

        kv_out_tensor_tile_id += kv_num_tiles;

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, out_num_tiles_read);
    }
}
