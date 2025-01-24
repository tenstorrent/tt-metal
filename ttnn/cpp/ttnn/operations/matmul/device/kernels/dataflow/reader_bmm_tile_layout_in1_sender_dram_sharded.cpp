// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }

    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
#ifdef FUSE_BIAS
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(2);
#endif
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t num_shard_to_write_back = get_arg_val<uint32_t>(5);
    const uint32_t reshard_tensor_start_offset = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* per_core_N_reshard_bytes = (tt_l1_ptr uint32_t*)(get_arg_addr(7));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(8));
    tt_l1_ptr uint32_t* in0_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(9));

    // COMPILE TIME ARGS
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(1);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    // WRITER
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_tensor_stride_w_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_reshard_tensor_stride_w_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t in3_num_pages = get_compile_time_arg_val(10);
    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(cb_id_in3);
    constexpr DataFormat bias_data_format = get_dataformat(cb_id_in3);
#endif

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_out_reshard = tt::CBIndex::c_6;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    //  READER
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;
    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);

    uint32_t in1_base_addr =
        noc_async_read_tile_dram_sharded_set_state<true>(in1_tensor_addr, in1_page_size, dram_bank_id, vc);

#ifdef ARCH_GRAYSKULL
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc_async_read_tile_dram_sharded_with_state(in1_base_addr, l1_read_addr_in1, l1_write_addr_in1);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
#else
    constexpr uint32_t total_num_blocks_in_buffer = 3;
    constexpr uint32_t total_num_trid = 4;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id_in1, in1_block_num_tiles);
    uint32_t l1_write_addr_in1_offset = 0;
    uint32_t l1_write_addr_in1_start = get_write_ptr(cb_id_in1);
    l1_write_addr_in1 = l1_write_addr_in1_start;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

        for (uint32_t h = 0; h < in1_num_pages; ++h) {
            noc_async_read_tile_dram_sharded_with_state_with_trid(
                in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        if (num_free_blocks_in_buffer == 2) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id_in1, in1_block_num_tiles);
            // wait for next block trid
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
            // reserve for next block
            cb_reserve_back(cb_id_in1, in1_block_num_tiles * 2);
        } else {
            num_free_blocks_in_buffer -= 1;
        }

        if (curr_block_trid == total_num_blocks_in_buffer) {
            l1_write_addr_in1_offset = 0;
            curr_block_trid = 1;
        } else {
            l1_write_addr_in1_offset += in1_block_size_bytes;
            curr_block_trid += 1;
        }
        l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
    }
    // last block to wait
    noc_async_read_barrier_with_trid(block_trid_to_wait);
    cb_push_back(cb_id_in1, in1_block_num_tiles);
#endif

#ifdef FUSE_BIAS
    // Operand 1
    cb_reserve_back(cb_id_in3, in1_block_w);
    uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
    uint32_t l1_read_addr_in3 = 0;

    uint32_t in3_base_addr =
        noc_async_read_tile_dram_sharded_set_state<true>(in3_tensor_addr, in3_page_size, dram_bank_id, vc);

    for (uint32_t h = 0; h < in3_num_pages; ++h) {
        noc_async_read_tile_dram_sharded_with_state(in3_base_addr, l1_read_addr_in3, l1_write_addr_in3);
        l1_read_addr_in3 += in3_page_size;
        l1_write_addr_in3 += in3_page_size;
    }

    // Barrier! make sure the reads are done
    noc_async_read_barrier();
    cb_push_back(cb_id_in3, in1_block_w);
#endif

    // WRITER
    cb_wait_front(cb_id_out, out_block_num_tiles);

#ifndef SKIP_WRITE_BACK
    uint32_t index_offset = 0;
    uint32_t l1_read_addr_out_offset = 0;

    for (uint32_t i = 0; i < num_shard_to_write_back; ++i) {
        uint32_t l1_read_addr_out = get_read_ptr(cb_id_out) + l1_read_addr_out_offset;
        uint32_t l1_write_addr_out_reshard = get_write_ptr(cb_id_out_reshard);

        if (i == 0) {
            l1_write_addr_out_reshard += reshard_tensor_start_offset;
        }

        uint64_t reshard_dest_addr = get_noc_addr(
            in0_mcast_sender_noc_x[index_offset], in0_mcast_sender_noc_y[index_offset], l1_write_addr_out_reshard);

        for (uint32_t h = 0; h < per_core_M; ++h) {
            noc_async_write(l1_read_addr_out, reshard_dest_addr, per_core_N_reshard_bytes[index_offset]);
            l1_read_addr_out += out_tensor_stride_w_bytes;
            reshard_dest_addr += out_reshard_tensor_stride_w_bytes;
        }
        l1_read_addr_out_offset += per_core_N_reshard_bytes[index_offset];

        index_offset += 3;
    }
    noc_async_write_barrier();
#endif
}
