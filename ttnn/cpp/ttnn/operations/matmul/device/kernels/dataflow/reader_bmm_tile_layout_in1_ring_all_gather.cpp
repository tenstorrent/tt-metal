// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "remote_circular_buffer_api.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

template <bool DRAM, uint32_t tile_hw>
void read_block_from_dram(
    uint32_t cb_id,
    InterleavedAddrGenFast<DRAM, tile_hw> s1,
    uint32_t tensor_width_in_tiles,
    uint32_t block_w_idx,
    uint32_t block_h_idx,
    uint32_t block_w_t,
    uint32_t block_h_t,
    uint32_t tile_size_bytes) {
    uint32_t l1_write_addr = get_write_ptr(cb_id);

    // Horizontal idx + vertical idx * width = row major index
    uint32_t block_tile_id = block_w_idx * block_w_t + (block_h_idx * block_h_t) * tensor_width_in_tiles;
    for (uint32_t h = 0; h < block_h_t; ++h) {
        uint32_t tile_id = block_tile_id + h * tensor_width_in_tiles;
        for (uint32_t w = 0; w < block_w_t; ++w) {
            noc_async_read_tile(tile_id + w, s1, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
    }
    noc_async_read_barrier();
}

void kernel_main() {
    // Compile time args
    constexpr const bool in1_is_dram_interleaved = get_compile_time_arg_val(0);
    constexpr const bool in1_is_dram_sharded = get_compile_time_arg_val(1);
    constexpr uint32_t in1_block_height_in_tiles = get_compile_time_arg_val(2);  // Padded block shape
    constexpr uint32_t in1_block_width_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_tensor_width_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t batch = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t in1_block_page_size_last = get_compile_time_arg_val(8);
    constexpr uint32_t in1_block_width_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t in1_shard_width_in_dram = get_compile_time_arg_val(10);

    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE || core_type == (uint32_t)CORE_TYPE::HOP_CORE) {
        return;
    }
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_read_offset = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t sync_cb = tt::CBIndex::c_3;
    constexpr uint32_t sync_cb2 = tt::CBIndex::c_4;
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;

    const uint32_t in1_block_num_tiles = in1_block_height_in_tiles * in1_block_width_in_tiles;

    // Address setup
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<in1_is_dram_interleaved, in1_tile_hw> s1 = {
        .bank_base_address = in1_tensor_addr, .page_size = in1_single_tile_size_bytes, .data_format = in1_data_format};

    // const uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    const uint32_t in1_shard_width_offset_bytes = in1_shard_width_in_dram * in1_single_tile_size_bytes;
    const uint32_t in1_dram_shard_block_size_bytes = in1_shard_width_offset_bytes * in1_block_height_in_tiles;
    const uint32_t dram_read_offset_bytes = dram_read_offset * in1_block_width_in_tiles * in1_single_tile_size_bytes;
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;
    uint32_t in1_base_addr = 0;
    // if constexpr(in1_is_dram_sharded) {
    //     in1_base_addr = noc_async_read_tile_dram_sharded_set_state<true>(in1_tensor_addr, in1_page_size,
    //     dram_bank_id, vc);
    // }
    DPRINT << "dram_read_offset_bytes " << dram_read_offset_bytes << ENDL();
    DPRINT << "in1_shard_width_offset_bytes " << in1_shard_width_offset_bytes << ENDL();
    DPRINT << "dram_bank_id " << dram_bank_id << ENDL();
    DPRINT << "in1_block_num_tiles " << in1_block_num_tiles << ENDL();

    // DPRINT << "in1_shard_width_in_dram " << in1_shard_width_in_dram << " in1_block_page_size_last " <<
    // in1_block_page_size_last << " in1_block_page_size " << in1_block_page_size << ENDL();

    for (uint32_t b = 0; b < batch; ++b) {
        cb_reserve_back(sync_cb2, 1);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif

        cb_push_back(sync_cb2, 1);

        if constexpr (in1_is_dram_interleaved) {
            for (uint32_t block = 0; block < num_blocks; ++block) {
                uint32_t block_idx = (ring_idx + block) % num_blocks;

                cb_reserve_back(cb_id_in1, in1_block_num_tiles);
                read_block_from_dram(
                    cb_id_in1,
                    s1,
                    in1_tensor_width_in_tiles,
                    ring_idx,
                    block_idx,
                    in1_block_width_in_tiles,
                    in1_block_height_in_tiles,
                    in1_single_tile_size_bytes);
                cb_push_back(cb_id_in1, in1_block_num_tiles);
            }
        } else if (in1_is_dram_sharded) {
            for (uint32_t block = 0; block < num_blocks; ++block) {
                uint32_t block_idx = (ring_idx + block) % num_blocks;
                l1_read_addr_in1 = block_idx * in1_dram_shard_block_size_bytes + dram_read_offset_bytes;
                // Operand 1
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);

                for (uint32_t h = 0; h < in1_block_height_in_tiles; ++h) {
                    uint32_t curr_l1_read_addr_in1 = l1_read_addr_in1;
                    for (uint32_t w = 0; w < in1_block_width_num_pages; ++w) {
                        uint32_t curr_page_size =
                            w == in1_block_width_num_pages - 1 ? in1_block_page_size_last : in1_block_page_size;
                        in1_base_addr = noc_async_read_tile_dram_sharded_set_state<true>(
                            in1_tensor_addr, curr_page_size, dram_bank_id, vc);
                        noc_async_read_tile_dram_sharded_with_state(
                            in1_base_addr, curr_l1_read_addr_in1, l1_write_addr_in1);
                        curr_l1_read_addr_in1 += curr_page_size;
                        l1_write_addr_in1 += curr_page_size;

                        // DPRINT << "curr_l1_read_addr_in1 " << curr_l1_read_addr_in1 << " l1_write_addr_in1 " <<
                        // l1_write_addr_in1 << ENDL();
                    }
                    l1_read_addr_in1 += in1_shard_width_offset_bytes;
                }

                noc_async_read_barrier();
                cb_push_back(cb_id_in1, in1_block_num_tiles);
            }

            // constexpr uint32_t total_num_blocks_in_buffer = 2;
            // uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
            // uint32_t curr_block_trid = 1;
            // uint32_t block_trid_to_wait = 1;

            // cb_reserve_back(cb_id_in1, in1_block_num_tiles * total_num_blocks_in_buffer);

            // uint32_t l1_write_addr_in1_offset = 0;
            // uint32_t l1_write_addr_in1_start = get_write_ptr(cb_id_in1);
            // l1_write_addr_in1 = l1_write_addr_in1_start;
            // for (uint32_t block = 0; block < num_blocks; ++block) {
            //     uint32_t block_idx = (ring_idx + block) % num_blocks;
            //     l1_read_addr_in1 = block_idx * in1_block_size_bytes + dram_read_offset_bytes;

            //     DPRINT << "block_idx " << block_idx << " l1_read_addr_in1 "  << l1_read_addr_in1<< ENDL();

            //     noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

            //     for (uint32_t h = 0; h < in1_block_height_in_tiles; ++h) {
            //         uint32_t curr_l1_read_addr_in1 = l1_read_addr_in1;
            //         for (uint32_t w = 0; w < in1_block_width_num_pages; ++w) {
            //             uint32_t curr_page_size = w == in1_block_width_num_pages - 1? in1_block_page_size_last :
            //             in1_block_page_size; in1_base_addr =
            //             noc_async_read_tile_dram_sharded_set_state<true>(in1_tensor_addr, curr_page_size,
            //             dram_bank_id, vc); noc_async_read_tile_dram_sharded_with_state_with_trid(
            //                 in1_base_addr, curr_l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
            //             curr_l1_read_addr_in1 += curr_page_size;
            //             l1_write_addr_in1 += curr_page_size;

            //             // DPRINT << "block " << block << " h " << h << " w " << w << " curr_page_size " <<
            //             curr_page_size << ENDL();
            //         }
            //         l1_read_addr_in1 += in1_shard_width_offset_bytes;
            //     }

            //     // noc_async_read_barrier();

            //     if (num_free_blocks_in_buffer == 1) {
            //         noc_async_read_barrier_with_trid(block_trid_to_wait);
            //         cb_push_back(cb_id_in1, in1_block_num_tiles);
            //         // wait for next block trid
            //         block_trid_to_wait = block_trid_to_wait == total_num_blocks_in_buffer ? 1 : (block_trid_to_wait +
            //         1);
            //         // reserve for next block
            //         cb_reserve_back(cb_id_in1, in1_block_num_tiles * (total_num_blocks_in_buffer - 1));
            //     } else {
            //         num_free_blocks_in_buffer -= 1;
            //     }

            //     if (curr_block_trid == total_num_blocks_in_buffer) {
            //         l1_write_addr_in1_offset = 0;
            //         curr_block_trid = 1;
            //     } else {
            //         l1_write_addr_in1_offset += in1_block_size_bytes;
            //         curr_block_trid += 1;
            //     }
            //     l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
            // }
            // // last block to wait
            // noc_async_read_barrier_with_trid(block_trid_to_wait);
            // cb_push_back(cb_id_in1, in1_block_num_tiles);
        }

#ifdef ENABLE_GLOBAL_CB
        cb_wait_front(sync_cb, 1);
        experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
        cb_pop_front(sync_cb, 1);
#endif
    }

#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
#endif
}
