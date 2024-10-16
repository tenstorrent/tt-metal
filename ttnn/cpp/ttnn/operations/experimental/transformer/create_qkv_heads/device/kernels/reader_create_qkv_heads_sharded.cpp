// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t q_heads_per_group = get_compile_time_arg_val(0);  // number of Q heads in the group, n
    constexpr uint32_t k_heads_per_group = get_compile_time_arg_val(1);  // number of K heads in the group, expecting 1
    constexpr uint32_t v_heads_per_group = get_compile_time_arg_val(2);  // number of V heads in the group, expecting 1

    constexpr uint32_t q_head_size_bytes = get_compile_time_arg_val(3);  // size of a Q head in bytes
    constexpr uint32_t k_head_size_bytes = get_compile_time_arg_val(4);  // size of a K head `` ``
    constexpr uint32_t v_head_size_bytes = get_compile_time_arg_val(5);  // size of a V head `` ``

    constexpr uint32_t group_t_size_bytes = get_compile_time_arg_val(6);  // size of each group
    constexpr uint32_t block_ht = get_compile_time_arg_val(7);  // number of tiles along the seq_len*batch dimension
    constexpr uint32_t groups_per_block = get_compile_time_arg_val(8);  // number of groups per shard

    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(9);   // total number of Q pages, used for CB reservation
    constexpr uint32_t k_num_tiles = get_compile_time_arg_val(10);  // total number of K pages
    constexpr uint32_t v_num_tiles = get_compile_time_arg_val(11);  // total number of V pages

    constexpr uint32_t q_size_per_group_t_bytes =
        get_compile_time_arg_val(12);  // total size of all n Q heads in a group
    constexpr uint32_t k_size_per_group_t_bytes =
        get_compile_time_arg_val(13);  // total size of all K heads (expecting 1) in a group
    constexpr uint32_t v_size_per_group_t_bytes =
        get_compile_time_arg_val(14);  // total size of all V heads (expecting 1) in a group

    constexpr uint32_t cb_in0 = tt::CB::c_in0;

    constexpr uint32_t cb_outq = tt::CB::c_out0;
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_outk = tt::CB::c_intermed0;
#else
    constexpr uint32_t cb_outk = tt::CB::c_out1;
#endif
    constexpr uint32_t cb_outv = tt::CB::c_out2;

    // copy one entire head_dim tile, then go to next sequence tile and do another head_dim.
    // after that, go to next head (head_dim > sequence * batch > head)
    // since Q's heads are shuffled and n Q heads are paired with a KV, need to iterate through multiple Q heads before
    // skipping to next sequence tile

    constexpr uint32_t block_wt_size_bytes = groups_per_block * (group_t_size_bytes);
    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_in0);
    const DataFormat data_format = get_dataformat(cb_in0);

    /**
     * Iterate over number of heads in each group (n Q, 1 K, 1 V) where total number of groups = total number of KV
     * heads block_ht is the number of tiles along the batch * seq_len dimension shard
     */

    uint64_t src_noc_addr = get_noc_addr(get_read_ptr(cb_in0));
    // re-order q
    cb_reserve_back(cb_outq, q_num_tiles);

    uint32_t q_write_addr = get_write_ptr(cb_outq);
    uint32_t src_noc_addr_offset_outer = 0;

    uint32_t group_addr_offset = 0;
    for (uint32_t k = 0; k < groups_per_block; k++) {  // number of kv heads inside the shard
        uint32_t head_in_group_offset = 0;
        for (uint32_t j = 0; j < q_heads_per_group;
             j++) {  // go to next Q heads in the group (0 to n-1 for the nQ per KV group)
            uint32_t seq_tile_offset = 0;
            for (uint32_t i = 0; i < block_ht; i++) {  // iterate across seq_len dimension tiles
                uint64_t q_src_noc_addr = src_noc_addr + seq_tile_offset + head_in_group_offset + group_addr_offset;
                noc_async_read(q_src_noc_addr, q_write_addr, q_head_size_bytes);  // read one head worth of tiles
                q_write_addr += q_head_size_bytes;       // go to output address for next Q head
                seq_tile_offset += block_wt_size_bytes;  // go to next tile along seq_len
            }
            head_in_group_offset += q_head_size_bytes;
        }
        group_addr_offset += group_t_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outq, q_num_tiles);

    // re-order k

    cb_reserve_back(cb_outk, k_num_tiles);
    uint32_t k_write_addr = get_write_ptr(cb_outk);
    group_addr_offset = q_size_per_group_t_bytes;
    for (uint32_t k = 0; k < groups_per_block; k++) {  // number of kv heads inside the shard
        uint32_t head_in_group_offset = 0;
        for (uint32_t j = 0; j < k_heads_per_group; j++) {  // go to next K heads in the group (expecting only 1 for K)
#ifdef TRANSPOSE_K_HEADS
            for (uint32_t k_head_tile_offset = 0; k_head_tile_offset < k_head_size_bytes;
                 k_head_tile_offset +=
                 single_tile_size_bytes) {  // finish head after sequence length when transposing K
                uint32_t seq_tile_offset = 0;
                for (uint32_t i = 0; i < block_ht; i++) {  // iterate across seq_len dimension tiles
                    uint64_t k_src_noc_addr =
                        src_noc_addr + seq_tile_offset + head_in_group_offset + group_addr_offset + k_head_tile_offset;
                    noc_async_read(k_src_noc_addr,
                                   k_write_addr,
                                   single_tile_size_bytes);  // read only one tile since we're transposing
                    k_write_addr += single_tile_size_bytes;  // output address of next K head
                    seq_tile_offset += block_wt_size_bytes;  // go to next tile in seq_len
                }
            }
#else
            uint32_t seq_tile_offset = 0;
            for (uint32_t i = 0; i < block_ht; i++) {  // iterate across seq_len dimension tiles
                uint64_t k_src_noc_addr = src_noc_addr + seq_tile_offset + head_in_group_offset + group_addr_offset;
                noc_async_read(k_src_noc_addr, k_write_addr, k_head_size_bytes);  // read one head worth of tiles
                k_write_addr += k_head_size_bytes;                                // output address of next K head
                seq_tile_offset += block_wt_size_bytes;                           // go to next tile in seq_len
            }
#endif
            head_in_group_offset += k_head_size_bytes;
        }
        group_addr_offset += group_t_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outk, k_num_tiles);

    // re-order v
    cb_reserve_back(cb_outv, v_num_tiles);
    uint32_t v_write_addr = get_write_ptr(cb_outv);
    group_addr_offset = q_size_per_group_t_bytes + k_size_per_group_t_bytes;
    for (uint32_t k = 0; k < groups_per_block; k++) {  // number of kv heads inide the hard
        uint32_t head_in_group_offset = 0;
        for (uint32_t j = 0; j < v_heads_per_group; j++) {  // go to next V heads in the group (expecting only 1 for V)
            uint32_t seq_tile_offset = 0;
            for (uint32_t i = 0; i < block_ht; i++) {  // iterate across seq_len dimension tiles
                uint64_t k_src_noc_addr = src_noc_addr + seq_tile_offset + head_in_group_offset + group_addr_offset;
                noc_async_read(k_src_noc_addr, v_write_addr, v_head_size_bytes);  // read one head worth of tiles
                v_write_addr += v_head_size_bytes;                                // output address of next V head
                seq_tile_offset += block_wt_size_bytes;                           // go to next tile in seq_len
            }
            head_in_group_offset += v_head_size_bytes;
        }
        group_addr_offset += group_t_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outv, v_num_tiles);
}
