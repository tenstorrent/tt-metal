// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t q_shard_ht = get_compile_time_arg_val(0);  // number of Q heads in the group, n
    constexpr uint32_t q_shard_wt = get_compile_time_arg_val(1);  // number of K heads in the group, expecting 1
    constexpr uint32_t k_shard_ht = get_compile_time_arg_val(2);  // number of V heads in the group, expecting 1
    constexpr uint32_t k_shard_wt = get_compile_time_arg_val(3);  // size of a Q head in bytes
    constexpr uint32_t q_num_heads_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t k_num_heads_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(6);  // size of a K head `` ``

    constexpr uint32_t cb_inq = tt::CBIndex::c_0;
    constexpr uint32_t cb_inkv = tt::CBIndex::c_1;

    constexpr uint32_t cb_outq = tt::CBIndex::c_16;
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_outk = tt::CBIndex::c_24;
#else
    constexpr uint32_t cb_outk = tt::CBIndex::c_17;
#endif
    constexpr uint32_t cb_outv = tt::CBIndex::c_18;

    // copy one entire head_dim tile, then go to next sequence tile and do another head_dim.
    // after that, go to next head (head_dim > sequence * batch > head)
    // since Q's heads are shuffled and n Q heads are paired with a KV, need to iterate through multiple Q heads before
    // skipping to next sequence tile

    constexpr uint32_t v_shard_ht = k_shard_ht;

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_inq);
    const DataFormat data_format = get_dataformat(cb_inq);

    /**
     * Iterate over number of heads in each group (n Q, 1 K, 1 V) where total number of groups = total number of KV
     * heads block_ht is the number of tiles along the batch * seq_len dimension shard
     */

    uint64_t q_src_noc_addr = get_noc_addr(get_read_ptr(cb_inq));
    uint32_t q_write_addr = get_write_ptr(cb_outq);

    // re-order q
    constexpr uint32_t q_num_tiles = q_shard_ht * q_shard_wt;
    constexpr uint32_t q_shard_wt_size_bytes = q_shard_wt * single_tile_size_bytes;  // tiles until next sequence
    constexpr uint32_t q_head_size_bytes = tiles_per_head * single_tile_size_bytes;

    cb_reserve_back(cb_outq, q_num_tiles);
    uint32_t head_offset = 0;
    for (uint32_t k = 0; k < q_num_heads_per_core; k++) {  // number of kv heads inside the shard
        uint32_t seq_tile_offset = 0;
        for (uint32_t i = 0; i < q_shard_ht; i++) {  // iterate across seq_len dimension tiles
            uint64_t q_src_noc_addr_head = q_src_noc_addr + seq_tile_offset + head_offset;
            noc_async_read(q_src_noc_addr_head, q_write_addr, q_head_size_bytes);  // read one head worth of tiles
            q_write_addr += q_head_size_bytes;         // go to output address for next Q head
            seq_tile_offset += q_shard_wt_size_bytes;  // go to next tile along seq_len
        }
        head_offset += q_head_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outq, q_num_tiles);

    // re-order k
    uint64_t kv_src_noc_addr = get_noc_addr(get_read_ptr(cb_inkv));
    constexpr uint32_t k_num_tiles = k_shard_ht * k_shard_wt;
    constexpr uint32_t kv_shard_wt_size_bytes = k_shard_wt * single_tile_size_bytes * 2;
    constexpr uint32_t k_head_size_bytes = tiles_per_head * single_tile_size_bytes;
    constexpr uint32_t kv_group_size_bytes = k_head_size_bytes * 2;

    cb_reserve_back(cb_outk, k_num_tiles);
    uint32_t k_write_addr = get_write_ptr(cb_outk);
    head_offset = 0;
    for (uint32_t k = 0; k < k_num_heads_per_core; k++) {  // number of k heads inside the shard
#ifdef TRANSPOSE_K_HEADS
        for (uint32_t k_head_tile_offset = 0; k_head_tile_offset < k_head_size_bytes;
             k_head_tile_offset += single_tile_size_bytes) {  // finish head after sequence length when transposing K
            uint32_t seq_tile_offset = 0;
            for (uint32_t i = 0; i < k_shard_ht; i++) {  // iterate across seq_len dimension tiles
                uint64_t k_src_noc_addr = kv_src_noc_addr + seq_tile_offset + head_offset + k_head_tile_offset;
                noc_async_read(k_src_noc_addr, k_write_addr, single_tile_size_bytes);  // read one head worth of tiles
                k_write_addr += single_tile_size_bytes;     // go to output address for next K head
                seq_tile_offset += kv_shard_wt_size_bytes;  // go to next tile along seq_len
            }
        }
#else
        uint32_t seq_tile_offset = 0;
        for (uint32_t i = 0; i < k_shard_ht; i++) {  // iterate across seq_len dimension tiles
            uint64_t k_src_noc_addr = kv_src_noc_addr + seq_tile_offset + head_offset;
            noc_async_read(k_src_noc_addr, k_write_addr, k_head_size_bytes);  // read one head worth of tiles
            k_write_addr += k_head_size_bytes;                                // go to output address for next K head
            seq_tile_offset += kv_shard_wt_size_bytes;                        // go to next tile along seq_len
        }
#endif
        head_offset += kv_group_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outk, k_num_tiles);

    // re-order v
    constexpr uint32_t v_num_tiles = k_num_tiles;
    constexpr uint32_t v_head_size_bytes = k_head_size_bytes;
    constexpr uint32_t v_num_heads_per_core = k_num_heads_per_core;
    cb_reserve_back(cb_outv, v_num_tiles);
    uint32_t v_write_addr = get_write_ptr(cb_outv);
    head_offset = k_head_size_bytes;                       // v1 is after one k head
    for (uint32_t k = 0; k < v_num_heads_per_core; k++) {  // number of kv heads inside the shard
        uint32_t seq_tile_offset = 0;
        for (uint32_t i = 0; i < v_shard_ht; i++) {  // iterate across seq_len dimension tiles
            uint64_t v_src_noc_addr = kv_src_noc_addr + seq_tile_offset + head_offset;
            noc_async_read(v_src_noc_addr, v_write_addr, v_head_size_bytes);  // read one head worth of tiles
            v_write_addr += v_head_size_bytes;                                // go to output address for next V head
            seq_tile_offset += kv_shard_wt_size_bytes;                        // go to next tile along seq_len
        }
        head_offset += kv_group_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_outv, v_num_tiles);
}
