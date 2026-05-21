// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-sparse matmul: in0 (A matrix) mcast sender kernel with K-block skip.
//
// For each K-block, checks a per-K-block active bitmask:
//   - Active  (bit=1): reads tile data from DRAM and multicasts to receivers.
//   - Inactive(bit=0): skips the K-block entirely — no DRAM read, no mcast,
//                      no CB push, no semaphore exchange.
//
// Supports up to 32 K-blocks (uint32_t bitmask). Pass 0xFFFFFFFF for dense path.
//
// Compile-time args:
//   0  in0_tensor_stride_w  -- tile stride in row direction (usually 1)
//   1  in0_tensor_stride_h  -- tile stride between rows (= Kt)
//   2  in0_k_stride         -- tile stride between K-blocks (= in0_block_w)
//   3  in0_m_stride         -- tile stride between M-blocks (= out_block_h * Kt)
//   4  in0_block_w          -- block width in K dim (tiles)
//   5  in0_block_h          -- block height in M dim (tiles)
//   6  in0_block_num_tiles  -- in0_block_w * in0_block_h
//   7  num_k_blocks         -- total K-blocks = Kt / in0_block_w
//   8  num_m_blocks         -- M outer blocks = per_core_M / out_block_h
//   9  sender_semaphore_id
//  10  receiver_semaphore_id
//  11  mcast_num_dests      -- num receiver cores (0 = single-core)
//  12  mcast_num_cores      -- same as mcast_num_dests (NOC uses cores not dests)
//  13+ TensorAccessorArgs for in0
//
// Runtime args:
//   0  in0_tensor_addr
//   1  in0_start_tile       -- start tile id = Kt * per_core_M * output_idx_y
//   2  mcast_noc_start_x
//   3  mcast_noc_start_y
//   4  mcast_noc_end_x
//   5  mcast_noc_end_y
//   6  k_active_bitmask     -- bit k = 1 → K-block k is active

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // ---- Runtime args ----
    uint32_t rt = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t in0_start = get_arg_val<uint32_t>(rt++);
    const uint32_t mc_sx = get_arg_val<uint32_t>(rt++);
    const uint32_t mc_sy = get_arg_val<uint32_t>(rt++);
    const uint32_t mc_ex = get_arg_val<uint32_t>(rt++);
    const uint32_t mc_ey = get_arg_val<uint32_t>(rt++);
    const uint32_t k_mask = get_arg_val<uint32_t>(rt++);

    // ---- Compile-time args ----
    constexpr uint32_t in0_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_k_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in0_m_stride = get_compile_time_arg_val(3);
    constexpr uint32_t in0_bw = get_compile_time_arg_val(4);
    constexpr uint32_t in0_bh = get_compile_time_arg_val(5);
    constexpr uint32_t in0_btn = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t mcast_num_dests = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_num_cores = get_compile_time_arg_val(12);

    constexpr auto in0_ta = TensorAccessorArgs<13>();

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t tile_bytes = get_tile_size(cb_in0);
    constexpr uint32_t block_bytes = in0_btn * tile_bytes;

    const auto s0 = TensorAccessor(in0_ta, in0_addr, tile_bytes);

    uint32_t sender_sem = get_semaphore(sender_sem_id);
    uint32_t receiver_sem = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem);

    // Pre-set our local receiver semaphore = VALID (will be multicast after each block)
    *receiver_sem_ptr = VALID;

    // Pre-compute mcast NOC addresses (used only when mcast_num_dests > 0)
    const uint64_t mcast_receiver_sem_noc = get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, receiver_sem);
    const uint64_t mcast_data_base = get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, 0);

    for (uint32_t bm = 0; bm < num_m_blocks; ++bm) {
        uint32_t m_tile_start = in0_start + bm * in0_m_stride;

        for (uint32_t bk = 0; bk < num_k_blocks; ++bk) {
            if (!((k_mask >> bk) & 1u)) {
                continue;  // Skip inactive K-block entirely
            }

            cb_reserve_back(cb_in0, in0_btn);
            uint32_t l1_addr = get_write_ptr(cb_in0);

            // Wait for all receivers to signal they are ready for this block
            if constexpr (mcast_num_dests > 0) {
                noc_semaphore_wait(sender_sem_ptr, mcast_num_dests);
                noc_semaphore_set(sender_sem_ptr, 0);
            }

            // Read A tiles from DRAM into the CB
            uint32_t k_tile_start = m_tile_start + bk * in0_k_stride;
            uint32_t dst = l1_addr;
            for (uint32_t h = 0; h < in0_bh; ++h) {
                uint32_t row_start = k_tile_start + h * in0_stride_h;
                for (uint32_t w = 0; w < in0_bw; ++w) {
                    noc_async_read_tile(row_start + w * in0_stride_w, s0, dst);
                    dst += tile_bytes;
                }
            }
            noc_async_read_barrier();

            // Multicast this block to all receiver cores
            if constexpr (mcast_num_dests > 0) {
                uint64_t mcast_dest = mcast_data_base | l1_addr;
                noc_async_write_multicast(l1_addr, mcast_dest, block_bytes, mcast_num_cores, true);
                // Signal receivers that this block is ready
                noc_semaphore_set_multicast(receiver_sem, mcast_receiver_sem_noc, mcast_num_cores);
            }

            cb_push_back(cb_in0, in0_btn);
        }
    }
    noc_async_write_barrier();
}
