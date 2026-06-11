// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention reader (NCRISC / NoC0).
//
// Per work unit (b, h, q_chunk):
//   0. GQA/MQA head mapping: Q head h reads KV head h / (H / H_kv); MHA is the H_kv == H case
//   1. push the Q chunk once (cur_cq * Dt tiles, retained by compute across the KV loop)
//   2. stream Nkv KV blocks: K-transposed tiles (Dt x cur_ckv, tile-order (d, n)),
//      V tiles (cur_ckv x Dt, row-major), mask tiles (cur_cq x cur_ckv, when HAS_MASK)
// Scaler tiles (MAX/ROW row0 fill for the running-max reduce, SUM/ROW col0 fill
// for the rowsum-via-matmul reduce) are prepared once via the pool-type-aware helper.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t c_q = get_compile_time_arg_val(4);
    constexpr uint32_t c_kv = get_compile_time_arg_val(5);
    constexpr uint32_t Nq = get_compile_time_arg_val(6);
    constexpr uint32_t Nkv = get_compile_time_arg_val(7);
    constexpr uint32_t c_q_last = get_compile_time_arg_val(8);
    constexpr uint32_t c_kv_last = get_compile_time_arg_val(9);
    constexpr bool HAS_MASK = get_compile_time_arg_val(10) != 0;
    constexpr bool MASK_PER_HEAD = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t H_kv = get_compile_time_arg_val(12);  // KV heads (== H for MHA, < H for GQA, 1 for MQA)
    constexpr uint32_t kv_rem = get_compile_time_arg_val(13);  // valid cols in last KV tile (0 = aligned)
    // KV row-mcast group size (0 = off): g consecutive cores share one head's
    // KV stream; the leader reads DRAM once per block and mcasts to followers.
    constexpr uint32_t GROUP = get_compile_time_arg_val(14);
    // Legacy V loop is skipped when V comes from the mcast path (reader,
    // GROUP > 0) or the writer (ones-col without mcast).
    constexpr bool SKIP_LEGACY_V = get_compile_time_arg_val(15) != 0;
    // Mcast V carries a leading all-ones tile per row (rowsum-in-matmul).
    constexpr bool ONES_COL = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t V_W = ONES_COL ? Dt + 1 : Dt;

    constexpr uint32_t HEAD_RATIO = H / H_kv;  // validate() enforces H % H_kv == 0

    constexpr auto q_args = TensorAccessorArgs<17>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);
    // Mcast extras (GROUP > 0): leader flag + group span + leader coords
    // (leader may sit mid-row — diagonal placement spreads DRAM traffic).
    uint32_t is_leader = 0, first_x = 0, first_y = 0, last_x = 0, last_y = 0, lead_x = 0, lead_y = 0;
    if constexpr (GROUP > 0) {
        is_leader = get_arg_val<uint32_t>(6);
        first_x = get_arg_val<uint32_t>(7);
        first_y = get_arg_val<uint32_t>(8);
        last_x = get_arg_val<uint32_t>(9);
        last_y = get_arg_val<uint32_t>(10);
        lead_x = get_arg_val<uint32_t>(11);
        lead_y = get_arg_val<uint32_t>(12);
    }

    if (num_units == 0) {
        return;
    }

    constexpr uint32_t cb_q_tiles = 0;
    constexpr uint32_t cb_kt_tiles = 1;
    constexpr uint32_t cb_v_tiles = 2;
    constexpr uint32_t cb_mask_tiles = 3;
    constexpr uint32_t cb_pad_mask = 4;
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;

    const uint32_t tile_bytes = get_tile_size(cb_q_tiles);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    [[maybe_unused]] const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);

    // Scalers once per program (pool-type-aware fill: MAX/ROW row0, SUM/ROW col0).
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // Pad-mask row for non-tile-aligned S_kv (Refinement 4): c_kv_last bf16
    // tiles, all zero except the LAST tile's columns >= kv_rem, which carry
    // -1e9 (not -inf: avoids inf-inf NaN, matching the kernel's m_prev init).
    // Compute adds this row to the scaled scores on the last KV block — before
    // the running-max update — so zero-padded K rows never enter rowmax/rowsum.
    // Pushed once, never popped (compute waits HeldBulk every last block).
    if constexpr (kv_rem != 0) {
        constexpr uint16_t NEG_BIG_BF16 = 0xCE6E;  // bf16(-1e9)
        constexpr uint32_t TILE_VALS = 32 * 32;
        cb_reserve_back(cb_pad_mask, c_kv_last);
        volatile tt_l1_ptr uint16_t* pad = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_pad_mask));
        for (uint32_t i = 0; i < c_kv_last * TILE_VALS; ++i) {
            pad[i] = 0;
        }
        // Faces are 16x16; index = face*256 + (r%16)*16 + (c%16), face = (r/16)*2 + (c/16).
        volatile tt_l1_ptr uint16_t* last = pad + (c_kv_last - 1) * TILE_VALS;
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = kv_rem; c < 32; ++c) {
                last[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)] = NEG_BIG_BF16;
            }
        }
        cb_push_back(cb_pad_mask, c_kv_last);
    }

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t bh = unit / Nq;  // flattened b*H + h
        const uint32_t qc = unit % Nq;  // q-chunk index
        const uint32_t cur_cq = (qc == Nq - 1) ? c_q_last : c_q;
        const uint32_t q_row0 = qc * c_q;

        // GQA/MQA head mapping: Q head h reads KV head h / (H / H_kv).
        // MHA (H_kv == H) degenerates to kv_bh == bh.
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t kv_bh = b * H_kv + h / HEAD_RATIO;
        const uint32_t kv_head_base = kv_bh * Skv_t * Dt;  // K/V tile base for this head
        const uint32_t q_head_base = bh * Sq_t * Dt;

        // 1. Q chunk: cur_cq * Dt tiles, row-major (r, d).
        {
            cb_reserve_back(cb_q_tiles, cur_cq * Dt);
            uint32_t l1_addr = get_write_ptr(cb_q_tiles);
            for (uint32_t r = 0; r < cur_cq; ++r) {
                const uint32_t row_base = q_head_base + (q_row0 + r) * Dt;
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc_async_read_tile(row_base + d, q_accessor, l1_addr);
                    l1_addr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_q_tiles, cur_cq * Dt);
        }

        // 2. KV blocks.
        for (uint32_t kb = 0; kb < Nkv; ++kb) {
            const uint32_t cur_ckv = (kb == Nkv - 1) ? c_kv_last : c_kv;
            const uint32_t n0 = kb * c_kv;

            if constexpr (GROUP > 0) {
                // All cores in the group share this head's KV stream and run
                // unit-locked, so reserve lands at the same CB write pointer
                // on every core. Leader reads DRAM, followers receive mcast.
                // K^T and V share one handshake; both ride NoC0.
                const uint32_t block_tiles = Dt * cur_ckv;
                cb_reserve_back(cb_kt_tiles, block_tiles);
                const uint32_t kt_addr = get_write_ptr(cb_kt_tiles);
                volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
                volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
                if (is_leader) {
                    uint32_t l1_addr = kt_addr;
                    for (uint32_t d = 0; d < Dt; ++d) {
                        for (uint32_t n = 0; n < cur_ckv; ++n) {
                            noc_async_read_tile(kv_head_base + (n0 + n) * Dt + d, k_accessor, l1_addr);
                            l1_addr += tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                    noc_semaphore_wait(ready, GROUP - 1);
                    noc_semaphore_set(ready, 0);
                    const uint64_t kt_mcast = get_noc_multicast_addr(first_x, first_y, last_x, last_y, kt_addr);
                    noc_async_write_multicast(kt_addr, kt_mcast, block_tiles * tile_bytes, GROUP - 1);
                    const uint64_t landed_mcast =
                        get_noc_multicast_addr(first_x, first_y, last_x, last_y, get_semaphore(1));
                    *landed = 1;
                    noc_semaphore_set_multicast(get_semaphore(1), landed_mcast, GROUP - 1);
                    noc_async_write_barrier();
                } else {
                    noc_semaphore_inc(get_noc_addr(lead_x, lead_y, get_semaphore(0)), 1);
                    noc_semaphore_wait(landed, 1);
                    noc_semaphore_set(landed, 0);
                }
                cb_push_back(cb_kt_tiles, block_tiles);
            } else {
                // K^T tiles: tile-order (d, n) -> K[bh, n0+n, d]. Intra-tile transpose
                // is done by the matmul's transpose=true — both halves are required.
                cb_reserve_back(cb_kt_tiles, Dt * cur_ckv);
                uint32_t l1_addr = get_write_ptr(cb_kt_tiles);
                for (uint32_t d = 0; d < Dt; ++d) {
                    for (uint32_t n = 0; n < cur_ckv; ++n) {
                        noc_async_read_tile(kv_head_base + (n0 + n) * Dt + d, k_accessor, l1_addr);
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_kt_tiles, Dt * cur_ckv);

                if constexpr (!SKIP_LEGACY_V) {
                    // V tiles: row-major (n, d).
                    cb_reserve_back(cb_v_tiles, cur_ckv * Dt);
                    l1_addr = get_write_ptr(cb_v_tiles);
                    for (uint32_t n = 0; n < cur_ckv; ++n) {
                        for (uint32_t d = 0; d < Dt; ++d) {
                            noc_async_read_tile(kv_head_base + (n0 + n) * Dt + d, v_accessor, l1_addr);
                            l1_addr += tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_v_tiles, cur_ckv * Dt);
                }
            }

            // Mask tiles: row-major (r, n) over [q_row0, q_row0+cur_cq) x [n0, n0+cur_ckv).
            if constexpr (HAS_MASK) {
                const uint32_t mask_head = MASK_PER_HEAD ? bh : b;  // mask indexes Q heads, not KV heads
                const uint32_t mask_base = mask_head * Sq_t * Skv_t;
                cb_reserve_back(cb_mask_tiles, cur_cq * cur_ckv);
                uint32_t l1_addr = get_write_ptr(cb_mask_tiles);
                for (uint32_t r = 0; r < cur_cq; ++r) {
                    const uint32_t row_base = mask_base + (q_row0 + r) * Skv_t + n0;
                    for (uint32_t n = 0; n < cur_ckv; ++n) {
                        noc_async_read_tile(row_base + n, mask_accessor, l1_addr);
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_mask_tiles, cur_cq * cur_ckv);
            }
        }
    }
}
