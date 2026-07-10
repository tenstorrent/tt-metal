// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for Flash-Attention scaled_dot_product_attention.
//
// Per Q-block (b, h_q, q_chunk): read the Q chunk once into cb_q, then for every
// KV-block read K, V (and the custom mask block) into cb_k / cb_v / cb_mask.
//
// Tile-grid ordering is dictated by the matmul_block helper's in0/in1 indexing:
//   cb_q : natural  [q_cnt rows x Dt cols]  -> for st(q): for dt
//   cb_k : QKᵀ uses transpose=true, in1 indexed [Dt(K) x kv(N)] -> for dt: for st(kv)
//   cb_v : PV uses transpose=false, in1 indexed [kv(K) x Dt(N)] -> for st(kv): for dt
//   cb_mask : [q_cnt x kv_cnt] row-major -> for st(q): for kv
//
// GQA/MQA is reader-only: h_kv = h_q / gqa_factor.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t mask_addr = get_arg_val<uint32_t>(3);
    uint32_t start_qb = get_arg_val<uint32_t>(4);
    uint32_t num_qb = get_arg_val<uint32_t>(5);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(3);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(4);
    constexpr uint32_t Dt = get_compile_time_arg_val(5);
    constexpr uint32_t q_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t kv_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t q_blocks_per_bh = get_compile_time_arg_val(8);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t gqa_factor = get_compile_time_arg_val(10);
    constexpr uint32_t mask_mode = get_compile_time_arg_val(11);
    constexpr uint32_t mask_H = get_compile_time_arg_val(12);
    constexpr uint32_t skv_non_aligned = get_compile_time_arg_val(13);
    constexpr uint32_t skv_last_valid = get_compile_time_arg_val(14);
    constexpr bool has_mask = (mask_mode == 1);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_mask = 3;
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;

    // Reduce scalers (MAX / SUM, both REDUCE_ROW). Prepared once; reduce never pops them.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    if constexpr (skv_non_aligned) {
        // Non-aligned S_kv: emit the [full, partial] SUM scaler tile pair. The
        // partial tile zeroes the (32 - skv_last_valid) padded key columns so
        // they are excluded from the softmax row-sum (compute applies it on the
        // last kv-tile of the last KV block via ReducePartialScaler::last_tile_at(1)).
        //
        // NOTE: the convenience wrapper calculate_and_prepare_partial_reduce_scalers
        // is unusable as shipped — it forwards a 4th `compute_uses_reduce_tile`
        // template arg to prepare_reduce_scaler, which only declares 3 (reduce_helpers_
        // dataflow.inl:270), so it fails to compile for every instantiation. We build
        // the same tile pair directly from the working 3-arg primitive: each call
        // reserves+fills+pushes exactly one tile (full fill when valid==32, partial
        // fill otherwise), giving tile 0 = full, tile 1 = partial in CB order.
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                1.0f, /*valid_reduce_dim_elements_in_tile=*/32);
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                1.0f, /*valid_reduce_dim_elements_in_tile=*/skv_last_valid);
    } else {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_scaler_sum,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
    }

    // TensorAccessors (all declared unconditionally; mask is a no-arg placeholder when absent).
    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    // Mask accessor is loop-invariant; build once (only meaningful when has_mask).
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, tile_bytes);

    for (uint32_t qi = 0; qi < num_qb; ++qi) {
        const uint32_t qb = start_qb + qi;
        const uint32_t bh = qb / q_blocks_per_bh;
        const uint32_t qci = qb % q_blocks_per_bh;
        const uint32_t b = bh / H_q;
        const uint32_t h_q = bh % H_q;
        const uint32_t h_kv = h_q / gqa_factor;

        const uint32_t q_row0 = qci * q_chunk_t;
        uint32_t q_cnt = q_chunk_t;
        if (q_row0 + q_cnt > Sq_t) {
            q_cnt = Sq_t - q_row0;
        }

        // --- Q chunk: for st(q): for dt ---
        cb_reserve_back(cb_q, q_cnt * Dt);
        {
            uint32_t w = get_write_ptr(cb_q);
            uint32_t idx = 0;
            const uint32_t q_head_base = (b * H_q + h_q) * Sq_t;
            for (uint32_t st = q_row0; st < q_row0 + q_cnt; ++st) {
                for (uint32_t dt = 0; dt < Dt; ++dt) {
                    uint32_t tid = (q_head_base + st) * Dt + dt;
                    noc_async_read_tile(tid, q_acc, w + idx * tile_bytes);
                    ++idx;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_q, q_cnt * Dt);

        const uint32_t k_head_base = (b * H_kv + h_kv) * Skv_t;
        const uint32_t mask_head = (mask_H == 1) ? 0 : h_q;
        const uint32_t mask_head_base = (b * mask_H + mask_head) * Sq_t;

        for (uint32_t j = 0; j < num_kv_blocks; ++j) {
            const uint32_t kv_row0 = j * kv_chunk_t;
            uint32_t kv_cnt = kv_chunk_t;
            if (kv_row0 + kv_cnt > Skv_t) {
                kv_cnt = Skv_t - kv_row0;
            }

            // --- K block: for dt: for st(kv)  (transposed grid for QKᵀ in1) ---
            cb_reserve_back(cb_k, kv_cnt * Dt);
            {
                uint32_t w = get_write_ptr(cb_k);
                uint32_t idx = 0;
                for (uint32_t dt = 0; dt < Dt; ++dt) {
                    for (uint32_t st = kv_row0; st < kv_row0 + kv_cnt; ++st) {
                        uint32_t tid = (k_head_base + st) * Dt + dt;
                        noc_async_read_tile(tid, k_acc, w + idx * tile_bytes);
                        ++idx;
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_k, kv_cnt * Dt);

            // --- V block: for st(kv): for dt  (natural for PV in1) ---
            cb_reserve_back(cb_v, kv_cnt * Dt);
            {
                uint32_t w = get_write_ptr(cb_v);
                uint32_t idx = 0;
                for (uint32_t st = kv_row0; st < kv_row0 + kv_cnt; ++st) {
                    for (uint32_t dt = 0; dt < Dt; ++dt) {
                        uint32_t tid = (k_head_base + st) * Dt + dt;
                        noc_async_read_tile(tid, v_acc, w + idx * tile_bytes);
                        ++idx;
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_v, kv_cnt * Dt);

            // --- Mask block (custom only): for st(q): for kv ---
            if constexpr (has_mask) {
                cb_reserve_back(cb_mask, q_cnt * kv_cnt);
                uint32_t w = get_write_ptr(cb_mask);
                uint32_t idx = 0;
                for (uint32_t st = q_row0; st < q_row0 + q_cnt; ++st) {
                    for (uint32_t kv = kv_row0; kv < kv_row0 + kv_cnt; ++kv) {
                        uint32_t tid = (mask_head_base + st) * Skv_t + kv;
                        noc_async_read_tile(tid, mask_acc, w + idx * tile_bytes);
                        ++idx;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_mask, q_cnt * kv_cnt);
            }
        }
    }
}
