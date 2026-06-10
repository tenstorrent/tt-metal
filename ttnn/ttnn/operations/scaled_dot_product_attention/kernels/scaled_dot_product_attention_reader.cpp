// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention reader (NCRISC / NoC0).
//
// Per work unit (b, h, q_chunk):
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

    constexpr auto q_args = TensorAccessorArgs<12>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    if (num_units == 0) {
        return;
    }

    constexpr uint32_t cb_q_tiles = 0;
    constexpr uint32_t cb_kt_tiles = 1;
    constexpr uint32_t cb_v_tiles = 2;
    constexpr uint32_t cb_mask_tiles = 3;
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;

    const uint32_t tile_bytes = get_tile_size(cb_q_tiles);
    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);

    // Scalers once per program (pool-type-aware fill: MAX/ROW row0, SUM/ROW col0).
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t bh = unit / Nq;  // flattened b*H + h
        const uint32_t qc = unit % Nq;  // q-chunk index
        const uint32_t cur_cq = (qc == Nq - 1) ? c_q_last : c_q;
        const uint32_t q_row0 = qc * c_q;

        // Phase 0 head: H_kv == H, so K/V share the bh head index.
        const uint32_t qkv_head_base = bh * Skv_t * Dt;  // K/V tile base for this head
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

            // K^T tiles: tile-order (d, n) -> K[bh, n0+n, d]. Intra-tile transpose
            // is done by the matmul's transpose=true — both halves are required.
            cb_reserve_back(cb_kt_tiles, Dt * cur_ckv);
            uint32_t l1_addr = get_write_ptr(cb_kt_tiles);
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t n = 0; n < cur_ckv; ++n) {
                    noc_async_read_tile(qkv_head_base + (n0 + n) * Dt + d, k_accessor, l1_addr);
                    l1_addr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_kt_tiles, Dt * cur_ckv);

            // V tiles: row-major (n, d).
            cb_reserve_back(cb_v_tiles, cur_ckv * Dt);
            l1_addr = get_write_ptr(cb_v_tiles);
            for (uint32_t n = 0; n < cur_ckv; ++n) {
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc_async_read_tile(qkv_head_base + (n0 + n) * Dt + d, v_accessor, l1_addr);
                    l1_addr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_tiles, cur_ckv * Dt);

            // Mask tiles: row-major (r, n) over [q_row0, q_row0+cur_cq) x [n0, n0+cur_ckv).
            if constexpr (HAS_MASK) {
                const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);
                const uint32_t b = bh / H;
                const uint32_t h = bh % H;
                const uint32_t mask_head = MASK_PER_HEAD ? (b * H + h) : b;
                const uint32_t mask_base = mask_head * Sq_t * Skv_t;
                cb_reserve_back(cb_mask_tiles, cur_cq * cur_ckv);
                l1_addr = get_write_ptr(cb_mask_tiles);
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
