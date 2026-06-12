// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention reader (NCRISC).
//
// Fills the constant CBs once at start:
//   cb_scale       — [0][0] = scale  (scalar-broadcast multiplier for scores)
//   cb_scaler_max  — 1.0, MAX/REDUCE_ROW fill layout (row-0)
//   cb_scaler_sum  — 1.0, SUM/REDUCE_ROW fill layout (col-0, matmul path)
//
// Then for each (b, h, q_block) work unit owned by this core: reads the Q block
// once (held resident across the KV loop) and, per KV block, streams K, V and —
// for custom masks — the matching mask block.
//
// q_chunk_t == k_chunk_t == 1: each Q/K/V block is D_t tiles; each mask block is
// one tile.
//
// Advisory: cb_scale is filled via prepare_reduce_scaler (row-0 fill) as the
// design's API mapping specifies — it puts `scale` at element [0][0], which is
// exactly what BroadcastDim::Scalar reads.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t D_t = get_compile_time_arg_val(0);
    constexpr uint32_t S_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t S_kv_t = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);  // H_q (Q/output heads)
    constexpr uint32_t mask_H = get_compile_time_arg_val(4);
    constexpr bool has_mask = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(6);
    constexpr uint32_t H_kv = get_compile_time_arg_val(7);  // K/V heads (== H for MHA)

    // GQA/MQA head broadcast: each Q head h maps to KV head h / group, where
    // group = H_q / H_kv (== 1 for MHA, == H_q for MQA). H % H_kv == 0 is
    // enforced in validate(), so this is exact integer division.
    constexpr uint32_t kv_group = H / H_kv;

    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_scale = 8;
    constexpr uint32_t cb_scaler_max = 9;
    constexpr uint32_t cb_scaler_sum = 15;

    // Constant CBs (filled once).
    const float scale_f = __builtin_bit_cast(float, scale_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_COL>(
        scale_f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);

    const uint32_t tile_bytes = get_tile_size(cb_q_in);

    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    [[maybe_unused]] const auto mask_acc = TensorAccessor(mask_args, mask_addr, tile_bytes);

    for (uint32_t idx = 0; idx < num_units; ++idx) {
        const uint32_t u = start_unit + idx;
        const uint32_t qi = u % S_q_t;
        const uint32_t bh = u / S_q_t;
        const uint32_t h = bh % H;
        const uint32_t b = bh / H;

        // Q block: D_t tiles, tile_id = ((b*H + h)*S_q_t + qi)*D_t + d
        {
            cb_reserve_back(cb_q_in, D_t);
            uint32_t l1 = get_write_ptr(cb_q_in);
            const uint32_t base = ((b * H + h) * S_q_t + qi) * D_t;
            for (uint32_t d = 0; d < D_t; ++d) {
                noc_async_read_tile(base + d, q_acc, l1 + d * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_q_in, D_t);
        }

        const uint32_t h_kv = h / kv_group;  // K/V head feeding this Q head

        for (uint32_t j = 0; j < S_kv_t; ++j) {
            const uint32_t kv_base = ((b * H_kv + h_kv) * S_kv_t + j) * D_t;

            // K block
            {
                cb_reserve_back(cb_k_in, D_t);
                uint32_t l1 = get_write_ptr(cb_k_in);
                for (uint32_t d = 0; d < D_t; ++d) {
                    noc_async_read_tile(kv_base + d, k_acc, l1 + d * tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_k_in, D_t);
            }

            // V block
            {
                cb_reserve_back(cb_v_in, D_t);
                uint32_t l1 = get_write_ptr(cb_v_in);
                for (uint32_t d = 0; d < D_t; ++d) {
                    noc_async_read_tile(kv_base + d, v_acc, l1 + d * tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_v_in, D_t);
            }

            // Mask block (custom only): one tile, broadcast across heads when mask_H == 1.
            if constexpr (has_mask) {
                const uint32_t mh = (mask_H == 1) ? 0 : h;
                const uint32_t mtile = ((b * mask_H + mh) * S_q_t + qi) * S_kv_t + j;
                cb_reserve_back(cb_mask_in, 1);
                uint32_t l1 = get_write_ptr(cb_mask_in);
                noc_async_read_tile(mtile, mask_acc, l1);
                noc_async_read_barrier();
                cb_push_back(cb_mask_in, 1);
            }
        }
    }
}
