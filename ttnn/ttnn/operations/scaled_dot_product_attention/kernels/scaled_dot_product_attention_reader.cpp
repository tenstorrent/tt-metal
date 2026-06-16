// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention reader kernel.
//
// Prepares the two reduce scaler tiles (resident for the whole kernel), then
// for each assigned work-unit (b, h, q-chunk) streams the Q-chunk once and
// every KV-chunk (K, V, and optionally the mask tile) into the input CBs.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t H_q = get_compile_time_arg_val(0);
    constexpr uint32_t H_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(3);
    constexpr uint32_t d_t = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);  // H_q / H_kv
    constexpr uint32_t has_mask = get_compile_time_arg_val(6);
    constexpr uint32_t mask_H = get_compile_time_arg_val(7);  // mask num-heads (1 or H_q)
    constexpr uint32_t mask_B = get_compile_time_arg_val(8);  // mask batch (1 or B)

    constexpr auto q_args = TensorAccessorArgs<9>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;

    // Reduce scalers (value 1.0, pool-type-aware fill), resident for the kernel.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);

    const uint32_t tile_bytes = get_tile_size(cb_q_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, tile_bytes);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        // Decode unit -> (b, h, qc).
        const uint32_t qc = u % Sq_t;
        const uint32_t tmp = u / Sq_t;
        const uint32_t h = tmp % H_q;
        const uint32_t b = tmp / H_q;
        const uint32_t h_kv = h / group;
        // Mask broadcasting: dim0 collapses to 0 when mask_B==1 (batch-broadcast);
        // dim1 collapses to 0 when mask_H==1 (head-broadcast).
        const uint32_t mask_h = (mask_H == 1) ? 0 : h;
        const uint32_t mask_b = (mask_B == 1) ? 0 : b;

        // --- Q-chunk: d_t head tiles of query tile-row qc ---
        const uint32_t q_base = ((b * H_q + h) * Sq_t + qc) * d_t;
        cb_reserve_back(cb_q_in, d_t);
        uint32_t q_wr = get_write_ptr(cb_q_in);
        for (uint32_t dd = 0; dd < d_t; ++dd) {
            noc_async_read_page(q_base + dd, q_acc, q_wr);
            q_wr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, d_t);

        // --- KV loop ---
        for (uint32_t j = 0; j < Skv_t; ++j) {
            const uint32_t kv_base = ((b * H_kv + h_kv) * Skv_t + j) * d_t;

            cb_reserve_back(cb_k_in, d_t);
            uint32_t k_wr = get_write_ptr(cb_k_in);
            for (uint32_t dd = 0; dd < d_t; ++dd) {
                noc_async_read_page(kv_base + dd, k_acc, k_wr);
                k_wr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_k_in, d_t);

            cb_reserve_back(cb_v_in, d_t);
            uint32_t v_wr = get_write_ptr(cb_v_in);
            for (uint32_t dd = 0; dd < d_t; ++dd) {
                noc_async_read_page(kv_base + dd, v_acc, v_wr);
                v_wr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_in, d_t);

            if constexpr (has_mask) {
                const uint32_t mask_tile = ((mask_b * mask_H + mask_h) * Sq_t + qc) * Skv_t + j;
                cb_reserve_back(cb_mask_in, 1);
                uint32_t m_wr = get_write_ptr(cb_mask_in);
                noc_async_read_page(mask_tile, mask_acc, m_wr);
                noc_async_read_barrier();
                cb_push_back(cb_mask_in, 1);
            }
        }
    }
}
