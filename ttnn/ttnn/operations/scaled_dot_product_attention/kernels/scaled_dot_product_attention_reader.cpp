// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flash-Attention SDPA reader (NCRISC).
//
// Per work unit (b, h, q):
//   - stream the resident Q tile-row once (Dt tiles, natural order) -> cb_q
//   - for each KV block j:
//       - K block -> cb_k in TRANSPOSED block order: cb_k[kd*kv_chunk_t + n] =
//         K_tile[kv-row j*kv_chunk_t+n, D-col kd]. Combined with the matmul's
//         within-tile transpose (transpose=true) this yields Q·Kᵀ.
//       - V block -> cb_v in NATURAL block order: cb_v[kj*Dt + nd] =
//         V_tile[kv-row j*kv_chunk_t+kj, D-col nd].
//       - mask block -> cb_mask natural [1, kv_chunk_t] (only if use_mask).
//
// Scaler CBs (cb_scaler_max row-0 fill for MAX, cb_scaler_sum col-0 fill for the
// matmul-path SUM REDUCE_ROW) are prepared once per core via the pool-type-aware
// dataflow helper.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t H_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(2);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(3);
    constexpr uint32_t Dt = get_compile_time_arg_val(4);
    constexpr uint32_t kv_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t num_kv_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t mask_H = get_compile_time_arg_val(7);
    constexpr uint32_t use_mask = get_compile_time_arg_val(8);
    constexpr uint32_t cb_q = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k = get_compile_time_arg_val(10);
    constexpr uint32_t cb_v = get_compile_time_arg_val(11);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaler_max = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scaler_sum = get_compile_time_arg_val(14);

    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    // GQA/MQA mapping (Phase 0: MHA, so group == 1, h_kv == h).
    constexpr uint32_t head_group = H / H_kv;

    const uint32_t q_page_bytes = get_local_cb_interface(cb_q).fifo_page_size;
    const uint32_t k_page_bytes = get_local_cb_interface(cb_k).fifo_page_size;
    const uint32_t v_page_bytes = get_local_cb_interface(cb_v).fifo_page_size;

    const auto q_acc = TensorAccessor(q_args, q_addr, q_page_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_page_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, v_page_bytes);

    Noc noc;
    CircularBuffer q_cb(cb_q), k_cb(cb_k), v_cb(cb_v);

    // Reduce scalers (value 1.0): pool-type-aware fill — MAX uses row-0, SUM
    // (matmul-path REDUCE_ROW) uses col-0. Filled once per core.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    constexpr uint32_t kv_block_tiles = kv_chunk_t * Dt;

    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t unit = start_unit + u;
        const uint32_t b = unit / (H * Sq_t);
        const uint32_t h = (unit / Sq_t) % H;
        const uint32_t q = unit % Sq_t;
        const uint32_t h_kv = h / head_group;

        const uint32_t q_row_base = ((b * H + h) * Sq_t + q) * Dt;  // tile (b,h,q,0)
        const uint32_t kv_head_base = (b * H_kv + h_kv) * Skv_t;    // tile-row base for K/V

        // ---- Q tile-row (Dt tiles, natural order) ----
        q_cb.reserve_back(Dt);
        for (uint32_t dc = 0; dc < Dt; ++dc) {
            noc.async_read(
                q_acc, q_cb, q_page_bytes, {.page_id = q_row_base + dc}, {.offset_bytes = dc * q_page_bytes});
        }
        noc.async_read_barrier();
        q_cb.push_back(Dt);

        for (uint32_t j = 0; j < num_kv_chunks; ++j) {
            const uint32_t kv_row0 = j * kv_chunk_t;  // first kv tile-row of this block

            // ---- K block, transposed block order: position kd*kv_chunk_t + n ----
            k_cb.reserve_back(kv_block_tiles);
            for (uint32_t kd = 0; kd < Dt; ++kd) {
                for (uint32_t n = 0; n < kv_chunk_t; ++n) {
                    const uint32_t kr = kv_row0 + n;
                    const uint32_t page = (kv_head_base + kr) * Dt + kd;
                    const uint32_t pos = kd * kv_chunk_t + n;
                    noc.async_read(k_acc, k_cb, k_page_bytes, {.page_id = page}, {.offset_bytes = pos * k_page_bytes});
                }
            }
            noc.async_read_barrier();
            k_cb.push_back(kv_block_tiles);

            // ---- V block, natural block order: position kj*Dt + nd ----
            v_cb.reserve_back(kv_block_tiles);
            for (uint32_t kj = 0; kj < kv_chunk_t; ++kj) {
                const uint32_t kr = kv_row0 + kj;
                for (uint32_t nd = 0; nd < Dt; ++nd) {
                    const uint32_t page = (kv_head_base + kr) * Dt + nd;
                    const uint32_t pos = kj * Dt + nd;
                    noc.async_read(v_acc, v_cb, v_page_bytes, {.page_id = page}, {.offset_bytes = pos * v_page_bytes});
                }
            }
            noc.async_read_barrier();
            v_cb.push_back(kv_block_tiles);

            // ---- mask block, natural [1, kv_chunk_t] ----
            if constexpr (use_mask) {
                CircularBuffer mask_cb(cb_mask);
                const uint32_t mask_page_bytes = get_local_cb_interface(cb_mask).fifo_page_size;
                const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_page_bytes);
                const uint32_t mh = (mask_H == 1) ? 0 : h;
                const uint32_t mask_row_base = ((b * mask_H + mh) * Sq_t + q) * Skv_t;
                mask_cb.reserve_back(kv_chunk_t);
                for (uint32_t n = 0; n < kv_chunk_t; ++n) {
                    const uint32_t page = mask_row_base + (kv_row0 + n);
                    noc.async_read(
                        mask_acc, mask_cb, mask_page_bytes, {.page_id = page}, {.offset_bytes = n * mask_page_bytes});
                }
                noc.async_read_barrier();
                mask_cb.push_back(kv_chunk_t);
            }
        }
    }
}
