// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for scaled_dot_product_attention (Flash Attention).
//
// Per-core work unit = one (batch, head) pair. For each work unit:
//   For each Q-block:
//     Load Q tiles into cb_q (retained across KV-blocks by compute)
//     For each KV-block:
//       Load K tiles (column-major for transpose=true matmul) into cb_k
//       Load V tiles (row-major for transpose=false PV matmul) into cb_v
//       Load mask tiles into cb_mask (if has_mask)
//
// GQA/MQA: kv_head_idx = q_head_idx // (H_q / H_kv). Reader maps each Q-head
// to its KV-head group via address computation — no compute kernel changes.
//
// Also prepares scaler tiles (MAX + SUM) and scale factor tile once.
//
// CT args: [has_mask, H_q, H_kv, mask_is_per_head,
//           ...Q_accessor, ...K_accessor, ...V_accessor, ...mask_accessor]
// RT args: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles,
//           b0, h0, b1, h1, ...,
//           q_addr, k_addr, v_addr, scale_bits, mask_addr]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices (match op_design.md CB layout).
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_max = 6;
constexpr uint32_t cb_scaler_sum = 7;
constexpr uint32_t cb_scale_factor = 5;

inline uint16_t fp32_bits_to_bf16_bits(uint32_t fp32_bits) {
    uint16_t lsw = static_cast<uint16_t>(fp32_bits & 0xFFFF);
    uint16_t bias = 0x7FFFu + (lsw >> 15);
    uint32_t rounded = fp32_bits + bias;
    return static_cast<uint16_t>(rounded >> 16);
}

inline void fill_bf16_tile_with_scalar_fp32(uint32_t cb_id, uint32_t fp32_bits) {
    uint16_t bf16_bits = fp32_bits_to_bf16_bits(fp32_bits);
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) ptr[i] = bf16_bits;
}

void kernel_main() {
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t mask_is_per_head = get_compile_time_arg_val(3);

    constexpr uint32_t tile_bytes = get_tile_size(cb_q);

    // --- Prepare reduce scalers (once per kernel) ---
    // MAX scaler: 1.0, row-0 fill (for MAX REDUCE_ROW, non-matmul path)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    // SUM scaler: 1.0, col-0 fill (for SUM REDUCE_ROW, matmul path)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // --- Read runtime args ---
    uint32_t rt_idx = 0;
    uint32_t num_work_units = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_q_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_kv_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t D_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_q_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_kv_tiles = get_arg_val<uint32_t>(rt_idx++);

    // Read (b, h) pairs
    uint32_t work_b[16], work_h[16];
    for (uint32_t i = 0; i < num_work_units; ++i) {
        work_b[i] = get_arg_val<uint32_t>(rt_idx++);
        work_h[i] = get_arg_val<uint32_t>(rt_idx++);
    }

    uint32_t q_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t k_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t v_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t scale_bits = get_arg_val<uint32_t>(rt_idx++);
    uint32_t mask_addr = get_arg_val<uint32_t>(rt_idx++);

    // Fill cb_scale_factor (once per kernel)
    cb_reserve_back(cb_scale_factor, 1);
    fill_bf16_tile_with_scalar_fp32(cb_scale_factor, scale_bits);
    cb_push_back(cb_scale_factor, 1);

    // TensorAccessor setup
    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);

    uint32_t h_q_div_h_kv = H_q / H_kv;
    uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t b = work_b[wu];
        uint32_t h_q = work_h[wu];
        uint32_t h_kv = h_q / h_q_div_h_kv;

        // Base tile indices in the interleaved tensor
        // Q: (B, H_q, S_q, D) → tile index = b*H_q*S_q_tiles*D_t + h_q*S_q_tiles*D_t + sq*D_t + d
        // K: (B, H_kv, S_kv, D) → tile index = b*H_kv*S_kv_tiles*D_t + h_kv*S_kv_tiles*D_t + sk*D_t + d
        // V: same as K
        // Mask: (B, 1orH, S_q, S_kv) → tile index = b*mask_h*S_q_tiles*S_kv_tiles + mh*S_q_tiles*S_kv_tiles +
        // sq*S_kv_tiles + sk
        uint32_t q_base = b * H_q * S_q_tiles * D_t + h_q * S_q_tiles * D_t;
        uint32_t k_base = b * H_kv * S_kv_tiles * D_t + h_kv * S_kv_tiles * D_t;
        uint32_t v_base = k_base;  // V has same shape as K
        uint32_t mask_h = mask_is_per_head ? h_q : 0;
        uint32_t mask_base =
            b * (mask_is_per_head ? H_q : 1) * S_q_tiles * S_kv_tiles + mask_h * S_q_tiles * S_kv_tiles;

        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            uint32_t q_row_start = qb * B_q_t;

            // Load Q tiles in row-major order: for r=0..B_q_t-1, for d=0..D_t-1
            // This matches the matmul's in0 access pattern with 1x1 subblocks
            for (uint32_t r = 0; r < B_q_t; ++r) {
                for (uint32_t d = 0; d < D_t; ++d) {
                    uint32_t q_row = q_row_start + r;
                    cb_reserve_back(cb_q, 1);
                    noc_async_read_tile(q_base + q_row * D_t + d, q_accessor, get_write_ptr(cb_q));
                    noc_async_read_barrier();
                    cb_push_back(cb_q, 1);
                }
            }

            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                uint32_t kv_col_start = kvb * B_kv_t;

                // Load K tiles in the order the QK^T matmul expects them.
                // With 1x1 subblocks and transpose=true, the matmul reads K tiles
                // with index pattern: for k=0..D_t-1, for n=0..B_kv_t-1: K[n][k]
                // (column-major in the (S_kv, D) matrix)
                for (uint32_t k = 0; k < D_t; ++k) {
                    for (uint32_t n = 0; n < B_kv_t; ++n) {
                        uint32_t kv_row = kv_col_start + n;
                        cb_reserve_back(cb_k, 1);
                        noc_async_read_tile(k_base + kv_row * D_t + k, k_accessor, get_write_ptr(cb_k));
                        noc_async_read_barrier();
                        cb_push_back(cb_k, 1);
                    }
                }

                // Load V tiles in row-major order: for n=0..B_kv_t-1, for d=0..D_t-1
                // This matches the PV matmul's in1 access pattern
                for (uint32_t n = 0; n < B_kv_t; ++n) {
                    for (uint32_t d = 0; d < D_t; ++d) {
                        uint32_t kv_row = kv_col_start + n;
                        cb_reserve_back(cb_v, 1);
                        noc_async_read_tile(v_base + kv_row * D_t + d, v_accessor, get_write_ptr(cb_v));
                        noc_async_read_barrier();
                        cb_push_back(cb_v, 1);
                    }
                }

                // Load mask tiles (if has_mask)
                if constexpr (has_mask) {
                    // Mask: (B, 1orH, S_q, S_kv) — row-major: for qr=0..B_q_t-1, for kc=0..B_kv_t-1
                    for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                        for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                            uint32_t q_row = q_row_start + qr;
                            uint32_t kv_col = kv_col_start + kc;
                            cb_reserve_back(cb_mask, 1);
                            noc_async_read_tile(
                                mask_base + q_row * S_kv_tiles + kv_col, mask_accessor, get_write_ptr(cb_mask));
                            noc_async_read_barrier();
                            cb_push_back(cb_mask, 1);
                        }
                    }
                }
            }
        }
    }
}
