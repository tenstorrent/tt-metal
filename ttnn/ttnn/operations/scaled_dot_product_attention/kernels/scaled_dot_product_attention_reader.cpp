// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for scaled_dot_product_attention (Flash Attention).
//
// Per-core work unit = one (batch, head) pair. For each work unit:
//   For each Q-block:
//     Push initial running state: m_i(-inf), l_i(0), O_i(0)
//     Load Q tiles into cb_q
//     For each KV-block:
//       Prepare MAX reduce scaler (1.0, row-0 fill)
//       Prepare SUM reduce scaler (1.0, col-0 fill)
//       Load K tiles (transposed order for QK^T matmul)
//       Load V tiles (row-major order for PV matmul)
//       Load mask tiles into cb_mask (if has_mask)
//
// GQA/MQA: kv_head_idx = q_head_idx // (H_q / H_kv). Reader maps each Q-head
// to its KV-head group via address computation — no compute kernel changes.
//
// CT args: [has_mask, H_q, H_kv,
//           ...Q_accessor, ...K_accessor, ...V_accessor, ...mask_accessor]
// RT args: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles,
//           b0, h0, b1, h1, ...,
//           q_addr, k_addr, scale_bits, v_addr, mask_addr]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices (match op_design.md CB layout)
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_reduce = 4;
constexpr uint32_t cb_scale_factor = 5;
constexpr uint32_t cb_o = tt::CBIndex::c_16;
constexpr uint32_t cb_max_old = 27;
constexpr uint32_t cb_sum_old = 30;

// bfloat16 bit pattern for -inf: sign=1, exponent=0xFF, mantissa=0 → 0xFF80
constexpr uint16_t NEG_INF_BFLOAT16 = 0xFF80;

// Fill a single bf16 tile (1024 elements) with a constant bf16 value.
inline void fill_bf16_tile_const(uint32_t cb_id, uint16_t val_bits) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) ptr[i] = val_bits;
}

// Fill a single bf16 tile with zeros.
inline void fill_bf16_tile_zero(uint32_t cb_id) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) ptr[i] = 0;
}

// Convert fp32 bits to bf16 bits (round-to-nearest-even).
inline uint16_t fp32_to_bf16_bits(uint32_t fp32_bits) {
    uint16_t lsw = static_cast<uint16_t>(fp32_bits & 0xFFFF);
    uint16_t bias = 0x7FFFu + (lsw >> 15);
    uint32_t rounded = fp32_bits + bias;
    return static_cast<uint16_t>(rounded >> 16);
}

// Fill a single bf16 tile with a scalar value from fp32 bits.
inline void fill_bf16_tile_scalar_fp32(uint32_t cb_id, uint32_t fp32_bits) {
    uint16_t bf16_bits = fp32_to_bf16_bits(fp32_bits);
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) ptr[i] = bf16_bits;
}

void kernel_main() {
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);

    // --- Read runtime args ---
    // RT layout: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles,
    //             b0, h0, b1, h1, ...,
    //             q_addr, k_addr, scale_bits, v_addr, mask_addr]
    uint32_t rt_idx = 0;
    uint32_t num_work_units = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_q_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t B_kv_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t D_t = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_q_tiles = get_arg_val<uint32_t>(rt_idx++);
    uint32_t S_kv_tiles = get_arg_val<uint32_t>(rt_idx++);

    uint32_t num_o_tiles = B_q_t * D_t;
    constexpr uint32_t tile_bytes = get_tile_size(cb_q);

    // Read (b, h) pairs — at most 16 work units per core
    uint32_t work_b[16], work_h[16];
    for (uint32_t i = 0; i < num_work_units; ++i) {
        work_b[i] = get_arg_val<uint32_t>(rt_idx++);
        work_h[i] = get_arg_val<uint32_t>(rt_idx++);
    }

    uint32_t q_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t k_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t scale_bits = get_arg_val<uint32_t>(rt_idx++);
    uint32_t v_addr = get_arg_val<uint32_t>(rt_idx++);
    uint32_t mask_addr = get_arg_val<uint32_t>(rt_idx++);

    // --- Fill cb_scale_factor (once per kernel, HeldBulk in compute) ---
    cb_reserve_back(cb_scale_factor, 1);
    fill_bf16_tile_scalar_fp32(cb_scale_factor, scale_bits);
    cb_push_back(cb_scale_factor, 1);

    // TensorAccessor setup — all accessors declared unconditionally
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const auto q_accessor = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_accessor = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_accessor = TensorAccessor(v_args, v_addr, tile_bytes);
    [[maybe_unused]] const auto mask_accessor = TensorAccessor(mask_args, mask_addr, tile_bytes);

    uint32_t h_q_div_h_kv = H_q / H_kv;
    uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t b = work_b[wu];
        uint32_t h_q = work_h[wu];
        uint32_t h_kv = h_q / h_q_div_h_kv;

        // Base tile indices in the interleaved DRAM buffers.
        // Q: (B, H_q, S_q_tiles, D_t)
        // K: (B, H_kv, S_kv_tiles, D_t)
        // V: same layout as K
        uint32_t q_base = b * H_q * S_q_tiles * D_t + h_q * S_q_tiles * D_t;
        uint32_t k_base = b * H_kv * S_kv_tiles * D_t + h_kv * S_kv_tiles * D_t;
        uint32_t v_base = k_base;
        // Mask: (B, 1, S_q, S_kv)
        uint32_t mask_base = b * S_q_tiles * S_kv_tiles;

        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            uint32_t q_row_start = qb * B_q_t;

            // --- Push initial running state for this Q-block ---
            // m_i = -inf (B_q_t tiles)
            for (uint32_t t = 0; t < B_q_t; ++t) {
                cb_reserve_back(cb_max_old, 1);
                fill_bf16_tile_const(cb_max_old, NEG_INF_BFLOAT16);
                cb_push_back(cb_max_old, 1);
            }
            // l_i = 0 (B_q_t tiles)
            for (uint32_t t = 0; t < B_q_t; ++t) {
                cb_reserve_back(cb_sum_old, 1);
                fill_bf16_tile_zero(cb_sum_old);
                cb_push_back(cb_sum_old, 1);
            }
            // O_i = 0 (num_o_tiles tiles)
            for (uint32_t t = 0; t < num_o_tiles; ++t) {
                cb_reserve_back(cb_o, 1);
                fill_bf16_tile_zero(cb_o);
                cb_push_back(cb_o, 1);
            }

            // --- Load Q tiles for this Q-block ---
            // Q layout: tile(q_row, d_col) at page q_base + q_row*D_t + d_col
            for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                uint32_t q_row = q_row_start + qr;
                for (uint32_t dc = 0; dc < D_t; ++dc) {
                    cb_reserve_back(cb_q, 1);
                    noc_async_read_tile(q_base + q_row * D_t + dc, q_accessor, get_write_ptr(cb_q));
                    noc_async_read_barrier();
                    cb_push_back(cb_q, 1);
                }
            }

            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                uint32_t kv_col_start = kvb * B_kv_t;

                // --- Prepare reduce scalers for this KV-block ---
                // MAX REDUCE_ROW: uses reduce_tile LLK → row-0 fill (scaler = 1.0)
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    cb_scaler_reduce,
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW>();
                // SUM REDUCE_ROW: uses matmul path → col-0 fill (scaler = 1.0)
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    cb_scaler_reduce,
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW>();

                // --- Load K tiles (transposed order for QK^T matmul) ---
                // matmul_block with transpose=true expects in1 in K-major order:
                // for k in [0, D_t): for n in [0, B_kv_t): K tile at (kv_row=n, k_col=k)
                for (uint32_t k = 0; k < D_t; ++k) {
                    for (uint32_t n = 0; n < B_kv_t; ++n) {
                        uint32_t kv_row = kv_col_start + n;
                        cb_reserve_back(cb_k, 1);
                        noc_async_read_tile(k_base + kv_row * D_t + k, k_accessor, get_write_ptr(cb_k));
                        noc_async_read_barrier();
                        cb_push_back(cb_k, 1);
                    }
                }

                // --- Load V tiles (row-major for PV matmul) ---
                // PV matmul: P(B_q x B_kv) @ V(B_kv x D) → out(B_q x D)
                // in1=cb_v needs tiles in column-major order for transpose=false:
                // for k in [0, B_kv_t): for n in [0, D_t): V tile at (kv_row=k, d_col=n)
                for (uint32_t k = 0; k < B_kv_t; ++k) {
                    uint32_t kv_row = kv_col_start + k;
                    for (uint32_t d = 0; d < D_t; ++d) {
                        cb_reserve_back(cb_v, 1);
                        noc_async_read_tile(v_base + kv_row * D_t + d, v_accessor, get_write_ptr(cb_v));
                        noc_async_read_barrier();
                        cb_push_back(cb_v, 1);
                    }
                }

                // --- Load mask tiles (if has_mask) ---
                // Mask: (B, 1, S_q, S_kv) — tile(q_row, kv_col) at mask_base + q_row*S_kv_tiles + kv_col
                if constexpr (has_mask) {
                    for (uint32_t qr = 0; qr < B_q_t; ++qr) {
                        uint32_t q_row = q_row_start + qr;
                        for (uint32_t kc = 0; kc < B_kv_t; ++kc) {
                            uint32_t kv_col = kv_col_start + kc;
                            cb_reserve_back(cb_mask, 1);
                            noc_async_read_tile(
                                mask_base + q_row * S_kv_tiles + kv_col, mask_accessor, get_write_ptr(cb_mask));
                            noc_async_read_barrier();
                            cb_push_back(cb_mask, 1);
                        }
                    }
                }
            }  // end KV-block loop
        }  // end Q-block loop
    }  // end work-unit loop
}
