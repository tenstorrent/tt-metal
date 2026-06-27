// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): Initializes the running-state CBs with constant fills:
//   cb_max_old (27) ← -inf tiles (B_q_t tiles, running max m_i)
//   cb_sum_old (30) ← 0.0   tiles (B_q_t tiles, running sum l_i)
//   cb_o       (16) ← 0.0   tiles (B_q_t * D_t tiles, running output O_i)
//
// Stage 1 (qkt_matmul): Streams Q-block and K-block tiles from DRAM into L1 CBs.
//   cb_q  (0) ← Q tiles (B_q_t * D_t tiles, loaded once, retained)
//   cb_k  (1) ← K tiles (B_kv_t * D_t tiles, streamed per KV-block)
//
// Stage 2 (scale): Fills cb_scale_factor (1 tile) with the scale value
//   (1/sqrt(D) or explicit). The scale arrives as fp32 bits (uint32_t)
//   in runtime args; the reader converts to bf16 and fills the tile.
//   cb_scale_factor is HeldBulk by the compute eltwise mul (never popped).
//
// Stage 4 (rowmax): Prepares the reduce scaler tile in cb_scaler_reduce (1 tile)
//   via calculate_and_prepare_reduce_scaler<MAX, REDUCE_ROW>. Scaler = 1.0 for MAX.
//   The reduce helper waits for this tile and never pops it (caller pops).
//
// The Q tiles are loaded in standard tile-row-major order:
//   tile(r, d) at DRAM page r * D_t + d  (row r, head-dim-tile d)
//   pushed to cb_q in the same order: [0,1,2,...,B_q_t*D_t-1]
//
// The K tiles for transpose matmul must be loaded in a specific order. With
// transpose=true, the matmul LLK accesses in1 tiles with stride in1_per_core_w
// along the K dimension. The expected in1 tile order for one K-block is:
//   For each head-dim-tile k in [0, D_t):
//     For each kv-col-tile n in [0, B_kv_t):
//       push K tile at DRAM page n * D_t + k
// This gives cb_k order: [K(0,0), K(1,0), ..., K(B_kv-1,0),
//                          K(0,1), K(1,1), ..., K(B_kv-1,1),
//                          ...]
// which the transpose matmul reads as K^T.
//
// Constant-fill pattern for init: cb_reserve_back → write loop → cb_push_back.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices (match op_design.md CB layout).
constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_reduce = 4;
constexpr uint32_t cb_scale_factor = 5;
constexpr uint32_t cb_o = tt::CBIndex::c_16;
constexpr uint32_t cb_max_old = 27;
constexpr uint32_t cb_sum_old = 30;

// bf16 representation of -inf (0xFF80).
constexpr uint16_t NEG_INF_BFLOAT16 = 0xFF80;

// Fill an entire CB tile (bf16, 32x32 = 1024 elements) with a constant
// uint16 bit pattern.
inline void fill_bf16_tile_with_const(uint32_t cb_id, uint16_t val_bits) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = val_bits;
    }
}

// Zero-fill a CB tile (bf16).
inline void fill_bf16_tile_zero(uint32_t cb_id) {
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
}

// Convert fp32 bits → bf16 bits (round-to-nearest-even, truncating low 16 bits).
inline uint16_t fp32_bits_to_bf16_bits(uint32_t fp32_bits) {
    // Round-to-nearest-even: add bias 0x7FFF + (sticky bit) before truncation.
    uint16_t lsw = static_cast<uint16_t>(fp32_bits & 0xFFFF);
    uint16_t bias = 0x7FFFu + (lsw >> 15);
    uint32_t rounded = fp32_bits + bias;
    return static_cast<uint16_t>(rounded >> 16);
}

// Fill an entire CB tile (bf16, 32x32 = 1024 elements) with a scalar value
// given as fp32 bits. Converts to bf16 and broadcasts across all elements.
inline void fill_bf16_tile_with_scalar_fp32(uint32_t cb_id, uint32_t fp32_bits) {
    uint16_t bf16_bits = fp32_bits_to_bf16_bits(fp32_bits);
    uint32_t write_addr = get_write_ptr(cb_id);
    auto ptr = reinterpret_cast<volatile uint16_t*>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = bf16_bits;
    }
}

void kernel_main() {
    // Compile-time args (from program descriptor).
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);       // Q-block tile rows (4)
    constexpr uint32_t D_t = get_compile_time_arg_val(1);         // head-dim tiles (D/32)
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(2);      // KV-block tile cols (4)

    constexpr uint32_t tile_bytes = get_tile_size(cb_q);

    // --- Stage 0: initialize running state ---

    // cb_max_old: B_q_t tiles of -inf (running max m_i = -inf).
    for (uint32_t t = 0; t < B_q_t; ++t) {
        cb_reserve_back(cb_max_old, 1);
        fill_bf16_tile_with_const(cb_max_old, NEG_INF_BFLOAT16);
        cb_push_back(cb_max_old, 1);
    }

    // cb_sum_old: B_q_t tiles of 0.0 (running sum l_i = 0).
    for (uint32_t t = 0; t < B_q_t; ++t) {
        cb_reserve_back(cb_sum_old, 1);
        fill_bf16_tile_zero(cb_sum_old);
        cb_push_back(cb_sum_old, 1);
    }

    // cb_o: B_q_t * D_t tiles of 0.0 (running output O_i = 0).
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    for (uint32_t t = 0; t < num_o_tiles; ++t) {
        cb_reserve_back(cb_o, 1);
        fill_bf16_tile_zero(cb_o);
        cb_push_back(cb_o, 1);
    }

    // --- Stage 2: fill cb_scale_factor with scale value ---
    // The scale (1/sqrt(D) or explicit) is passed as fp32 bits in runtime args.
    // The reader converts to bf16 and fills a single tile. The compute kernel
    // uses this as a HeldBulk scalar broadcast for the eltwise mul.
    {
        uint32_t rt_arg_idx = 2;  // scale_bits is runtime arg index 2
        uint32_t scale_bits = get_arg_val<uint32_t>(rt_arg_idx);
        cb_reserve_back(cb_scale_factor, 1);
        fill_bf16_tile_with_scalar_fp32(cb_scale_factor, scale_bits);
        cb_push_back(cb_scale_factor, 1);
    }

    // --- Stage 4: prepare reduce scaler for MAX REDUCE_ROW ---
    // Scaler = 1.0 for MAX. The pool-type-aware helper selects the correct
    // layout (row-0 fill for MAX reduce_tile path). The reduce helper waits
    // for this tile and never pops it (caller pops after reduce completes).
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler_reduce, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();

    // --- Stage 1: stream Q and K tiles from DRAM ---

    // Runtime args: [q_addr, k_addr, scale_bits (fp32 bits)]
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t scale_bits = get_arg_val<uint32_t>(2);  // scale as fp32 bit representation

    // Reconstruct TensorAccessor layout from compile-time args.
    // CT args layout: [B_q_t(0), D_t(1), B_kv_t(2), ...Q_accessor(3+), ...K_accessor(after Q)]
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    // --- Read Q tiles (tile-row-major: row r, head-dim d → page r*D_t + d) ---
    {
        const auto accessor = TensorAccessor(q_args, q_addr, tile_bytes);
        constexpr uint32_t num_q_tiles = B_q_t * D_t;
        for (uint32_t t = 0; t < num_q_tiles; ++t) {
            cb_reserve_back(cb_q, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_q);
            noc_async_read_tile(t, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_q, 1);
        }
    }

    // --- Read K tiles (transposed order for transpose=true matmul) ---
    // For transpose matmul: K tiles must be loaded as K^T column tiles.
    // in1_per_core_w = B_kv_t, in0_block_k = D_t.
    // The matmul reads in1 with stride in1_per_core_w along K.
    // Required cb_k order: for k in [0,D_t): for n in [0,B_kv_t): K[n*D_t + k]
    {
        const auto accessor = TensorAccessor(k_args, k_addr, tile_bytes);
        constexpr uint32_t num_k_tiles = B_kv_t * D_t;
        for (uint32_t k = 0; k < D_t; ++k) {
            for (uint32_t n = 0; n < B_kv_t; ++n) {
                uint32_t dram_page = n * D_t + k;
                cb_reserve_back(cb_k, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_k);
                noc_async_read_tile(dram_page, accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_k, 1);
            }
        }
    }
}
