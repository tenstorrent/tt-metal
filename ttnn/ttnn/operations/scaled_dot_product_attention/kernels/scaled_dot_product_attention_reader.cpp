// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for scaled_dot_product_attention (FlashAttention-2).
//
// Fills cb_scaler (1.0, for the MAX and SUM row-reduces) and cb_scale (the
// resolved attention scale, whole-tile fill for scalar-broadcast mul) once.
// Then, for each work unit (b, h, q-chunk) assigned to this core, reads the Q
// chunk once and streams every (K, V[, mask]) chunk along S_kv.
//
// All page addressing is via TensorAccessor (no InterleavedAddrGen). K/V point
// at kv_head = h / (H / H_kv) — the whole of GQA/MQA correctness.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_scale = 5;
}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(3);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(4);
    constexpr uint32_t Dt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t mask_H = get_compile_time_arg_val(10);
    constexpr uint32_t has_mask_v = get_compile_time_arg_val(11);
    constexpr bool has_mask = has_mask_v != 0;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(12);

    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_wu = get_arg_val<uint32_t>(4);
    const uint32_t num_wu = get_arg_val<uint32_t>(5);

    const uint32_t tile_bytes = get_tile_size(cb_q_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    // Mask accessor built once (constexpr args + fixed addr) — not re-created per KV chunk.
    const auto mask_acc = TensorAccessor(mask_args, mask_addr, tile_bytes);

    // --- scaler (1.0) for both MAX and SUM REDUCE_ROW; one tile serves both ---
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // --- scale tile: fill the whole tile with the resolved scale value ---
    {
        cb_reserve_back(cb_scale, 1);
        uint32_t wptr = get_write_ptr(cb_scale);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
        // fp32 -> bf16 round-to-nearest-even (truncation biases the scale toward
        // zero, shifting every softmax score low; RNE removes that systematic bias).
        const uint32_t rne_bias = 0x7FFFu + ((scale_bits >> 16) & 1u);
        const uint16_t sb = static_cast<uint16_t>((scale_bits + rne_bias) >> 16);
        const uint32_t packed = (static_cast<uint32_t>(sb) << 16) | sb;
        const uint32_t words = tile_bytes / 4;
        for (uint32_t i = 0; i < words; ++i) {
            p[i] = packed;
        }
        cb_push_back(cb_scale, 1);
    }

    const uint32_t HQ = H * n_q_chunks;
    const uint32_t group = H / H_kv;

    for (uint32_t wi = 0; wi < num_wu; ++wi) {
        const uint32_t w = start_wu + wi;
        const uint32_t b = w / HQ;
        const uint32_t r = w % HQ;
        const uint32_t h = r / n_q_chunks;
        const uint32_t qc = r % n_q_chunks;
        const uint32_t kv_head = h / group;
        const uint32_t mask_head = (mask_H == 1) ? 0 : h;

        // Q chunk: (Sq_chunk_t x Dt) tiles, row-major (sq, d)
        const uint32_t q_base = (b * H + h) * Sq_t;
        for (uint32_t sq = 0; sq < Sq_chunk_t; ++sq) {
            const uint32_t sq_g = qc * Sq_chunk_t + sq;
            for (uint32_t d = 0; d < Dt; ++d) {
                cb_reserve_back(cb_q_in, 1);
                noc_async_read_tile((q_base + sq_g) * Dt + d, q_acc, get_write_ptr(cb_q_in));
                noc_async_read_barrier();
                cb_push_back(cb_q_in, 1);
            }
        }

        const uint32_t kv_base = (b * H_kv + kv_head) * Skv_t;
        const uint32_t mask_base = (b * mask_H + mask_head) * Sq_t;

        for (uint32_t j = 0; j < n_kv_chunks; ++j) {
            // K chunk for Q.K^T: the transposed matmul reads in1 in K-major block
            // order (in1[k=d][n=skv] at d*Skv_chunk_t + skv), so lay K out D-major
            // (outer d, inner skv). The transpose flag flips each 32x32 tile's
            // contents; it does NOT reorder the block indices. DRAM page for K
            // tile (skv, d) is still (kv_base + skv_g)*Dt + d.
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t skv = 0; skv < Skv_chunk_t; ++skv) {
                    const uint32_t skv_g = j * Skv_chunk_t + skv;
                    cb_reserve_back(cb_k_in, 1);
                    noc_async_read_tile((kv_base + skv_g) * Dt + d, k_acc, get_write_ptr(cb_k_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_k_in, 1);
                }
            }
            // V chunk: (Skv_chunk_t x Dt) tiles, row-major (skv, d)
            for (uint32_t skv = 0; skv < Skv_chunk_t; ++skv) {
                const uint32_t skv_g = j * Skv_chunk_t + skv;
                for (uint32_t d = 0; d < Dt; ++d) {
                    cb_reserve_back(cb_v_in, 1);
                    noc_async_read_tile((kv_base + skv_g) * Dt + d, v_acc, get_write_ptr(cb_v_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_v_in, 1);
                }
            }
            // mask chunk: (Sq_chunk_t x Skv_chunk_t) tiles, row-major (sq, skv)
            if constexpr (has_mask) {
                for (uint32_t sq = 0; sq < Sq_chunk_t; ++sq) {
                    const uint32_t sq_g = qc * Sq_chunk_t + sq;
                    for (uint32_t skv = 0; skv < Skv_chunk_t; ++skv) {
                        const uint32_t skv_g = j * Skv_chunk_t + skv;
                        cb_reserve_back(cb_mask_in, 1);
                        noc_async_read_tile((mask_base + sq_g) * Skv_t + skv_g, mask_acc, get_write_ptr(cb_mask_in));
                        noc_async_read_barrier();
                        cb_push_back(cb_mask_in, 1);
                    }
                }
            }
        }
    }
}
