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
constexpr uint32_t cb_kv_mask = 8;

// bf16 tile: 4 faces of 16x16, each face row-major, 2 bf16 packed per uint32
// (low16 = even col, high16 = odd col), 8 uint32 words per face-row, 128 words
// per face, 512 words per tile. -inf(bf16) = 0xFF80, packed pair = 0xFF80FF80.
constexpr uint32_t NEGINF_PAIR = 0xFF80FF80u;

FORCE_INLINE void fill_zeros_tile(uint32_t wptr, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0u;
    }
}

// Vertical column mask: columns [0, unpad_col) = 0, columns [unpad_col, 32) = -inf.
// (Face-aware; mirrors the production SDPA fill_vertical_tile_bf16.)
FORCE_INLINE void fill_vertical_mask_tile(uint32_t wptr, uint32_t unpad_col) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
    constexpr uint32_t FACE_W = 16;
    constexpr uint32_t FACE_H = 16;
    constexpr uint32_t words_per_face_row = FACE_W / 2;               // 8
    constexpr uint32_t words_per_face = FACE_H * words_per_face_row;  // 128
    const uint32_t face_col_start[4] = {0, 16, 0, 16};
    for (uint32_t f = 0; f < 4; ++f) {
        const uint32_t base = f * words_per_face;
        const uint32_t col_start = face_col_start[f];
        if (unpad_col <= col_start) {
            for (uint32_t i = 0; i < words_per_face; ++i) {
                ptr[base + i] = NEGINF_PAIR;  // whole face padded
            }
        } else if (unpad_col >= col_start + FACE_W) {
            for (uint32_t i = 0; i < words_per_face; ++i) {
                ptr[base + i] = 0u;  // whole face valid
            }
        } else {
            const uint32_t local_col = unpad_col - col_start;  // 1..15 (first padded col)
            const uint32_t boundary_word = local_col / 2;
            const uint32_t boundary_pos = local_col % 2;
            for (uint32_t row = 0; row < FACE_H; ++row) {
                const uint32_t row_base = base + row * words_per_face_row;
                for (uint32_t w = 0; w < boundary_word; ++w) {
                    ptr[row_base + w] = 0u;  // valid columns
                }
                uint32_t start_word = boundary_word;
                if (boundary_pos != 0) {
                    // low16 = even col (valid, 0), high16 = odd col (-inf)
                    ptr[row_base + boundary_word] = 0xFF800000u;
                    start_word = boundary_word + 1;
                }
                for (uint32_t w = start_word; w < words_per_face_row; ++w) {
                    ptr[row_base + w] = NEGINF_PAIR;  // padded columns
                }
            }
        }
    }
}
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
    constexpr uint32_t skv_partial = get_compile_time_arg_val(13);  // valid cols in last S_kv tile (0 => aligned)
    constexpr bool has_kv_pad = skv_partial != 0;

    constexpr auto q_args = TensorAccessorArgs<14>();
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

        // R1b partial q-chunk: only the valid tile-rows of the last q-chunk exist in
        // DRAM. sq_valid <= Sq_chunk_t; compute/writer consume the same count.
        const uint32_t sq_off = qc * Sq_chunk_t;
        const uint32_t sq_valid = (Sq_chunk_t < Sq_t - sq_off) ? Sq_chunk_t : (Sq_t - sq_off);

        // Q chunk: (sq_valid x Dt) tiles, row-major (sq, d)
        const uint32_t q_base = (b * H + h) * Sq_t;
        for (uint32_t sq = 0; sq < sq_valid; ++sq) {
            const uint32_t sq_g = sq_off + sq;
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
            // R1b partial KV chunk: skv_valid <= Skv_chunk_t whole tiles this chunk
            // (only the last chunk is partial). The read layouts stay contiguous at
            // width skv_valid, which is exactly the matmul N (QKᵀ) / K (PV) extent.
            const uint32_t skv_off = j * Skv_chunk_t;
            const uint32_t skv_valid = (Skv_chunk_t < Skv_t - skv_off) ? Skv_chunk_t : (Skv_t - skv_off);

            // K chunk for Q.K^T: the transposed matmul reads in1 in K-major block
            // order (in1[k=d][n=skv] at d*skv_valid + skv), so lay K out D-major
            // (outer d, inner skv). The transpose flag flips each 32x32 tile's
            // contents; it does NOT reorder the block indices. DRAM page for K
            // tile (skv, d) is still (kv_base + skv_g)*Dt + d.
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t skv = 0; skv < skv_valid; ++skv) {
                    const uint32_t skv_g = skv_off + skv;
                    cb_reserve_back(cb_k_in, 1);
                    noc_async_read_tile((kv_base + skv_g) * Dt + d, k_acc, get_write_ptr(cb_k_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_k_in, 1);
                }
            }
            // V chunk: (skv_valid x Dt) tiles, row-major (skv, d)
            for (uint32_t skv = 0; skv < skv_valid; ++skv) {
                const uint32_t skv_g = skv_off + skv;
                for (uint32_t d = 0; d < Dt; ++d) {
                    cb_reserve_back(cb_v_in, 1);
                    noc_async_read_tile((kv_base + skv_g) * Dt + d, v_acc, get_write_ptr(cb_v_in));
                    noc_async_read_barrier();
                    cb_push_back(cb_v_in, 1);
                }
            }
            // mask chunk: (sq_valid x skv_valid) tiles, row-major (sq, skv)
            if constexpr (has_mask) {
                for (uint32_t sq = 0; sq < sq_valid; ++sq) {
                    const uint32_t sq_g = sq_off + sq;
                    for (uint32_t skv = 0; skv < skv_valid; ++skv) {
                        const uint32_t skv_g = skv_off + skv;
                        cb_reserve_back(cb_mask_in, 1);
                        noc_async_read_tile((mask_base + sq_g) * Skv_t + skv_g, mask_acc, get_write_ptr(cb_mask_in));
                        noc_async_read_barrier();
                        cb_push_back(cb_mask_in, 1);
                    }
                }
            }

            // KV-padding softmax mask (h_non_aligned): last KV chunk only. Build a
            // score-block-shaped additive mask — the last S_kv-column tile of each
            // Q row carries the vertical -inf mask, all other tiles are zero — so
            // compute drives the last KV tile's padding columns to -inf before the
            // row-max/exp/row-sum. The boundary tile is the last VALID tile of the
            // (possibly partial R1b) last chunk, at local index skv_valid - 1.
            if constexpr (has_kv_pad) {
                if (j == n_kv_chunks - 1) {
                    const uint32_t mask_words = tile_bytes / 4;
                    for (uint32_t sq = 0; sq < sq_valid; ++sq) {
                        for (uint32_t skv = 0; skv < skv_valid; ++skv) {
                            cb_reserve_back(cb_kv_mask, 1);
                            const uint32_t wptr = get_write_ptr(cb_kv_mask);
                            if (skv == skv_valid - 1) {
                                fill_vertical_mask_tile(wptr, skv_partial);
                            } else {
                                fill_zeros_tile(wptr, mask_words);
                            }
                            cb_push_back(cb_kv_mask, 1);
                        }
                    }
                }
            }
        }
    }
}
