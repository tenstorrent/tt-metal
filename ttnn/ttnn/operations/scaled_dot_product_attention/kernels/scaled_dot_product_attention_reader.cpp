// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SDPA reader (NCRISC).
//
// Streams Q (Dt tiles, one-shot per query row), K (Dt tiles per K-iter),
// V (Dt tiles per K-iter), and the optional attention mask (1 tile per
// K-iter) from DRAM into per-core L1 CBs. Also emits the bf16 1.0
// scaler tile that the row-reduce phase needs.
//
// Refinement 4: per query row the K-axis is now traversed TWICE to
// support the two-pass output-normalization compute path. Pass 1 streams
// K + mask only (compute uses them to find global_max + global_sum_exp).
// Pass 2 streams K + V + mask again (compute does the direct output
// accumulation against the fixed global_max). Re-reading K/V from DRAM
// is the verifier-named L1 trade-off — parking attention weights in L1
// across the K-loop would cost Kt × Wt tile slots (~MB at S=8192).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// ---------------------------------------------------------------------------
// Tile-internal addressing helper (Refinement 3 alignment-mask overlay).
//
// A 32×32 tile is stored as 4 16×16 faces in faced order:
//   face 0 = rows  0-15, cols  0-15
//   face 1 = rows  0-15, cols 16-31
//   face 2 = rows 16-31, cols  0-15
//   face 3 = rows 16-31, cols 16-31
// Elements within a face are row-major. The byte offset of logical
// position (row, col) in a tile is therefore
//     face   = (row >> 4) * 2 + (col >> 4)
//     in_face = (row & 0xF) * 16 + (col & 0xF)
//     offset = (face * 256 + in_face) * ElemBytes
// All math is constexpr-foldable for a fixed (row, col).
// ---------------------------------------------------------------------------
namespace {
template <uint32_t ElemBytes>
FORCE_INLINE void overlay_neg_inf_on_padded_cols(uint32_t l1_addr, uint32_t valid_cols) {
    auto* p_bytes = reinterpret_cast<volatile uint8_t*>(l1_addr);
    // BF16 / FP32 -inf bit patterns. These match what `torch.tensor(-inf,
    // dtype=...)` produces, which is what the PyTorch reference scores
    // get when softmax sees a -inf score (→ exp(-inf) = 0).
    constexpr uint16_t BF16_NEG_INF = 0xFF80u;
    constexpr uint32_t FP32_NEG_INF = 0xFF800000u;

    for (uint32_t col = valid_cols; col < 32; ++col) {
        const uint32_t face_col_idx = col >> 4;
        const uint32_t face_col = col & 0xFu;
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t face_row_idx = row >> 4;
            const uint32_t face_row = row & 0xFu;
            const uint32_t face = face_row_idx * 2u + face_col_idx;
            const uint32_t off_elem = face * 256u + face_row * 16u + face_col;
            const uint32_t off_bytes = off_elem * ElemBytes;
            if constexpr (ElemBytes == 2u) {
                *reinterpret_cast<volatile uint16_t*>(p_bytes + off_bytes) = BF16_NEG_INF;
            } else if constexpr (ElemBytes == 4u) {
                *reinterpret_cast<volatile uint32_t*>(p_bytes + off_bytes) = FP32_NEG_INF;
            }
        }
    }
}

FORCE_INLINE void zero_fill_tile(uint32_t l1_addr, uint32_t tile_bytes) {
    // Simple word-wise zero fill. Compiler unrolls / vectorises.
    auto* p = reinterpret_cast<volatile uint32_t*>(l1_addr);
    const uint32_t n_words = tile_bytes >> 2;
    for (uint32_t i = 0; i < n_words; ++i) {
        p[i] = 0u;
    }
}
}  // namespace

void kernel_main() {
    // --- Compile-time args -------------------------------------------------
    // Refinement 2: H_q and H_kv are passed separately so the reader can
    // broadcast KV heads when H_kv < H_q (GQA / MQA). H_q indexes the
    // query / mask layouts; H_kv indexes the K/V tile layouts. The
    // contract `H_q % H_kv == 0` is enforced by host-side validate().
    //
    // Refinement 3: HAS_MASK now also covers the synthetic-alignment
    // case (S_kv non-aligned). HAS_USER_MASK gates the NoC read of the
    // user mask tensor — when HAS_MASK && !HAS_USER_MASK, the reader
    // builds the mask tile entirely in L1. KEYS_IN_LAST_TILE>0 drives
    // the -inf overlay on the last K iter. MASK_ELEM_BYTES selects the
    // overlay's bit-pattern (2 → bf16 -inf, 4 → fp32 -inf).
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Qt = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(4);
    constexpr uint32_t Dt = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_MASK = get_compile_time_arg_val(6);
    constexpr uint32_t MASK_PER_HEAD = get_compile_time_arg_val(7);
    constexpr uint32_t HAS_USER_MASK = get_compile_time_arg_val(8);
    constexpr uint32_t KEYS_IN_LAST_TILE = get_compile_time_arg_val(9);
    constexpr uint32_t MASK_ELEM_BYTES = get_compile_time_arg_val(10);
    constexpr uint32_t NEEDS_ALIGNMENT_MASK = (KEYS_IN_LAST_TILE != 0u) ? 1u : 0u;

    // KV-head broadcast: every `kv_group_size` consecutive Q heads share
    // one K/V head. Constexpr so the divide collapses to a shift / immediate
    // multiply at compile time when the ratio is a power of two (the common
    // case — MQA: group = H_q, GQA: group ∈ {2, 3, 4, 8, ...}).
    constexpr uint32_t kv_group_size = H_q / H_kv;

    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    // --- Runtime args ------------------------------------------------------
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t mask_addr = get_arg_val<uint32_t>(3);
    uint32_t num_rows = get_arg_val<uint32_t>(4);
    uint32_t start_row = get_arg_val<uint32_t>(5);

    // --- CB indices --------------------------------------------------------
    constexpr uint32_t cb_query = 0;
    constexpr uint32_t cb_key = 1;
    constexpr uint32_t cb_value = 2;
    constexpr uint32_t cb_attn_mask = 3;
    constexpr uint32_t cb_reduction_scaler = 5;

    // --- One-shot scaler tile for the row-MAX reduce -----------------------
    // PoolType::MAX + REDUCE_ROW selects row-axis fill, which is the layout
    // the reduce LLK path expects. value = 1.0 (default factor).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduction_scaler,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW>();

    // --- Tensor accessors --------------------------------------------------
    const uint32_t tile_bytes = get_tile_size(cb_query);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);

    [[maybe_unused]] uint32_t mask_tile_bytes = 0;
    if constexpr (HAS_MASK) {
        mask_tile_bytes = get_tile_size(cb_attn_mask);
    }

    // Mask stride math: per-head mask has stride Qt*Kt per Q head;
    // broadcast mask (shape (B,1,Sq,Skv)) has H_stride = 0 and
    // B_stride = Qt*Kt. The mask is Q-head-indexed in both flavors
    // (per-head: one mask per Q head; broadcast: same mask for every
    // Q head) — independent of GQA's KV broadcast.
    constexpr uint32_t mask_h_stride = (HAS_MASK && MASK_PER_HEAD) ? (Qt * Kt) : 0u;
    constexpr uint32_t mask_b_stride = (HAS_MASK && MASK_PER_HEAD) ? (H_q * Qt * Kt) : (Qt * Kt);

    // --- Main work loop ----------------------------------------------------
    const uint32_t end_row = start_row + num_rows;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Decode r -> (b, h_q, qt). Query/output layout is (B, H_q, Qt).
        const uint32_t b = r / (H_q * Qt);
        const uint32_t h_q = (r / Qt) % H_q;
        const uint32_t qt = r % Qt;

        // KV head for this Q head (Refinement 2). With kv_group_size =
        // H_q / H_kv:
        //   mha (H_kv == H_q) → group=1 → h_kv = h_q
        //   gqa (1 < H_kv < H_q) → group = H_q/H_kv → h_kv = h_q / group
        //   mqa (H_kv == 1)     → group = H_q  → h_kv = 0 for every Q head
        const uint32_t h_kv = h_q / kv_group_size;

        // ----- Push Q row (Dt tiles) -----
        // Q tile index range: [((b*H_q + h_q)*Qt + qt) * Dt, ... + Dt).
        // Layout uses H_q (NOT H_kv) — query has the full Q-head set.
        const uint32_t q_base = ((b * H_q + h_q) * Qt + qt) * Dt;
        cb_reserve_back(cb_query, Dt);
        {
            uint32_t l1_addr = get_write_ptr(cb_query);
            for (uint32_t d = 0; d < Dt; ++d) {
                noc_async_read_tile(q_base + d, q_acc, l1_addr);
                l1_addr += tile_bytes;
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_query, Dt);

        // KV-row base: K and V tile rows for this (b, h_kv). The KV cache
        // is laid out (B, H_kv, S_kv, D) — H_kv heads, not H_q. The
        // h_q→h_kv divide above is what implements the MQA / GQA
        // broadcast inside the address math (no tensor replication
        // needed; multiple Q heads' iterations hit the same KV tile).
        const uint32_t kv_base = (b * H_kv + h_kv) * Kt;

        // Mask offset (only used when HAS_MASK). Mask is Q-head-indexed
        // when MASK_PER_HEAD (one (S_q, S_kv) plane per Q head), so we
        // use h_q here. Broadcast mask has mask_h_stride=0 so h_q is
        // ignored — same code path covers both.
        const uint32_t mask_base = (b * mask_b_stride) + (h_q * mask_h_stride) + (qt * Kt);

        // ----- Pass 1 K-loop: stream K + mask only (V is NOT consumed
        //       by the compute kernel in pass 1) -----
        for (uint32_t k_i = 0; k_i < Kt; ++k_i) {
            // K row: Dt tiles at index (kv_base + k_i) * Dt.
            const uint32_t k_row_base = (kv_base + k_i) * Dt;
            cb_reserve_back(cb_key, Dt);
            {
                uint32_t l1_addr = get_write_ptr(cb_key);
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc_async_read_tile(k_row_base + d, k_acc, l1_addr);
                    l1_addr += tile_bytes;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_key, Dt);

            if constexpr (HAS_MASK) {
                cb_reserve_back(cb_attn_mask, 1);
                uint32_t l1_addr = get_write_ptr(cb_attn_mask);
                if constexpr (HAS_USER_MASK) {
                    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_tile_bytes);
                    noc_async_read_tile(mask_base + k_i, mask_acc, l1_addr);
                    noc_async_read_barrier();
                } else {
                    zero_fill_tile(l1_addr, mask_tile_bytes);
                }
                if constexpr (NEEDS_ALIGNMENT_MASK) {
                    if (k_i == Kt - 1u) {
                        overlay_neg_inf_on_padded_cols<MASK_ELEM_BYTES>(l1_addr, KEYS_IN_LAST_TILE);
                    }
                }
                cb_push_back(cb_attn_mask, 1);
            }
        }

        // ----- Pass 2 K-loop: re-stream K, plus V and mask -----
        // V is only needed in pass 2 (compute's S@V matmul) so this is
        // the FIRST push of V for this query row. K and mask are
        // pushed AGAIN (re-read from DRAM); compute's pass-2 matmul
        // recomputes scores against the fixed global_max from pass 1.
        for (uint32_t k_i = 0; k_i < Kt; ++k_i) {
            // K row (re-read).
            const uint32_t k_row_base = (kv_base + k_i) * Dt;
            cb_reserve_back(cb_key, Dt);
            {
                uint32_t l1_addr = get_write_ptr(cb_key);
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc_async_read_tile(k_row_base + d, k_acc, l1_addr);
                    l1_addr += tile_bytes;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_key, Dt);

            if constexpr (HAS_MASK) {
                cb_reserve_back(cb_attn_mask, 1);
                uint32_t l1_addr = get_write_ptr(cb_attn_mask);
                if constexpr (HAS_USER_MASK) {
                    const auto mask_acc = TensorAccessor(mask_args, mask_addr, mask_tile_bytes);
                    noc_async_read_tile(mask_base + k_i, mask_acc, l1_addr);
                    noc_async_read_barrier();
                } else {
                    zero_fill_tile(l1_addr, mask_tile_bytes);
                }
                if constexpr (NEEDS_ALIGNMENT_MASK) {
                    if (k_i == Kt - 1u) {
                        overlay_neg_inf_on_padded_cols<MASK_ELEM_BYTES>(l1_addr, KEYS_IN_LAST_TILE);
                    }
                }
                cb_push_back(cb_attn_mask, 1);
            }

            // V row: Dt tiles at index (kv_base + k_i) * Dt.
            const uint32_t v_row_base = (kv_base + k_i) * Dt;
            cb_reserve_back(cb_value, Dt);
            {
                uint32_t l1_addr = get_write_ptr(cb_value);
                for (uint32_t d = 0; d < Dt; ++d) {
                    noc_async_read_tile(v_row_base + d, v_acc, l1_addr);
                    l1_addr += tile_bytes;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_value, Dt);
        }
    }
}
