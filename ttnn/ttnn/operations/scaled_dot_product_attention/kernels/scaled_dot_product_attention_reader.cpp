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

// R4 (causal): additive mask sentinel — a LARGE FINITE negative, NOT true −∞. The
// reference (and production) mask with `-1e9`, not −∞: exp(score + (-1e9) − max)
// underflows to 0 just like −∞, but a *finite* value avoids the −∞ − (−∞) = NaN and
// the bf8b corruption a true −∞ tile inflicts on the additive-mask path. (Empirically:
// bf8b + true-−∞ causal missed tolerance catastrophically — the last valid row of a
// tile adjacent to a fully-masked tile-col — while bf8b + custom, whose streamed mask
// converts −∞ to a large finite value, passed. bf16/fp32 pass either way; −1e9 keeps
// them exact.) bf16(-1e9) ≈ 0xCE6E; packed pair = 0xCE6ECE6E.
constexpr uint32_t NEG_MASK_HALF = 0xCE6Eu;
constexpr uint32_t NEG_MASK_PAIR = 0xCE6ECE6Eu;

FORCE_INLINE void fill_zeros_tile(uint32_t wptr, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0u;
    }
}

// R4 (causal): fully-masked tile — every element the large-negative sentinel (a KV
// tile strictly above the causal diagonal relative to the Q tile; every key is in the
// future of every query).
FORCE_INLINE void fill_neg_mask_tile(uint32_t wptr, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = NEG_MASK_PAIR;
    }
}

// R4 (causal): diagonal tile — element (r, c) is the large-negative sentinel iff key
// column c is in the future of query row r (c > r), else 0. This is the strict
// upper-triangular additive causal bias for a tile sitting ON the diagonal (global
// query tile-row == global key tile-col). Face-aware bf16 layout (4 faces of 16x16,
// row-major, two bf16 packed per uint32: low16 = even col, high16 = odd col).
FORCE_INLINE void fill_causal_diag_tile(uint32_t wptr) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
    constexpr uint32_t FACE_H = 16;
    constexpr uint32_t words_per_face_row = 8;                        // 16 cols / 2 per word
    constexpr uint32_t words_per_face = FACE_H * words_per_face_row;  // 128
    const uint32_t face_row0[4] = {0, 0, 16, 16};
    const uint32_t face_col0[4] = {0, 16, 0, 16};
    for (uint32_t f = 0; f < 4; ++f) {
        const uint32_t base = f * words_per_face;
        const uint32_t frow = face_row0[f];
        const uint32_t fcol = face_col0[f];
        for (uint32_t lr = 0; lr < FACE_H; ++lr) {
            const uint32_t r = frow + lr;  // tile-global row 0..31
            const uint32_t row_base = base + lr * words_per_face_row;
            for (uint32_t w = 0; w < words_per_face_row; ++w) {
                const uint32_t c_lo = fcol + 2u * w;  // even column (low16)
                const uint32_t c_hi = c_lo + 1u;      // odd column  (high16)
                const uint32_t lo = (c_lo > r) ? NEG_MASK_HALF : 0x0000u;
                const uint32_t hi = (c_hi > r) ? NEG_MASK_HALF : 0x0000u;
                ptr[row_base + w] = (hi << 16) | lo;
            }
        }
    }
}

// R3 (data-movement): batch a block of async reads then ONE barrier, instead of
// one read + barrier + push per tile (the double_buffer anti-pattern — the KV CBs
// are already KV_DEPTH-deep, so the NoC was left latency-bound). `page_of(t)`
// maps the linear tile index t (0..n-1, in the CB's fill order) to its DRAM page.
//
// `batch` is a COMPILE-TIME predicate that is only true when the per-chunk block
// is exactly one CB slot (no partial chunk on this axis): then every multi-page
// reserve starts slot-aligned in the KV_DEPTH-slot ring and never straddles the
// buffer wrap, so the linear write-pointer walk is contiguous. Partial-chunk
// shapes keep the per-tile path (byte-identical to phase-0; not the perf target).
//
// `ablate` (MEASUREMENT-ONLY, /perf-measure classify-the-bound): when true, SKIP the
// noc_async_read_tile + barrier but KEEP cb_reserve_back/cb_push_back — DRAM bytes
// moved drop to zero while the CB producer/consumer counts (and thus compute) are
// unchanged. If wall-time is flat with reads stubbed, the reads were hidden behind
// compute (compute-bound); if it craters, they were on the critical path (DM-bound).
// Compile-time-elided at its default (ablate=false => byte-identical to shipped). Gated
// by env SDPA_ABLATE_READER in the descriptor; 0 for every shipped build.
template <uint32_t cb, bool batch, bool ablate, typename Acc, typename PageFn>
FORCE_INLINE void read_tiles(uint32_t n, uint32_t tile_bytes, const Acc& acc, PageFn page_of) {
    if constexpr (batch) {
        cb_reserve_back(cb, n);
        if constexpr (!ablate) {
            uint32_t wptr = get_write_ptr(cb);
            for (uint32_t t = 0; t < n; ++t) {
                noc_async_read_tile(page_of(t), acc, wptr);
                wptr += tile_bytes;
            }
            noc_async_read_barrier();  // ONE barrier for n reads -> up to n reads in flight
        }
        cb_push_back(cb, n);
    } else {
        for (uint32_t t = 0; t < n; ++t) {
            cb_reserve_back(cb, 1);
            if constexpr (!ablate) {
                noc_async_read_tile(page_of(t), acc, get_write_ptr(cb));
                noc_async_read_barrier();
            }
            cb_push_back(cb, 1);
        }
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
    constexpr bool has_kv_pad_raw = skv_partial != 0;
    // R4: is_causal generates the triangular −∞ bias on-device. Causal SUBSUMES the
    // R1 KV-padding mask (a padding key at index >= S_kv is always in the future of
    // every valid query, so the causal diagonal already drives it to −∞), so when
    // causal is set the vertical-pad path is disabled to avoid double-generating.
    constexpr uint32_t is_causal_v = get_compile_time_arg_val(14);
    constexpr bool is_causal = is_causal_v != 0;
    constexpr bool has_kv_pad = has_kv_pad_raw && !is_causal;

    // MEASUREMENT-ONLY reader NoC stub (see read_tiles). 0 for every shipped build.
    constexpr uint32_t ablate_reader_v = get_compile_time_arg_val(15);
    constexpr bool ablate_reader = ablate_reader_v != 0;

    // R3 DM-batching knob — RE-MEASURED in R3a, kept PARKED at its trivial (per-tile)
    // default. R3a re-enabled the divisor predicates on top of the compute-side
    // coarsen (chunk 4->8) and re-measured the flagged shape: batched reads 9.05 ms
    // vs the per-tile 9.01 ms — FLAT (within noise). The reads are STILL hidden behind
    // the KV_DEPTH=2 double-buffer (the flagged shape stays compute-bound even after
    // the coarsen — FPU util only 0.07->0.08), so there is no wall-time win to bank
    // (the refinement's "reads stay hidden -> leave the knob parked" branch). Parking
    // also keeps the reader runtime byte-identical to the gate-passing R2/R3b per-tile
    // reader — decisive here because R3a's prefer-divisor `_chunk_size` (see the
    // descriptor) makes the batch predicate true for nearly every supported shape, so
    // enabling batching would widen the blast radius of the (dormant, zero-win, and in
    // R3b intermittently-regression-flagged) bursty-read pattern for no measured gain.
    //
    // The read_tiles<cb,batch> scaffolding stays as a LIVE tunable. To re-enable once
    // a future scheme-change (overlap the softmax vector phases with the matmul) puts
    // the reads on the critical path, restore the straddle-safe divisor predicates:
    //   batch_q  = (Sq_t % Sq_chunk_t) == 0;   batch_kv = (Skv_t % Skv_chunk_t) == 0;
    // (they slot-align every multi-page reserve in the KV_DEPTH-slot ring), then re-run
    // the full golden suite under --dev to confirm no intermittent hang before banking.
    // R3 batching PARKED at per-tile default. Re-enabling it (straddle-safe divisor predicates)
    // was MEASURED a REGRESSION now that the op is read-bound: flagged e2e 5.275 -> 7.352 ms
    // (+39%), reads-visible 7.18% -> 33.4%. Batching front-loads a whole-KV-chunk read burst per
    // chunk that congests the NoC across 110 cores and overlaps WORSE with compute than the
    // per-tile trickle under the KV_DEPTH=2 double-buffer. => bandwidth/congestion-bound, not
    // latency-bound; batching is the wrong lever. Reduce BYTES (bf8 K/V) or redundant re-reads
    // (mcast) instead. Kept parked.
    constexpr bool batch_q = false;
    constexpr bool batch_kv = false;
    constexpr bool batch_mask = batch_q && batch_kv;

    constexpr auto q_args = TensorAccessorArgs<16>();
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

        // R4 causal block-skip: process only the KV chunks that are NOT fully in the
        // future of this Q chunk. A chunk starting at key tile-row skv_off is fully
        // masked once skv_off >= sq_off + sq_valid (every key beyond the Q chunk's
        // last query). skv_off increases with j, so capping the loop at the first
        // such chunk skips all later ones — roughly halving the KV work for causal
        // self-attention (and eliding their K/V DRAM reads entirely).
        uint32_t n_kv_active = n_kv_chunks;
        if constexpr (is_causal) {
            const uint32_t kv_limit = sq_off + sq_valid;  // tiles
            n_kv_active = (kv_limit + Skv_chunk_t - 1) / Skv_chunk_t;
            if (n_kv_active > n_kv_chunks) {
                n_kv_active = n_kv_chunks;
            }
        }

        // Q chunk: (sq_valid x Dt) tiles, row-major (sq, d).
        const uint32_t q_base = (b * H + h) * Sq_t;
        read_tiles<cb_q_in, batch_q, ablate_reader>(sq_valid * Dt, tile_bytes, q_acc, [&](uint32_t t) {
            const uint32_t sq_g = sq_off + (t / Dt);
            const uint32_t d = t % Dt;
            return (q_base + sq_g) * Dt + d;
        });

        const uint32_t kv_base = (b * H_kv + kv_head) * Skv_t;
        const uint32_t mask_base = (b * mask_H + mask_head) * Sq_t;

        for (uint32_t j = 0; j < n_kv_active; ++j) {
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
            read_tiles<cb_k_in, batch_kv, ablate_reader>(Dt * skv_valid, tile_bytes, k_acc, [&](uint32_t t) {
                const uint32_t d = t / skv_valid;
                const uint32_t skv_g = skv_off + (t % skv_valid);
                return (kv_base + skv_g) * Dt + d;
            });
            // V chunk: (skv_valid x Dt) tiles, row-major (skv, d).
            read_tiles<cb_v_in, batch_kv, ablate_reader>(skv_valid * Dt, tile_bytes, v_acc, [&](uint32_t t) {
                const uint32_t skv_g = skv_off + (t / Dt);
                const uint32_t d = t % Dt;
                return (kv_base + skv_g) * Dt + d;
            });
            // mask chunk: (sq_valid x skv_valid) tiles, row-major (sq, skv).
            if constexpr (has_mask) {
                read_tiles<cb_mask_in, batch_mask, ablate_reader>(
                    sq_valid * skv_valid, tile_bytes, mask_acc, [&](uint32_t t) {
                        const uint32_t sq_g = sq_off + (t / skv_valid);
                        const uint32_t skv_g = skv_off + (t % skv_valid);
                        return (mask_base + sq_g) * Skv_t + skv_g;
                    });
            }

            // R4 causal: on a straddling KV chunk (some key tile-col >= some query
            // tile-row — i.e. skv_off + skv_valid > sq_off), generate a score-block-
            // shaped triangular additive mask into cb_kv_mask. Per tile, compare the
            // global query tile-row (sq_off+si) to the global key tile-col (skv_off+sj):
            // below-diagonal tiles are unmasked (0), above-diagonal tiles fully masked
            // (−∞), the on-diagonal tile is triangular (c > r → −∞). Compute adds this
            // before the row-max via the same additive-mask phase. Fully-past chunks
            // (skv_off + skv_valid <= sq_off) are all-below-diagonal, so no mask is
            // generated or consumed — keeping the CB producer/consumer counts matched
            // with compute's identical predicate.
            if constexpr (is_causal) {
                if (skv_off + skv_valid > sq_off) {
                    // Fill word-count MUST follow cb_kv_mask's OWN tile size (always bf16 =
                    // 512 words), NOT the input tile_bytes: for bf8b inputs tile_bytes is
                    // smaller (1088B/272w) than the bf16 mask tile (2048B/512w), so sizing
                    // the count-parameterized fills from tile_bytes under-fills each bf16
                    // mask tile and leaves stale L1 in its tail rows (leaking attention
                    // across masked columns); for fp32 it over-fills past the tile (OOB).
                    const uint32_t mask_words = get_tile_size(cb_kv_mask) / 4;
                    for (uint32_t si = 0; si < sq_valid; ++si) {
                        const uint32_t q_tile = sq_off + si;  // global query tile-row
                        for (uint32_t sj = 0; sj < skv_valid; ++sj) {
                            const uint32_t k_tile = skv_off + sj;  // global key tile-col
                            cb_reserve_back(cb_kv_mask, 1);
                            const uint32_t wptr = get_write_ptr(cb_kv_mask);
                            if (q_tile > k_tile) {
                                fill_zeros_tile(wptr, mask_words);  // below diagonal: unmasked
                            } else if (q_tile < k_tile) {
                                fill_neg_mask_tile(wptr, mask_words);  // above diagonal: masked
                            } else {
                                fill_causal_diag_tile(wptr);  // on diagonal: triangular c>r
                            }
                            cb_push_back(cb_kv_mask, 1);
                        }
                    }
                }
            } else if constexpr (has_kv_pad) {
                // KV-padding softmax mask (h_non_aligned): last KV chunk only. Build a
                // score-block-shaped additive mask — the last S_kv-column tile of each
                // Q row carries the vertical -inf mask, all other tiles are zero — so
                // compute drives the last KV tile's padding columns to -inf before the
                // row-max/exp/row-sum. The boundary tile is the last VALID tile of the
                // (possibly partial R1b) last chunk, at local index skv_valid - 1.
                if (j == n_kv_chunks - 1) {
                    // Size from cb_kv_mask (bf16, 512 words), not tile_bytes — see the
                    // causal branch above. (R1's supported cells are bf16/fp32 where
                    // tile_bytes >= the bf16 tile, so this was benign before, but fp32
                    // over-filled past the tile; bf8b+non_aligned is an R2 EXCLUSION.)
                    const uint32_t mask_words = get_tile_size(cb_kv_mask) / 4;
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
