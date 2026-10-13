// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"
#include "topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// The first fused chunk in each row runs the full TopK init. Later chunks can
// use this hot-path reinit because copy init only clobbers ADDR_MOD_0/3 and the
// MOP, while index stamping additionally changes ADDR_MOD_6 from +32 to +4.
// Fused sort/merge never consumes ADDR_MOD_0/3/4; ADDR_MOD_1/5 and the other
// TopK state remain live. Restoring +32 and the fused MOP therefore avoids the
// two redundant ADDR_MOD writes in every subsequent chunk's full init.
inline void _topk_large_indices_reinit_fused_after_stamp_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_6);
    topk_mop_config<true>();
}

// The final TopK XL survivor stores FP32 value words followed by UINT32 index
// words. Mark indices paired with exact -inf values as invalid without
// disturbing the engine-specific survivor layout.
inline void _topk_large_indices_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

// Replay: FP32 keys record four iterations into the replay buffer and replay
// the rest, so the math RISC issues a few instructions instead of 5 per
// iteration. Same instructions in the same order as the BF16 loop.
template <uint32_t K, bool Replay = false>
inline void _topk_large_indices_mark_neginf_indices_() {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;
    constexpr uint32_t indices_offset = tiles_per_sequence * 64;
    constexpr uint32_t iterations = (K == 512 ? 1 : K == 1024 ? 2 : 4) * 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    if constexpr (Replay) {
        constexpr uint32_t group = 4;
        static_assert(iterations % group == 0, "iterations must be a multiple of the recorded group");
        lltt::record<lltt::Exec>(0, 5 * group);
#pragma GCC unroll 4
        for (uint32_t i = 0; i < group; ++i) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
            TTI_SFPENCC(0, 0, 0, 0);
        }
        for (uint32_t i = group; i < iterations; i += group) {
            lltt::replay(0, 5 * group);
        }
    } else {
        for (uint32_t i = 0; i < iterations; ++i) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
            TTI_SFPENCC(0, 0, 0, 0);
        }
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;

using ttnn::operations::experimental::topk_large_indices::program::ComputeBodyMode;
using ttnn::operations::experimental::topk_large_indices::program::Fp32Key;

// FP32 input: a segment holds 2^chunk_bits chunks, so every fused word keeps
// 12 - chunk_bits key mantissa bits. chunk_bits is the smallest width that
// fits the row in one segment, capped at max_chunk_bits.
template <uint32_t max_chunk_bits>
FORCE_INLINE uint32_t fp32_chunk_bits(uint32_t num_chunks) {
    uint32_t bits = 0;
    while (bits < max_chunk_bits && (1u << bits) < num_chunks) {
        ++bits;
    }
    return bits;
}

// FP32 split: a chunk_bits-wide chunk field and value words that keep every key
// bit above it, so unfused cross-segment merges rank at the key's precision.
// Requires topk_xl_separate_indices_row_major_global_init().
// LiveShift: requires the live-shift split init instead, and keeps the row's -T in LREG13.
template <uint32_t K, bool LiveShift = false>
FORCE_INLINE void split_fp32_key(uint32_t dst, uint32_t seg_base, uint32_t chunk_bits) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_masked_<K, LiveShift>,
        dst,
        VectorMode::RC_custom,
        seg_base,
        chunk_bits)));
}

// FP32 bits of -0.0: a shift that leaves every value unchanged.
constexpr uint32_t no_shift = 0x80000000u;

// Shifted rows of >= 2 windows (FusedSegmented, K >= 1024) take T from their own first window: it is keyed
// on trunc(x - S) (S: preshift_first_chunk) and sorted, the word at stamp position target_pos (the rank
// matching k / search length) gives d, T = S + d, LREG13 = -T, and the window is re-keyed in place to x - T.
// Later stamps add the live LREG13.
template <uint32_t K>
FORCE_INLINE void shift_first_chunk(uint32_t dst, uint32_t target_pos, uint32_t chunk_bits) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_load_row_shift_preshifted_<K>, dst, VectorMode::RC_custom, target_pos)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_shift_sorted_chunk_<K, true>, dst, VectorMode::RC_custom, chunk_bits)));
}

// Before T is known the first window keeps only 12 mantissa bits of its key. Keying it on x - S, with S a
// moment estimate of the k-th score (m + z' * mean|x - m| of a strided quarter of the window), makes those
// bits resolve |x - S| instead of |x| near the threshold. Leaves LREG13 = -S for the live stamp.
template <uint32_t K>
FORCE_INLINE void preshift_first_chunk(uint32_t dst, uint32_t dev_scale) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_row_preshift_<K>, dst, VectorMode::RC_custom, dev_scale)));
}

// sqrt(pi / 2) * Phi^-1(1 - q) in Q12 (mean absolute deviation -> standard normal quantile): at q = i / 64
// (clamped to [1/128, 127/128]), and per octave q in [2^p, 2^(p+1)) / 65536 below 1/64.
constexpr int16_t preshift_z_q12[65] = {
    12411, 11057, 9562,  8604,  7876,  7278,  6766,  6314,  5905,  5532,  5185,  4860,  4554,
    4263,  3986,  3719,  3463,  3214,  2973,  2738,  2509,  2285,  2065,  1849,  1636,  1426,
    1218,  1012,  808,   605,   403,   201,   0,     -201,  -403,  -605,  -808,  -1012, -1218,
    -1426, -1636, -1849, -2065, -2285, -2509, -2738, -2973, -3214, -3463, -3719, -3986, -4263,
    -4554, -4860, -5185, -5532, -5905, -6314, -6766, -7278, -7876, -8604, -9562, -11057, -12411};
constexpr int16_t preshift_z_tail_q12[10] = {20996, 20155, 19282, 18372, 17420, 16420, 15365, 14244, 13045, 11749};
constexpr int32_t preshift_z_below_tail_q12 = 22202;

FORCE_INLINE uint32_t msb_index(uint32_t v) {
    uint32_t e = 0;
    while (v >>= 1) {
        ++e;
    }
    return e;
}

// FP32 bits of z' / (K / 4) for q = user_k / searched: the scale of `_topk_xl_row_preshift_`'s deviation sum.
template <uint32_t K>
FORCE_INLINE uint32_t preshift_dev_scale(uint32_t user_k, uint32_t searched) {
    uint32_t q16 = searched == 0 ? 65536u : (user_k << 16) / searched;
    q16 = q16 > 65536u ? 65536u : q16;
    int32_t z;
    if (q16 >= 1024) {
        const uint32_t idx = q16 >> 10;
        const int32_t lo = preshift_z_q12[idx];
        const int32_t hi = idx < 64 ? preshift_z_q12[idx + 1] : lo;
        z = lo + (((hi - lo) * static_cast<int32_t>(q16 & 1023)) >> 10);
    } else if (q16 == 0) {
        z = preshift_z_below_tail_q12;
    } else {
        z = preshift_z_tail_q12[msb_index(q16)];
    }
    if (z == 0) {
        return 0;
    }
    const uint32_t sign = z < 0 ? 0x80000000u : 0u;
    const uint32_t a = static_cast<uint32_t>(z < 0 ? -z : z);
    const uint32_t e = msb_index(a);
    const uint32_t mant = e >= 23 ? a >> (e - 23) : a << (23 - e);
    // Q12 and 1 / (K / 4).
    constexpr int32_t scale_log2 = K == 2048 ? -21 : -20;
    return sign | (static_cast<uint32_t>(static_cast<int32_t>(e) + scale_log2 + 127) << 23) | (mant & 0x7FFFFFu);
}

// Shifted rows refine T once, after this many leading chunks are merged into the survivor, if more follow.
constexpr uint32_t refine_after_chunks = 4;
// Until the refinement the survivor holds only chunks 0..refine_after_chunks - 1, so rows with a wider field
// key them on this narrower chunk field (finer keys and a finer refinement read); the refinement re-key moves
// them to the row's field.
constexpr uint32_t early_chunk_bits = 2;
static_assert(refine_after_chunks <= (1u << early_chunk_bits), "early chunks must fit the early chunk field");
// A core with fewer rows than the busiest core (at most 7/8 of its load) has idle time at the end of the
// kernel. It keys shifted K = 2048 rows of > 4 windows on this chunk field instead (segments of 4 windows:
// one or two more key bits than the row's cheapest field). Each extra segment costs ~2.6 us, ~7 % of the
// row's time per row in all, well inside the core's >= 1/8 idle share. The refinement stays in segment 0.
constexpr uint32_t spare_chunk_bits = 2;
static_assert(refine_after_chunks <= (1u << spare_chunk_bits), "the refinement must fit in segment 0");

// Rank r of a descending local sort or rebuilt survivor -> its stamp position. The stamp numbers a lane
// (tile row, tile in sequence, tile column) as row | tile << 5 | column << 6. Sort position r is a bit
// permutation of that lane: from the most significant field down, column bits [3:0], tile (K=2048),
// row bit 4, column bit 4, row bits [3:0].
template <uint32_t K>
FORCE_INLINE uint32_t rank_stamp_position(uint32_t rank) {
    constexpr uint32_t col_lo_shift = K == 2048 ? 7 : 6;
    const uint32_t row = (rank & 15) | (((rank >> 5) & 1) << 4);
    const uint32_t col = ((rank >> col_lo_shift) & 15) | (((rank >> 4) & 1) << 4);
    const uint32_t tile = K == 2048 ? (rank >> 6) & 1 : 0;
    return row | (tile << 5) | (col << 6);
}

// Stamp position of the rank matching k / searched in the top-K of `merged` leading values, or K if the
// top-K does not reach that rank.
template <uint32_t K>
FORCE_INLINE uint32_t quantile_stamp_position(uint32_t user_k, uint32_t merged, uint32_t searched) {
    uint32_t rank = (user_k * merged + searched / 2) / searched;
    rank = rank > 0 ? rank - 1 : 0;
    return rank < K ? rank_stamp_position<K>(rank) : K;
}

template <uint32_t K>
FORCE_INLINE void copy_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb = input.get_cb_id();

    input.wait_front(tiles_per_sequence);
    topk_xl_copy_tile_init(input_cb);
    topk_xl_copy_tile<K>(input_cb, dst, 0, active_elements);
    input.pop_front(tiles_per_sequence);
}

template <uint32_t K, Fp32Key Key = Fp32Key::None>
FORCE_INLINE void sort_classic_chunk(
    CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending, uint32_t neg_shift = 0) {
    copy_chunk<K>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    if constexpr (Key == Fp32Key::Shifted) {
        // Same column-major stamp as core_id 0, on the 16-bit key of x - T. The
        // per-chunk split below still reads its 0xF / 0x1 constants from LREG13/14.
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, true>,
            dst,
            VectorMode::RC_custom,
            0,
            5,
            neg_shift)));
    } else {
        topk_xl_add_lsb_indices<K, 0>(dst);
    }
    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// FullInit is compile-time and this helper is force-inlined so the hot loop has
// neither a mode branch nor a repeated full TopK configuration sequence.
// Shifted keys stamp a chunk_bits-wide chunk field over trunc(x - T) (see
// _topk_xl_add_lsb_indices_rt_shifted_). BF16 and TF32 words have zero low
// bits under the index, so they keep the plain stamp.
template <uint32_t K, bool FullInit, Fp32Key Key = Fp32Key::None>
FORCE_INLINE void sort_fused_chunk(
    CircularBuffer& input,
    uint32_t dst,
    uint32_t active_elements,
    bool ascending,
    uint32_t local_chunk_id,
    uint32_t chunk_bits = 5,
    uint32_t neg_shift = no_shift,
    bool live_shift = false) {
    copy_chunk<K>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    if constexpr (Key == Fp32Key::Shifted) {
        if (live_shift) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, false, true>,
                dst,
                VectorMode::RC_custom,
                local_chunk_id,
                chunk_bits,
                0)));
        } else {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K>,
                dst,
                VectorMode::RC_custom,
                local_chunk_id,
                chunk_bits,
                neg_shift)));
        }
    } else {
        topk_xl_add_lsb_indices_rt<K>(dst, local_chunk_id);
    }
    if constexpr (FullInit) {
        topk_xl_init<K, true>();
    } else {
        MATH((ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_()));
    }
    topk_xl_local_sort<K>(dst, ascending);
}

// Shifted rows (full FP32 unpacked straight to DEST) with K = 2048: in segment 0, chunk c + 1 is unpacked into
// a free DEST staging slot while the SFPU stamps and merges chunk c, instead of serially after the rebuild (the
// stock copy stalls math until each tile lands). Segment 0 leaves tiles 4..7 free; later segments use them.
// Tile 0 of the next chunk is unpacked during the stamp, tile 1 during the merge: both are SFPU-only, while
// sort and rebuild move faces through SrcA/SrcB. The UNPACK thread's half of each step sits at the same point
// in program order as the math half, after the local sort's dummy SrcB valid, so neither thread waits on the other.
constexpr uint32_t staging_slot(uint32_t chunk) { return chunk & 1 ? 6 : 4; }

FORCE_INLINE void prefetch_tile(CircularBuffer& input, uint32_t staging, uint32_t tile) {
    const uint32_t input_cb = input.get_cb_id();
    if (tile == 0) {
        input.wait_front(2);
        UNPACK((llk_unpack_topk_xl_copy_init(input_cb)));
    }
    UNPACK((llk_unpack_topk_xl_copy_one_tile_unpack(input_cb, tile, elements_per_tile)));
    if (tile == 1) {
        UNPACK(TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0));
        input.pop_front(2);
    }
    MATH((llk_math_topk_xl_copy_to_dest_issue(staging + tile)));
}

FORCE_INLINE void prefetch_wait(CircularBuffer& input, uint32_t staging, uint32_t tile) {
    MATH((llk_math_topk_xl_copy_to_dest_complete(input.get_cb_id(), staging + tile)));
}

// Stamp (live shift) a chunk that was unpacked to `staging` into the incoming slot, unpacking tile 0 of the
// next chunk to `next_staging` meanwhile (0: none), then sort it ascending.
template <uint32_t K>
FORCE_INLINE void sort_staged_chunk(
    CircularBuffer& input, uint32_t dst, uint32_t staging, uint32_t local_chunk_id, uint32_t chunk_bits, uint32_t next_staging) {
    if (next_staging != 0) {
        prefetch_tile(input, next_staging, 0);
    }
    topk_xl_add_lsb_indices_init();
    // A DEST tile is 64 rows; the staging slot is 2 or 4 tiles past the incoming slot.
    if (staging - dst == 2) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, false, true, 2 * 64>,
            dst,
            VectorMode::RC_custom,
            local_chunk_id,
            chunk_bits,
            0)));
    } else {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, false, true, 4 * 64>,
            dst,
            VectorMode::RC_custom,
            local_chunk_id,
            chunk_bits,
            0)));
    }
    if (next_staging != 0) {
        prefetch_wait(input, next_staging, 0);
    }
    MATH((ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_()));
    topk_xl_local_sort<K>(dst, true);
}

template <uint32_t K, Fp32Key Key>
FORCE_INLINE void reduce_classic_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, uint32_t neg_shift) {
    // The host routes TF32 keys only to rows of at most 4 windows, which never take
    // this body; if one did, its chunk-0 stamp and per-chunk split read only bits
    // [10:0], which TF32 leaves zero, so it would rank like BF16 at TF32 precision.
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    constexpr uint32_t incoming_slot = 2 * tiles_per_sequence;

    topk_xl_separate_indices_row_major_init_static<0, 0>();
    sort_classic_chunk<K, Key>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, neg_shift);

    if (num_chunks == 1) {
        topk_xl_init<K, false>();
        topk_xl_rebuild<K, false>(survivor_slot, false);
        return;
    }

    for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
        const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
        sort_classic_chunk<K, Key>(input, incoming_slot, active_elements, true, neg_shift);

        topk_xl_init<K, false>();
        topk_xl_merge<K, false>(survivor_slot);
        topk_xl_rebuild<K, false>(survivor_slot, false);
    }
}

template <uint32_t K, Fp32Key Key>
FORCE_INLINE void reduce_fused_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, uint32_t neg_shift) {
    // A row of at most 32 chunks: the narrowest chunk field that numbers them all.
    const uint32_t chunk_bits = Key == Fp32Key::None ? 5 : fp32_chunk_bits<5>(num_chunks);
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor has no separate index tiles. Its merge operand is
    // therefore adjacent at one sequence, rather than two sequences, away.
    constexpr uint32_t incoming_slot = tiles_per_sequence;

    sort_fused_chunk<K, true, Key>(
        input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, 0, chunk_bits, neg_shift);

    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, false);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
            sort_fused_chunk<K, false, Key>(input, incoming_slot, active_elements, true, chunk, chunk_bits, neg_shift);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, false);
        }
    }

    // Decode the five-bit chunk stamp once, after the final fused survivor is
    // known, and leave the standard [values, indices] epilogue layout.
    topk_xl_separate_indices_row_major_global_init();
    if constexpr (Key == Fp32Key::None) {
        topk_xl_separate_indices_row_major_global<K>(survivor_slot);
    } else {
        split_fp32_key<K>(survivor_slot, 0, chunk_bits);
    }
}

// TF32 keys hold 10 mantissa bits in DEST, so their chunk field never needs
// more than 2 bits (the host routes only rows of <= 4 windows here). Shifted
// keys use up to the full 5 bits: no more segments than BF16 rows.
template <uint32_t K, Fp32Key Key>
FORCE_INLINE void reduce_segmented_row(
    CircularBuffer& input,
    uint32_t num_chunks,
    uint32_t tail_elements,
    uint32_t target_pos,
    uint32_t refine_pos,
    uint32_t dev_scale,
    bool allow_prefetch,
    bool spare_time) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    // Shifted K = 2048 rows of 9..32 windows take one chunk-field bit less than one segment needs (two
    // segments: 3 bits for 9..16 windows, 4 for 17..32) for one more key bit.
    const uint32_t chunk_bits = Key == Fp32Key::None                         ? 5
                                : Key == Fp32Key::Tf32                       ? fp32_chunk_bits<2>(num_chunks)
                                : K == 2048 && spare_time && num_chunks > 4      ? spare_chunk_bits
                                : K == 2048 && num_chunks > 8 && num_chunks <= 32 ? fp32_chunk_bits<5>(num_chunks) - 1
                                                                             : fp32_chunk_bits<5>(num_chunks);
    const uint32_t segment_capacity = 1u << chunk_bits;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence;

    const uint32_t num_segments = (num_chunks + segment_capacity - 1) >> chunk_bits;
    // Shifted rows of >= 2 windows key on x - T with T read from the sorted first window; a single window
    // keeps the exact 12-mantissa-bit key (chunk_bits = 0) and needs no shift.
    constexpr bool can_live_shift = Key == Fp32Key::Shifted && K != 512;
    const bool live_shift = can_live_shift && num_chunks >= 2;
    // T from the top-K of the first refine_after_chunks windows (a 4x larger sample than chunk 0), when
    // more chunks follow them. They must lie in segment 0: the refinement re-keys its fused survivor (at
    // its last chunk, before the split, when segment 0 holds exactly refine_after_chunks windows).
    const uint32_t seg0_last_chunk = (num_chunks < segment_capacity ? num_chunks : segment_capacity) - 1;
    const bool refine = live_shift && refine_after_chunks - 1 <= seg0_last_chunk && refine_after_chunks < num_chunks &&
                        refine_pos < K;
    const bool early_field = refine && chunk_bits > early_chunk_bits;
    const uint32_t early_bits = early_field ? early_chunk_bits : chunk_bits;
    // Segment-0 chunks [1, prefetch_end) are unpacked ahead into staging slots; a partial tail chunk keeps the
    // -inf padding copy. A core's first row does not prefetch: the reader holds chunk 1 back until chunk 0
    // is consumed (see reader.cpp), and chunk 1 is not needed before chunk 0 is sorted and re-keyed.
    constexpr bool can_prefetch = can_live_shift && K == 2048;
    uint32_t prefetch_end = 0;
    if constexpr (can_prefetch) {
        if (live_shift && allow_prefetch) {
            const uint32_t seg0_end = segment_capacity < num_chunks ? segment_capacity : num_chunks;
            prefetch_end = seg0_end == num_chunks && tail_elements != K ? seg0_end - 1 : seg0_end;
        }
    }
    for (uint32_t segment = 0; segment < num_segments; ++segment) {
        const uint32_t first_chunk = segment * segment_capacity;
        const uint32_t end_chunk =
            first_chunk + segment_capacity < num_chunks ? first_chunk + segment_capacity : num_chunks;
        const uint32_t last_chunk = end_chunk - 1;

        // Segment zero is already in the final accumulator slot. Later
        // segments use the second unfused sequence. Their fused input chunk
        // is adjacent to the fused survivor at base + tiles_per_sequence.
        const uint32_t base = segment == 0 ? accumulator_slot : segment_slot;
        const uint32_t chunk_slot = base + tiles_per_sequence;
        const bool mirror_final_survivor = segment != 0;

        const uint32_t first_elements = first_chunk + 1 == num_chunks ? tail_elements : K;
        if (segment == 0) {
            if constexpr (can_live_shift) {
                if (live_shift) {
                    // Exact key first (11-bit coordinate field), so T can be read from the sorted run.
                    if constexpr (can_prefetch) {
                        const bool next = 1 < prefetch_end;
                        copy_chunk<K>(input, base, first_elements);
                        topk_xl_add_lsb_indices_init();
                        preshift_first_chunk<K>(base, dev_scale);
                        if (next) {
                            prefetch_tile(input, staging_slot(1), 0);
                        }
                        MATH((_llk_math_eltwise_unary_sfpu_params_(
                            ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, false, true>,
                            base,
                            VectorMode::RC_custom,
                            0,
                            0,
                            0)));
                        if (next) {
                            prefetch_wait(input, staging_slot(1), 0);
                        }
                        topk_xl_init<K, true>();
                        topk_xl_local_sort<K>(base, false);
                        if (next) {
                            prefetch_tile(input, staging_slot(1), 1);
                        }
                        shift_first_chunk<K>(base, target_pos, early_bits);
                        if (next) {
                            prefetch_wait(input, staging_slot(1), 1);
                        }
                    } else {
                        copy_chunk<K>(input, base, first_elements);
                        topk_xl_add_lsb_indices_init();
                        preshift_first_chunk<K>(base, dev_scale);
                        MATH((_llk_math_eltwise_unary_sfpu_params_(
                            ckernel::sfpu::_topk_xl_add_lsb_indices_rt_shifted_<K, false, true>,
                            base,
                            VectorMode::RC_custom,
                            0,
                            0,
                            0)));
                        topk_xl_init<K, true>();
                        topk_xl_local_sort<K>(base, false);
                        shift_first_chunk<K>(base, target_pos, early_bits);
                    }
                } else {
                    sort_fused_chunk<K, true, Key>(input, base, first_elements, false, 0, chunk_bits, no_shift);
                }
            } else {
                sort_fused_chunk<K, true, Key>(input, base, first_elements, false, 0, chunk_bits, no_shift);
            }
        } else {
            sort_fused_chunk<K, false, Key>(input, base, first_elements, false, 0, chunk_bits, no_shift, live_shift);
        }

        if (first_chunk == last_chunk) {
            topk_xl_rebuild<K, true>(base, mirror_final_survivor);
        } else {
            for (uint32_t chunk = first_chunk + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
                // early_field is false outside segment 0's leading chunks.
                const uint32_t stamp_bits = early_field && chunk < refine_after_chunks ? early_bits : chunk_bits;
                // prefetch_end is 0 outside segment 0.
                const bool staged = can_prefetch && chunk < prefetch_end;
                [[maybe_unused]] const bool next = can_prefetch && chunk + 1 < prefetch_end;
                if constexpr (can_prefetch) {
                    if (staged) {
                        sort_staged_chunk<K>(
                            input,
                            chunk_slot,
                            staging_slot(chunk),
                            chunk - first_chunk,
                            stamp_bits,
                            next ? staging_slot(chunk + 1) : 0);
                    }
                }
                if (!staged) {
                    sort_fused_chunk<K, false, Key>(
                        input, chunk_slot, active_elements, true, chunk - first_chunk, stamp_bits, no_shift, live_shift);
                }
                if constexpr (can_prefetch) {
                    if (next) {
                        prefetch_tile(input, staging_slot(chunk + 1), 1);
                    }
                }
                topk_xl_merge<K, true>(base);
                if constexpr (can_prefetch) {
                    if (next) {
                        prefetch_wait(input, staging_slot(chunk + 1), 1);
                    }
                }
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
                if constexpr (can_live_shift) {
                    if (refine && segment == 0 && chunk + 1 == refine_after_chunks) {
                        if (early_field) {
                            MATH((_llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::_topk_xl_refine_row_shift_<K, true>,
                                base,
                                VectorMode::RC_custom,
                                refine_pos,
                                chunk_bits)));
                        } else {
                            MATH((_llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::_topk_xl_refine_row_shift_<K>,
                                base,
                                VectorMode::RC_custom,
                                refine_pos,
                                chunk_bits)));
                        }
                    }
                }
            }
        }

        // Split once per segment. The base is a multiple of segment_capacity*K,
        // so it does not overlap the decoded segment-local index bits.
        if constexpr (can_live_shift) {
            // Keeps LREG13 (-T) for the next segment's stamps.
            MATH((ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_live_shift_()));
            split_fp32_key<K, true>(base, segment * (segment_capacity * K), chunk_bits);
        } else if constexpr (Key != Fp32Key::None) {
            topk_xl_separate_indices_row_major_global_init();
            split_fp32_key<K>(base, segment * (segment_capacity * K), chunk_bits);
        } else {
            topk_xl_separate_indices_row_major_global_init();
            topk_xl_separate_indices_row_major_global_base<K>(base, segment * (segment_capacity * K));
        }

        if (segment != 0) {
            // Operand zero is descending and each later segment was rebuilt
            // ascending, satisfying the unfused bitonic merge precondition.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, false);
        }
    }
}

template <uint32_t K, bool Replay>
FORCE_INLINE void mark_neginf_indices(uint32_t dst) {
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K, Replay>, dst, VectorMode::RC_custom)));
}

template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t dst, uint32_t indices_cb) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence; ++tile) {
        transpose_dest<true, false>(dst + tiles_per_sequence + tile);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t max_rows = get_arg_val<uint32_t>(1);
    const bool spare_time = num_rows * 8 <= max_rows * 7;
    const uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::compute_search_length);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto body_mode = static_cast<ComputeBodyMode>(get_compile_time_arg_val(3));
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(5);
    constexpr auto key = static_cast<Fp32Key>(get_compile_time_arg_val(6));
    constexpr uint32_t user_k = get_compile_time_arg_val(7);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(
        body_mode == ComputeBodyMode::Classic || body_mode == ComputeBodyMode::FusedEndToEnd ||
            body_mode == ComputeBodyMode::FusedSegmented,
        "invalid TopK compute body mode");

    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t final_survivor = 0;

    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);

    uint32_t num_chunks;
    uint32_t tail_elements;
    if constexpr (valid_length_from_metadata) {
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.wait_front(1);
        num_chunks = ckernel::read_tile_value(meta_cb, 0, topk_metadata_num_chunks_word);
        tail_elements = ckernel::read_tile_value(meta_cb, 0, topk_metadata_tail_elements_word);
        meta_cb_obj.pop_front(1);
    } else {
        const auto bounds = calculate_topk_bounds(search_len, K);
        num_chunks = bounds.num_chunks;
        tail_elements = bounds.tail_elements;
    }

    CircularBuffer input(input_cb);
    CircularBuffer indices(indices_cb);

    // Shifted keys: the row's k-th value is estimated by the first window's element at the rank matching
    // k / search length, then refined from the survivor of the first refine_after_chunks windows.
    uint32_t target_pos = 0;
    uint32_t refine_pos = K;
    uint32_t dev_scale = 0;
    if constexpr (key == Fp32Key::Shifted && K != 512) {
        const uint32_t searched = num_chunks == 0 ? 1 : (num_chunks - 1) * K + tail_elements;
        dev_scale = preshift_dev_scale<K>(user_k, searched);
        target_pos = quantile_stamp_position<K>(user_k, K, searched);
        if (target_pos == K) {
            target_pos = rank_stamp_position<K>(K - 1);
        }
        if (num_chunks > refine_after_chunks) {
            refine_pos = quantile_stamp_position<K>(user_k, refine_after_chunks * K, searched);
        }
    }

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if constexpr (body_mode == ComputeBodyMode::FusedSegmented) {
            reduce_segmented_row<K, key>(
                input, num_chunks, tail_elements, target_pos, refine_pos, dev_scale, row != 0, spare_time);
        } else if constexpr (body_mode == ComputeBodyMode::FusedEndToEnd) {
            reduce_fused_row<K, key>(input, num_chunks, tail_elements, no_shift);
        } else {
            reduce_classic_row<K, key>(input, num_chunks, tail_elements, no_shift);
        }

        mark_neginf_indices<K, key != Fp32Key::None>(final_survivor);
        materialize_index_rank_order<K>(final_survivor, indices_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(
            indices_cb, 1, 0, final_survivor + tiles_per_sequence);
        indices.push_back(1);

        tile_regs_release();
    }
}
