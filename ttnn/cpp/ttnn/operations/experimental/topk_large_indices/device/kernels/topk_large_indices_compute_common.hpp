// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute helpers for every role of the unified compute kernel
// (compute.cpp: row-parallel default, TOPK_TREE node, TOPK_TREE_ROOT root).

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// topk_large_indices keeps final values and indices in the TopK XL LLK DST
// layout until the last rank-order materialization step. The generic compute
// APIs (`isneginf_tile` + `where_tile`) operate on normal tile layouts and do
// not line up with this intermediate value/index pairing, so they only replaced
// a subset of the final -inf lanes during validation.
//
// Keep this as op-local SFPU functionality for now instead of exporting a
// public LLK API: it is tied to the final TopK XL LLK DST contract below,
// where the value words start at the normal `idst` base and the UINT32 index
// words start at `indices_offset`. The helper walks that layout directly,
// compares final values against exact BF16 -inf stored in the FP32 DST
// container (`0xFF800000`), and conditionally writes the sentinel index
// `0xFFFFFFFF`.
inline void _topk_large_indices_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

template <uint32_t K>
inline void _topk_large_indices_mark_neginf_indices_() {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;
    constexpr uint32_t indices_offset = tiles_per_sequence * 64;
    constexpr uint32_t iterations = (K == 512 ? 1 : K == 1024 ? 2 : 4) * 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
#ifdef TOPK_XL_STABLE_TIES
        // The value words carry stale rank stamps in lo16 after the last
        // stamped merge/rebuild; the exact-(-inf) compare needs the bare value.
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0);
#endif
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace topk_large_indices {

constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;

// Transposes the UINT32 index tiles of the final DST sequence into the
// rank-order layout that pack_untilize + the writer's face reorder turn into
// row-major output.
template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t idst, uint32_t indices_cb) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;

    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
        transpose_dest<true, false>(idst + tiles_per_sequence + t);
    }
}

// Transposes the FP32 value tiles of the final DST sequence into the same
// rank-order layout as the indices, for the optional values output.
// The value words are moved with the 32-bit (INT-mode) transpose so the moves
// are bit-exact; the fp32->bf16 conversion happens later in the packer.
// PRECONDITION: materialize_index_rank_order already ran this row (it programs
// the shared 32-bit transpose_dest init, which this reuses), and
// mark_neginf_indices ran before EITHER transpose (it reads the value words in
// the pre-transpose engine layout).
template <uint32_t K>
FORCE_INLINE void materialize_values_rank_order(uint32_t idst) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    for (uint32_t t = 0; t < tiles_per_sequence; ++t) {
        transpose_dest<true, false>(idst + t);
    }
}

// Replaces the index of every exact BF16 -inf value lane with the 0xFFFFFFFF
// sentinel. Must run after the last merge/rebuild, before materialization.
template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t idst) {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, idst, VectorMode::RC_custom)));
}

// Sequential-ties rank stamp of a single sorted unfused run at `idst`. Must
// run before any unfused rebuild that is NOT preceded by a merge (the merge
// stamps internally): a rebuild of tied [bf16|0] keys scrambles the tie order
// the fused sort established. Requires topk_xl_init<K, false> first.
template <uint32_t K>
FORCE_INLINE void topk_xl_stamp_seq_ranks(uint32_t idst) {
#ifdef TOPK_XL_STABLE_TIES
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_xl_stamp_seq_ranks_<K, false>, idst, VectorMode::RC_custom)));
#else
    (void)idst;
#endif
}

// First half of chunk processing: pull the chunk from the input CB into DST.
// Always runs -- the chunk must be resident in DST to be inspected at all.
template <uint32_t K>
FORCE_INLINE void copy_chunk(CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb_id = input_cb.get_cb_id();
    input_cb.wait_front(tiles_per_sequence);
    topk_xl_copy_tile_init(input_cb_id);
    topk_xl_copy_tile<K>(input_cb_id, dst_base, 0, active_elements);
    input_cb.pop_front(tiles_per_sequence);
}

// Classic (unfused) second half: stamp the fused LSB indices, locally sort
// (fused), then split into unfused FP32 values + row-major UINT32 global
// indices and advance the chunk base.
template <uint32_t K>
FORCE_INLINE void finish_chunk_classic(uint32_t dst_base, bool ascending) {
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst_base);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst_base);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// Full classic chunk step. Leaves the sequence at
// [dst_base .. dst_base + tiles) values and
// [dst_base + tiles .. dst_base + 2*tiles) indices, sorted in `ascending`
// direction in the TopK XL engine layout.
template <uint32_t K>
FORCE_INLINE void process_chunk(CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements, bool ascending) {
    copy_chunk<K>(input_cb, dst_base, active_elements);
    finish_chunk_classic<K>(dst_base, ascending);
}

}  // namespace topk_large_indices
