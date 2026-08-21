// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified topk_large_indices compute kernel. The factory selects the role per
// CreateKernel site via compile-time defines:
//
//   (no role define) -- ROW-PARALLEL: each core reduces full rows and
//   materializes the index output itself. The row-reduction body is selected
//   host-side (compute_body_mode) and passed as a compile-time arg:
//   Classic (unfused per-chunk split), FusedEndToEnd (stay fused through
//   every merge/rebuild, one global split per row; <= 32 chunks), or
//   FusedSegmented (per-<=32-chunk-segment fusion folded into an unfused
//   cross-segment survivor).
//
//   TOPK_TREE -- COLUMN-PARALLEL tree node (every rectangle core except the
//   root). Phase 1 (leaf): reduce this core's chunk slice to one K survivor in
//   DST. Phase 2 (tree): for each of this core's num_merges winning levels,
//   consume one partner sequence from the recv CB (delivered by this core's
//   writer via the pairwise handshake), copy it to DST operand-1, and
//   merge+rebuild IN PLACE -- the survivor never leaves DST between levels.
//   Phase 3 (ship): pack the survivor raw (FP32 values + UINT32 indices) for
//   the writer to send to this core's winner.
//
//   TOPK_TREE + TOPK_TREE_ROOT -- COLUMN-PARALLEL tree root (rectangle core
//   (0,0), slice 0). Same leaf reduction and in-DST tree merges, but the root
//   never ships: every rebuild stays DESCENDING (it is always the next merge's
//   operand 0), and after the last level it materializes the indices exactly
//   like the row-parallel epilogue.
//
// Tree directions: the bitonic halver needs operand0 descending and operand1
// ascending, so every DST-mutating action rebuilds DESCENDING except a node's
// LAST one before shipping, which rebuilds ASCENDING (validated upstream by
// test_topk_xl_rebuild_ascending: rebuild(asc) after a merge is the exact
// rank-mirror of rebuild(desc)).
//
// Tree empty slice (valid_length cut, nodes only -- slice 0 always holds at
// least one element):
//   * num_merges == 0 -> nothing to compute; the writer ships the prefilled
//     all--inf scratch sequence instead.
//   * num_merges > 0 -> the FIRST partner is ADOPTED (copied to operand-0 and
//     rebuilt) instead of merged; an ascending monotonic run is bitonic, so
//     the rebuild is valid in either direction.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"
#include "topk_large_indices_compute_body_mode.hpp"

#ifdef TOPK_TREE
#include "api/compute/tile_move_copy.h"  // copy_tile of received sequences
#ifndef TOPK_TREE_ROOT
#include "api/compute/pack.h"  // pack_block of the shipped survivor
#endif
#endif

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
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;

using ttnn::operations::experimental::topk_large_indices::program::ComputeBodyMode;

template <uint32_t K>
FORCE_INLINE void copy_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb = input.get_cb_id();

    input.wait_front(tiles_per_sequence);
    topk_xl_copy_tile_init(input_cb);
    topk_xl_copy_tile<K>(input_cb, dst, 0, active_elements);
    input.pop_front(tiles_per_sequence);
}

template <uint32_t K>
FORCE_INLINE void sort_classic_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending) {
    copy_chunk<K>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, 0>(dst);
    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst, ascending);

    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(dst);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// FullInit is compile-time and this helper is force-inlined so the hot loop has
// neither a mode branch nor a repeated full TopK configuration sequence.
template <uint32_t K, bool FullInit>
FORCE_INLINE void sort_fused_chunk(
    CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending, uint32_t local_chunk_id) {
    copy_chunk<K>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst, local_chunk_id);
    if constexpr (FullInit) {
        topk_xl_init<K, true>();
    } else {
        MATH((ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_()));
    }
    topk_xl_local_sort<K>(dst, ascending);
}

template <uint32_t K>
FORCE_INLINE void reduce_classic_row(CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    constexpr uint32_t incoming_slot = 2 * tiles_per_sequence;

    topk_xl_separate_indices_row_major_init_static<0, 0>();
    sort_classic_chunk<K>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false);

    if (num_chunks == 1) {
        topk_xl_init<K, false>();
        topk_xl_rebuild<K, false>(survivor_slot, false);
        return;
    }

    for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
        const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
        sort_classic_chunk<K>(input, incoming_slot, active_elements, true);

        topk_xl_init<K, false>();
        topk_xl_merge<K, false>(survivor_slot);
        topk_xl_rebuild<K, false>(survivor_slot, false);
    }
}

template <uint32_t K>
FORCE_INLINE void reduce_fused_row(CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor has no separate index tiles. Its merge operand is
    // therefore adjacent at one sequence, rather than two sequences, away.
    constexpr uint32_t incoming_slot = tiles_per_sequence;

    sort_fused_chunk<K, true>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, 0);

    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, false);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
            sort_fused_chunk<K, false>(input, incoming_slot, active_elements, true, chunk);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, false);
        }
    }

    // Decode the five-bit chunk stamp once, after the final fused survivor is
    // known, and leave the standard [values, indices] epilogue layout.
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

template <uint32_t K>
FORCE_INLINE void reduce_segmented_row(CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t segment_capacity = 32;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence;

    const uint32_t num_segments = (num_chunks + segment_capacity - 1) / segment_capacity;
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
            sort_fused_chunk<K, true>(input, base, first_elements, false, 0);
        } else {
            sort_fused_chunk<K, false>(input, base, first_elements, false, 0);
        }

        if (first_chunk == last_chunk) {
            topk_xl_rebuild<K, true>(base, mirror_final_survivor);
        } else {
            for (uint32_t chunk = first_chunk + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
                sort_fused_chunk<K, false>(input, chunk_slot, active_elements, true, chunk - first_chunk);
                topk_xl_merge<K, true>(base);
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
            }
        }

        // Split once per segment. The base is a multiple of 32*K, so it does
        // not overlap the decoded segment-local index bits.
        topk_xl_separate_indices_row_major_global_init();
        topk_xl_separate_indices_row_major_global_base<K>(base, segment * (segment_capacity * K));

        if (segment != 0) {
            // Operand zero is descending and each later segment was rebuilt
            // ascending, satisfying the unfused bitonic merge precondition.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, false);
        }
    }
}

template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t dst) {
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, dst, VectorMode::RC_custom)));
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
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);

#if defined(TOPK_TREE_ROOT)
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
#elif defined(TOPK_TREE)
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t ship_values_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ship_indices_cb = get_compile_time_arg_val(2);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
#else
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto body_mode = static_cast<ComputeBodyMode>(get_compile_time_arg_val(3));
    static_assert(
        body_mode == ComputeBodyMode::Classic || body_mode == ComputeBodyMode::FusedEndToEnd ||
            body_mode == ComputeBodyMode::FusedSegmented,
        "invalid TopK compute body mode");
#endif

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t slot0 = 0;

#ifdef TOPK_TREE
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    // ------------------------------------------------------------------
    // COLUMN-PARALLEL tree body (node and root)
    // ------------------------------------------------------------------
    const uint32_t start_chunk = get_arg_val<uint32_t>(3);  // always 0 for slice 0; kept for arg-layout parity
    const uint32_t num_merges = get_arg_val<uint32_t>(4);

#ifdef TOPK_TREE_ROOT
    constexpr bool is_tree_root = true;
#else
    constexpr bool is_tree_root = false;
#endif
    constexpr uint32_t slot1 = sequence_tiles;

    // An empty slice (valid_length cut) with no partners contributes nothing;
    // the writer ships the -inf scratch sequence. The root's slice 0 always
    // contains at least one element (valid_length >= 1), so it never takes
    // the empty/adopt paths and they fold away.
    const bool own_empty = !is_tree_root && (num_chunks == 0);
    if (own_empty && num_merges == 0) {
        return;
    }

#ifdef TOPK_TREE_ROOT
    compute_kernel_hw_startup(input_cb, indices_out_cb);
#else
    compute_kernel_hw_startup(input_cb, ship_values_cb);
#endif

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer recv_obj(recv_cb);
#ifdef TOPK_TREE_ROOT
    CircularBuffer indices_out_obj(indices_out_cb);
#else
    CircularBuffer ship_values_obj(ship_values_cb);
    CircularBuffer ship_indices_obj(ship_indices_cb);
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if (!own_empty) {
            // Leaf reduction of this core's chunk slice. The chunk base is
            // latched at 0 and stepped with the all-constant advance
            // primitive: a runtime chunk_base cannot reach the
            // TTI-immediate config write inside the init.
            topk_xl_separate_indices_row_major_init_static<0, 0>();
            for (uint32_t c = 0; c < start_chunk; ++c) {
                topk_xl_separate_indices_row_major_advance_chunk_base<K>();
            }

            // A node whose leaf survivor ships directly (no tree merges)
            // rebuilds ASCENDING on its last DST action; the root (and any
            // survivor with merges still ahead) rebuilds DESCENDING.
            const bool ship_after_leaf = !is_tree_root && (num_merges == 0);

            const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
            sort_classic_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

            if (num_chunks == 1) {
                topk_xl_init<K, false>();
                topk_xl_rebuild<K, false>(slot0, ship_after_leaf);
            }

            for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
                const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
                sort_classic_chunk<K>(input_cb_obj, slot1, active_elements, true);

                topk_xl_init<K, false>();
                topk_xl_merge<K, false>(slot0);
                const bool last_chunk = (chunk + 1 == num_chunks);
                topk_xl_rebuild<K, false>(slot0, last_chunk && ship_after_leaf);
            }
        }

        // Tree merges: one partner sequence per winning level, in level order.
        for (uint32_t m = 0; m < num_merges; ++m) {
            const bool ship_next = !is_tree_root && (m + 1 == num_merges);
            const bool adopt = own_empty && (m == 0);
            const uint32_t dst_base = adopt ? slot0 : slot1;

            recv_obj.wait_front(sequence_tiles);
            // EXPLICIT format reconfig (bf16 -> UInt32): the preceding chunk phase
            // (topk_xl_copy_tile_init) hw-configures the unpacker for the BF16
            // input CB, and the plain _short init's state_configure is a no-op
            // in production JIT builds (the compute-kernel sentinel is compiled
            // out), so the _with_dt variant's llk reconfig is required here.
            // The reverse transition is explicit inside topk_xl_copy_tile_init.
            copy_tile_to_dst_init_short_with_dt(input_cb, recv_cb);
            for (uint32_t t = 0; t < sequence_tiles; ++t) {
                copy_tile(recv_cb, t, dst_base + t);
            }
            recv_obj.pop_front(sequence_tiles);

            topk_xl_init<K, false>();
            if (!adopt) {
                topk_xl_merge<K, false>(slot0);
            }
            topk_xl_rebuild<K, false>(slot0, ship_next /* ascending for the winner's halver */);
        }

#ifdef TOPK_TREE_ROOT
        // Root epilogue: materialize the indices exactly like the row-parallel
        // kernel (mark_neginf + rank-order transpose + pack_untilize).
        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_out_cb);
#endif

        tile_regs_commit();
        tile_regs_wait();

#ifdef TOPK_TREE_ROOT
        indices_out_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_out_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_out_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_out_obj.push_back(1);
#else
        // Ship the survivor raw: bit-exact FP32 value tiles then UINT32 index
        // tiles; the winner's copy_tile round-trips the DST image unchanged.
        ship_values_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0, ship_values_cb, tiles_per_sequence);
        ship_values_obj.push_back(tiles_per_sequence);

        ship_indices_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0 + tiles_per_sequence, ship_indices_cb, tiles_per_sequence);
        ship_indices_obj.push_back(tiles_per_sequence);
#endif

        tile_regs_release();
    }

#else  // !TOPK_TREE

    // ------------------------------------------------------------------
    // ROW-PARALLEL body (Classic / FusedEndToEnd / FusedSegmented)
    // ------------------------------------------------------------------
    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);

    CircularBuffer input(input_cb);
    CircularBuffer indices(indices_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if constexpr (body_mode == ComputeBodyMode::FusedSegmented) {
            reduce_segmented_row<K>(input, num_chunks, tail_elements);
        } else if constexpr (body_mode == ComputeBodyMode::FusedEndToEnd) {
            reduce_fused_row<K>(input, num_chunks, tail_elements);
        } else {
            reduce_classic_row<K>(input, num_chunks, tail_elements);
        }

        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_cb, 1, 0, slot0 + tiles_per_sequence);
        indices.push_back(1);

        tile_regs_release();
    }

#endif  // TOPK_TREE
}
