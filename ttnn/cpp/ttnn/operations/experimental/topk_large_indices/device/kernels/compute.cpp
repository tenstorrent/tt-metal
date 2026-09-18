// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
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
// The fused index stamp carries the chunk id in five bits.
constexpr uint32_t fused_chunk_id_mask = 31;

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

// Every body reduces the row chunks [first_chunk, first_chunk + num_chunks) into an unfused
// [values, indices] survivor at DST slot 0 with row-global indices. The survivor is sorted
// descending unless final_ascending, which prepares it as merge operand one of another core.
template <uint32_t K>
FORCE_INLINE void reduce_fused_row(
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements, bool final_ascending) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor has no separate index tiles. Its merge operand is
    // therefore adjacent at one sequence, rather than two sequences, away.
    constexpr uint32_t incoming_slot = tiles_per_sequence;

    // The row has at most 32 chunks, so the row-global chunk id fits the stamp directly.
    sort_fused_chunk<K, true>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, first_chunk);

    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, final_ascending);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
            sort_fused_chunk<K, false>(input, incoming_slot, active_elements, true, first_chunk + chunk);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, final_ascending && chunk + 1 == num_chunks);
        }
    }

    // Decode the five-bit chunk stamp once, after the final fused survivor is
    // known, and leave the standard [values, indices] epilogue layout.
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

template <uint32_t K>
FORCE_INLINE void reduce_segmented_row(
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements, bool final_ascending) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t segment_capacity = 32;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence;

    const uint32_t end_chunk_total = first_chunk + num_chunks;
    // Segments follow the row's 32-chunk blocks: the stamp is the chunk id within its block and the
    // block base is a multiple of 32*K, so it ORs into the decoded index without overlap.
    uint32_t segment_first = first_chunk;
    for (uint32_t segment = 0; segment_first < end_chunk_total; ++segment) {
        const uint32_t block_end = (segment_first / segment_capacity + 1) * segment_capacity;
        const uint32_t end_chunk = block_end < end_chunk_total ? block_end : end_chunk_total;
        const uint32_t last_chunk = end_chunk - 1;
        const bool last_segment = end_chunk == end_chunk_total;

        // Segment zero is already in the final accumulator slot. Later
        // segments use the second unfused sequence. Their fused input chunk
        // is adjacent to the fused survivor at base + tiles_per_sequence.
        const uint32_t base = segment == 0 ? accumulator_slot : segment_slot;
        const uint32_t chunk_slot = base + tiles_per_sequence;
        const bool mirror_final_survivor = segment != 0 || (last_segment && final_ascending);

        const uint32_t first_elements = segment_first + 1 == end_chunk_total ? tail_elements : K;
        if (segment == 0) {
            sort_fused_chunk<K, true>(input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        } else {
            sort_fused_chunk<K, false>(input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        }

        if (segment_first == last_chunk) {
            topk_xl_rebuild<K, true>(base, mirror_final_survivor);
        } else {
            for (uint32_t chunk = segment_first + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == end_chunk_total ? tail_elements : K;
                sort_fused_chunk<K, false>(input, chunk_slot, active_elements, true, chunk & fused_chunk_id_mask);
                topk_xl_merge<K, true>(base);
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
            }
        }

        // Split once per segment. The base is a multiple of 32*K, so it does
        // not overlap the decoded segment-local index bits.
        topk_xl_separate_indices_row_major_global_init();
        topk_xl_separate_indices_row_major_global_base<K>(
            base, (segment_first / segment_capacity) * (segment_capacity * K));

        if (segment != 0) {
            // Operand zero is descending and each later segment was rebuilt
            // ascending, satisfying the unfused bitonic merge precondition.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, last_segment && final_ascending);
        }
        segment_first = end_chunk;
    }
}

// Tree merge round: the child's unfused [values, indices] survivor lands as raw FP32 tiles and is
// copied next to our own survivor, then folded in with the same unfused merge the bodies use.
template <uint32_t K>
FORCE_INLINE void merge_landed_survivor(CircularBuffer& landing, uint32_t survivor_slot, bool final_ascending) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence;
    const uint32_t landing_cb = landing.get_cb_id();
    const uint32_t incoming_slot = survivor_slot + survivor_tiles;

    landing.wait_front(survivor_tiles);
    reconfig_data_format_srca(landing_cb);
    copy_init(landing_cb);
    for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
        copy_tile(landing_cb, tile, incoming_slot + tile);
    }
    landing.pop_front(survivor_tiles);

    topk_xl_init<K, false>();
    topk_xl_merge<K, false>(survivor_slot);
    topk_xl_rebuild<K, false>(survivor_slot, final_ascending);
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
    const uint32_t num_rows = get_arg_val<uint32_t>(topk_core_args::compute_num_rows);
    const uint32_t seg_first_chunk = get_arg_val<uint32_t>(topk_core_args::compute_seg_first_chunk);
    const uint32_t seg_end_chunk = get_arg_val<uint32_t>(topk_core_args::compute_seg_end_chunk);
    const uint32_t num_recv_rounds = get_arg_val<uint32_t>(topk_core_args::compute_num_recv_rounds);
    const bool sends_survivor = get_arg_val<uint32_t>(topk_core_args::compute_sends_survivor) != 0;
    const uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::compute_search_length);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto body_mode = static_cast<ComputeBodyMode>(get_compile_time_arg_val(3));
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(5);
    constexpr uint32_t landing_cb = get_compile_time_arg_val(6);
    constexpr uint32_t send_cb = get_compile_time_arg_val(7);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(
        body_mode == ComputeBodyMode::FusedEndToEnd || body_mode == ComputeBodyMode::FusedSegmented,
        "invalid TopK compute body mode");

    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence;
    constexpr uint32_t final_survivor = 0;

    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);

    uint32_t num_chunks;
    uint32_t tail_elements;
    if constexpr (valid_length_from_metadata) {
        // The reader already clipped the segment against the metadata length.
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.wait_front(1);
        num_chunks = ckernel::read_tile_value(meta_cb, 0, topk_metadata_num_chunks_word);
        tail_elements = ckernel::read_tile_value(meta_cb, 0, topk_metadata_tail_elements_word);
        meta_cb_obj.pop_front(1);
    } else {
        const auto segment =
            calculate_topk_segment_bounds(calculate_topk_bounds(search_len, K), K, seg_first_chunk, seg_end_chunk);
        num_chunks = segment.num_chunks;
        tail_elements = segment.tail_elements;
    }

    CircularBuffer input(input_cb);
    CircularBuffer indices(indices_cb);
    CircularBuffer landing(landing_cb);
    CircularBuffer send(send_cb);

    // A survivor that leaves this core is merge operand one of its parent and must end ascending.
    const bool body_final_ascending = sends_survivor && num_recv_rounds == 0;

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if constexpr (body_mode == ComputeBodyMode::FusedSegmented) {
            reduce_segmented_row<K>(input, seg_first_chunk, num_chunks, tail_elements, body_final_ascending);
        } else {
            reduce_fused_row<K>(input, seg_first_chunk, num_chunks, tail_elements, body_final_ascending);
        }

        for (uint32_t round = 0; round < num_recv_rounds; ++round) {
            merge_landed_survivor<K>(landing, final_survivor, sends_survivor && round + 1 == num_recv_rounds);
        }

        if (sends_survivor) {
            tile_regs_commit();
            tile_regs_wait();

            // Raw tile pack so the parent can unpack the survivor straight back into DST.
            send.reserve_back(survivor_tiles);
            pack_untilize_uninit(send_cb);
            pack_reconfig_data_format(send_cb);
            for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
                pack_tile(final_survivor + tile, send_cb);
            }
            send.push_back(survivor_tiles);
        } else {
            mark_neginf_indices<K>(final_survivor);
            materialize_index_rank_order<K>(final_survivor, indices_cb);

            tile_regs_commit();
            tile_regs_wait();

            indices.reserve_back(1);
            pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(
                indices_cb, 1, 0, final_survivor + tiles_per_sequence);
            indices.push_back(1);
        }

        tile_regs_release();

        if (sends_survivor) {
            // Re-arm the untilize pack the row output path expects.
            pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
        }
    }
}
