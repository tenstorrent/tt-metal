// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// One RISC's share of a split row, and the raw SFPU routines those shares call.
//
// A compute kernel is compiled once per RISC (jit_build/genfiles.cpp defines TRISC_UNPACK, TRISC_MATH or
// TRISC_PACK), and in a split row the three do not run the same program: UNPACK walks chunks, MATH walks
// topk_large_indices_split's 9-steps-per-chunk schedule, PACK walks its interleaved stage order. That cannot be
// expressed with the statement-level UNPACK()/MATH()/PACK() macros, so each share is a function here whose guard
// is on the inside and which compiles to nothing on the other two threads. compute.cpp then reads as plain calls
// and carries no TRISC guard of its own.
//
// This is op glue, and it stays with the op: the shares name this op's CBs and epilogues, and the SFPU routines
// below are this op's semantics (the -inf index sentinel, the segment-result move). What it drives --
// topk_large_indices_split.hpp -- is the generic part that belongs in tt-llk. Keeping the two apart means that
// move does not have to disentangle op logic first.
//
// Collecting the shares here is also what lets compute.cpp carry no TRISC guard at all.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_split.hpp"

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

}  // namespace ckernel::sfpu
#endif

// A split row runs its SFPU segments on PACK, so the pack thread needs these too.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {

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

// A K = 2048 segment's unfused result from tiles 0 to 3 to tiles 4 to 7, raw words; ADDR_MOD_0 steps two rows.
inline void _topk_large_indices_move_segment_result_() {
    constexpr uint32_t rows_per_tile = 64;
    constexpr uint32_t result_tiles = 4;
    constexpr uint32_t rows_per_access = 2;
    constexpr uint32_t iterations = result_tiles * rows_per_tile / rows_per_access;
    constexpr uint32_t destination_offset = result_tiles * rows_per_tile;
    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, destination_offset);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace topk_large_indices_split_stages {

template <uint32_t K>
constexpr uint32_t tiles_per_sequence = (K + TILE_R_DIM * TILE_C_DIM - 1) / (TILE_R_DIM * TILE_C_DIM);

using ttnn::operations::experimental::topk_large_indices::program::max_fused_chunks;

// UNPACK's share of a row: feed every chunk once, plus the next row's first chunk when MATH will prefetch it.
template <uint32_t K>
FORCE_INLINE void feed_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, bool first_prefetched, bool prefetch_next) {
#ifdef TRISC_UNPACK
    const uint32_t input_cb = input.get_cb_id();
    for (uint32_t chunk = first_prefetched ? 1 : 0; chunk < num_chunks; ++chunk) {
        input.wait_front(tiles_per_sequence<K>);
        if (chunk == (first_prefetched ? 1 : 0)) {
            topk_xl_copy_tile_init(input_cb);
        }
        topk_xl_copy_tile<K>(
            input_cb, 0 /*dst*/, 0 /*first tile*/, chunk + 1 == num_chunks ? tail_elements : K /*active_elements*/);
        input.pop_front(tiles_per_sequence<K>);
        llk_unpack_set_srcb_dummy_valid();
        if (chunk > 0) {
            llk_unpack_set_srcb_dummy_valid();
        }
    }
    if (prefetch_next) {
        // MATH copies the next row's first chunk before this row's index transposes, so UNPACK feeds it first.
        input.wait_front(tiles_per_sequence<K>);
        topk_xl_copy_tile<K>(input_cb, 0 /*dst*/, 0 /*first tile*/, K /*active_elements*/);
        input.pop_front(tiles_per_sequence<K>);
        llk_unpack_set_srcb_dummy_valid();
    }
#endif
}

// MATH's share: the copies and face transposes, scheduled against PACK's SFPU work.
template <uint32_t K>
FORCE_INLINE void math_row(
    CircularBuffer& input, uint32_t num_chunks, uint32_t tail_elements, bool first_prefetched, bool prefetch_next) {
#ifdef TRISC_MATH
    const uint32_t input_cb = input.get_cb_id();
    _llk_math_eltwise_unary_sfpu_init_once_();
    if (first_prefetched) {
        // PACK may take the SFPU over only after this init.
        topk_large_indices_split::post_token();
    }
    // The previous row's index transposes reprogram MATH after a prefetched first chunk, so the first copy made here
    // inits it.
    bool init_copy = true;
    topk_large_indices_split::math_row<K>(
        num_chunks,
        [&](uint32_t chunk, uint32_t tile) {
            if (init_copy) {
                topk_xl_copy_tile_init(input_cb);
                init_copy = false;
            }
            topk_xl_copy_tile<K>(
                input_cb, tile, 0 /*first tile*/, chunk + 1 == num_chunks ? tail_elements : K /*active_elements*/);
        },
        first_prefetched,
        prefetch_next);
#endif
}

// PACK's share of a whole row: every SFPU stage, then the index split and the -inf marking.
template <uint32_t K>
FORCE_INLINE void sfpu_row(uint32_t num_chunks, bool first_prefetched) {
#ifdef TRISC_PACK
    if (first_prefetched) {
        topk_large_indices_split::take_math_token();
    }
    topk_large_indices_split::pack_row<K>(num_chunks, false /*mirror_last*/, [] {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
        ckernel::sfpu::_topk_xl_split_begin_(0);
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>();
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_();
        ckernel::sfpu::_topk_xl_split_begin_(0);
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>();
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    });
#endif
}

// UNPACK's share of one 32-chunk segment of a long K' 2048 row.
template <uint32_t K>
FORCE_INLINE void feed_segment(CircularBuffer& input, uint32_t seg_chunks, uint32_t seg_tail) {
#ifdef TRISC_UNPACK
    const uint32_t input_cb = input.get_cb_id();
    for (uint32_t chunk = 0; chunk < seg_chunks; ++chunk) {
        input.wait_front(tiles_per_sequence<K>);
        if (chunk == 0) {
            topk_xl_copy_tile_init(input_cb);
        }
        topk_xl_copy_tile<K>(input_cb, 0 /*dst*/, 0 /*first tile*/, chunk + 1 == seg_chunks ? seg_tail : K);
        input.pop_front(tiles_per_sequence<K>);
        llk_unpack_set_srcb_dummy_valid();
        if (chunk > 0) {
            llk_unpack_set_srcb_dummy_valid();
        }
    }
#endif
}

// MATH's share of one segment.
template <uint32_t K>
FORCE_INLINE void math_segment(CircularBuffer& input, uint32_t seg_chunks, uint32_t seg_tail) {
#ifdef TRISC_MATH
    const uint32_t input_cb = input.get_cb_id();
    _llk_math_eltwise_unary_sfpu_init_once_();
    topk_large_indices_split::math_row<K>(seg_chunks, [&](uint32_t chunk, uint32_t tile) {
        if (chunk == 0) {
            topk_xl_copy_tile_init(input_cb);
        }
        topk_xl_copy_tile<K>(input_cb, tile, 0 /*first tile*/, chunk + 1 == seg_chunks ? seg_tail : K);
    });
#endif
}

// PACK's share of one segment. `block` is the segment's 32-chunk block of the row, which decodes the
// segment-local stamps; `mirror` ends the segment ascending, which a later segment needs for its unfused merge
// and the last segment needs when the row's survivor leaves this core.
template <uint32_t K>
FORCE_INLINE void sfpu_segment(uint32_t seg_chunks, bool later, uint32_t block, bool mirror) {
#ifdef TRISC_PACK
    constexpr uint32_t accumulator_slot = 0;
    auto epilogue = [block, later] {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
        ckernel::sfpu::_topk_xl_split_begin_(accumulator_slot);
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>(block * (max_fused_chunks * K));
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
        if (later) {
            ckernel::sfpu::_topk_xl_split_begin_(accumulator_slot);
            ckernel::sfpu::_topk_large_indices_move_segment_result_();
            TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
        }
    };
    topk_large_indices_split::pack_row<K>(seg_chunks, mirror, epilogue);
#endif
}

// MATH hands Dst over and PACK spills the unfused accumulator to L1, so the next segment can use all eight tiles.
template <uint32_t K>
FORCE_INLINE void spill_accumulator(CircularBuffer& acc, uint32_t segment) {
#ifdef TRISC_MATH
    topk_large_indices_split::release_accumulator();
#endif
#ifdef TRISC_PACK
    constexpr uint32_t accumulator_tiles = 2 * tiles_per_sequence<K>;
    constexpr uint32_t accumulator_slot = 0;
    const uint32_t acc_cb = acc.get_cb_id();
    topk_large_indices_split::hold_accumulator([&] {
        // Segment 1 finds the packer in its untilize mode, a later segment in pack_tile mode with the MOP the pack
        // thread's SFPU work reprogrammed.
        if (segment == 1) {
            pack_untilize_uninit(acc_cb);
        } else {
            pack_init(acc_cb);
        }
        acc.reserve_back(accumulator_tiles);
        for (uint32_t tile = 0; tile < accumulator_tiles; ++tile) {
            pack_tile<true>(accumulator_slot + tile, acc_cb, tile);
        }
        acc.push_back(accumulator_tiles);
    });
#endif
}

// The rest of this op's per-thread LLK calls, named so compute.cpp holds no MATH()/PACK()/UNPACK() of its own.
// The macros are the ordinary way to write a one-thread statement -- copy_tile() expands into them -- but every
// one of these wraps a ckernel:: or _llk_ call, which is the part that does not belong in an op kernel.

// MATH's hot-path reinit between fused chunks of a row (see the routine above for why it is sound).
FORCE_INLINE void reinit_fused_after_stamp() { MATH((ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_())); }

// Rewrite every index paired with an exact -inf value to the sentinel.
template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t dst) {
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, dst, VectorMode::RC_custom)));
}

// Hand Dst back to MATH clearing only bank 0: bank 1 may already hold the next row's first chunk.
template <bool fp32_dest_acc>
FORCE_INLINE void release_dest_bank0() {
    PACK((topk_large_indices_split::release_dest_bank0<fp32_dest_acc>()));
}

// One word of the reader-to-compute mailbox.
FORCE_INLINE uint32_t mailbox_word(uint32_t meta_cb, uint32_t word) {
    return ckernel::read_tile_value(meta_cb, 0, word);
}

// MATH raises the two Tensix semaphores it shares with PACK to the token depth the schedule needs (their default
// max of 1 would saturate), then signals that its SFPU init is done so PACK may take the SFPU over.
FORCE_INLINE void begin_thread_split() {
    MATH((topk_large_indices_split::start()));
    PACK((topk_large_indices_split::wait_start()));
}

// Restores those two semaphores to their defaults, which the rest of the LLK expects.
FORCE_INLINE void end_thread_split() { MATH((topk_large_indices_split::finish())); }

}  // namespace topk_large_indices_split_stages
