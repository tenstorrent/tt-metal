// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute — perf_experiments/tilize_pack_throughput variant source (HAND-AUTHORED).
// make_variants.py copies the op's kernels/ into kernels_<name>/, replaces tilize_compute.cpp
// with this file and writes pt_knobs.hpp next to it:
//
//   PT_MODE     0 = the op's helper call (compute_kernel_lib::tilize), knobs below applied to it
//               1 = RAW fast tilize: llk_{unpack,math,pack}_fast_tilize_block driven directly,
//                   with a caller-chosen DEST section size (tiles per math->pack handoff)
//               2 = RAW batched standard tilize: unpack-tilize per tile (HW tileize mode) +
//                   A2D datacopy + standard llk_pack, PT_SECTION tiles per DEST section
//   PT_RECONFIG 1 = UnpackAndPackReconfigure after compute_kernel_hw_startup (head), 0 = none
//   PT_UNINIT   1 = uninit at kernel end (head), 0 = skip (InitOnly)
//   PT_SECTION  tiles per DEST section (mode 1/2), clamped to the SyncHalf DEST half
//   PT_NOZERO   1 = pack releases DEST without ZEROACC (mode 1/2)
//
// Raw-LLK justification (modes 1/2): bypasses compute_kernel_lib::tilize (and the
// fast_tilize_block API wrapper), whose WH DEST-section schedule is fixed at "fill the DEST
// half" (8 bf16 tiles): the helper cannot express a smaller math->pack handoff, which is the
// lever measured here (pack is the long pole; a smaller first section starts the packer earlier
// and shortens the post-math pack tail).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "pt_knobs.hpp"
#ifndef PT_NOINIT
#define PT_NOINIT 0  // ablation (NOT correct): skip init/uninit/reconfig entirely
#endif
#ifndef PT_BPS
#define PT_BPS 8  // mode 3: max tile-row blocks sharing one DEST section
#endif
#ifndef PT_GROUP_CB
#define PT_GROUP_CB 0  // mode 3: 1 = one CB wait/reserve/push/pop per DEST section when the CB allows it
#endif
#ifndef PT_STD_NO_TO_DEST
#define PT_STD_NO_TO_DEST 0  // mode 5: 1 = leave the unpack-to-DEST (32-bit input) path on the helper
#endif
#ifndef PT_AB
#define PT_AB 0  // ablation bits (NOT correct): 1 = no unpack/math payload, 2 = no pack payload
#endif

namespace {

using namespace compute_kernel_lib::tilize_config;

constexpr auto RECONFIG = PT_RECONFIG ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                      : ReconfigureRegisterDatatypeMode::NoReconfigure;
constexpr auto FULL_MODE = PT_UNINIT ? InitUninitMode::InitAndUninit : InitUninitMode::InitOnly;
constexpr auto LAST_MODE = PT_UNINIT ? InitUninitMode::UninitOnly : InitUninitMode::Neither;

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void tilize_one_row(bool first, bool last) {
    if (first && last) {
        compute_kernel_lib::tilize<block_width, cb_in, cb_out, FULL_MODE, WaitMode::WaitBlock, RECONFIG, fp32_mode>(1);
    } else if (first) {
        compute_kernel_lib::
            tilize<block_width, cb_in, cb_out, InitUninitMode::InitOnly, WaitMode::WaitBlock, RECONFIG, fp32_mode>(1);
    } else if (last) {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            LAST_MODE,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            fp32_mode>(1);
    } else {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::Neither,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            fp32_mode>(1);
    }
}

// ---------------------------------------------------------------- raw paths (mode 1 / 2)

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
constexpr bool raw_fast_ok() {
    constexpr bool lossless = fp32_mode == Fp32Mode::Lossless && compute_kernel_lib::is_fp32_input_format<cb_in>();
    return (block_width > 1 || PT_MODE == 3) && compute_kernel_lib::can_use_fast_tilize<block_width, cb_in, cb_out>() &&
           !lossless;
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
constexpr bool helper_uses_fast() {
    constexpr bool lossless = fp32_mode == Fp32Mode::Lossless && compute_kernel_lib::is_fp32_input_format<cb_in>();
    return compute_kernel_lib::can_use_fast_tilize<block_width, cb_in, cb_out>() && !lossless;
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
constexpr bool raw_std_ok() {
    // Mode 2 (bf16 study) keeps unpack-to-DEST and full-sync out; mode 5 batches the helper's own
    // slow path, unpack-to-DEST (32-bit inputs) and dst_full_sync_en included: the per-tile
    // unpack<->math UNPACK_TO_DEST handshake takes the DEST tile index from math, so it is
    // independent of where the DEST section boundaries fall.
    constexpr bool to_dest = compute_kernel_lib::is_fp32_input_format<cb_in>() && fp32_mode == Fp32Mode::Lossless;
    constexpr bool tiles_32x32 =
        compute_kernel_lib::dfb_has_32x32_tiles<cb_out>() && compute_kernel_lib::dfb_has_32x32_tiles<cb_in>();
    if constexpr (PT_MODE == 5) {
        return tiles_32x32 && !(PT_STD_NO_TO_DEST && to_dest);
    } else {
        return !to_dest && !compute_kernel_lib::get_dst_full_sync_enabled() && tiles_32x32 &&
               !compute_kernel_lib::is_fp32_input_format<cb_in>();
    }
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
constexpr uint32_t raw_mode() {
    if constexpr ((PT_MODE == 1 || PT_MODE == 3) && raw_fast_ok<block_width, cb_in, cb_out, fp32_mode>()) {
        return 1;
    } else if constexpr (PT_MODE == 2 && raw_std_ok<block_width, cb_in, cb_out, fp32_mode>()) {
        return 2;
    } else if constexpr (
        PT_MODE == 5 && !helper_uses_fast<block_width, cb_in, cb_out, fp32_mode>() &&
        raw_std_ok<block_width, cb_in, cb_out, fp32_mode>()) {
        return 2;
    } else {
        return 0;
    }
}

constexpr uint32_t DEST_HALF_TILES = DST_ACCUM_MODE ? 4 : 8;
constexpr uint32_t SECTION = PT_SECTION < DEST_HALF_TILES ? PT_SECTION : DEST_HALF_TILES;
// Standard (slow-path) tiles per DEST section: the whole DEST under dst_full_sync_en, else half.
constexpr uint32_t STD_DEST_TILES = (compute_kernel_lib::get_dst_full_sync_enabled() ? 16 : 8) >>
                                    (DST_ACCUM_MODE ? 1 : 0);
constexpr uint32_t STD_SECTION = PT_SECTION < STD_DEST_TILES ? PT_SECTION : STD_DEST_TILES;

FORCE_INLINE void pack_section_done() {
#if PT_NOZERO
    // _llk_pack_dest_section_done_ without the ZEROACC: tilize overwrites every DEST row it packs.
#ifdef TRISC_PACK
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    _llk_packer_set_math_semaphore_<p_stall::NONE>();
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DST_SYNC_MODE>();
#endif
#else
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
#endif
}

// Next section size: SECTION tiles, never leaving a 1-tile remainder (fast tilize has no
// unit_dim-1 unit once initialised for unit_dim 2).
FORCE_INLINE uint32_t next_section(uint32_t rem) {
    uint32_t sec = rem < SECTION ? rem : SECTION;
    if (rem - sec == 1) {
        sec = rem <= DEST_HALF_TILES ? rem : sec - 1;
    }
    return sec;
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void raw_fast_block() {
    uint32_t done = 0;
    while (done < block_width) {
        const uint32_t sec = next_section(block_width - done);
        const uint32_t n2 = (sec & 1) ? (sec - 3) >> 1 : sec >> 1;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        if (n2) {
            if constexpr (!(PT_AB & 1)) {
                UNPACK((llk_unpack_fast_tilize_block(cb_in, done, 2, n2, block_width)));
                MATH((llk_math_fast_tilize_block_(0, cb_in, 2, n2)));
            }
            if constexpr (!(PT_AB & 2)) {
                PACK((llk_pack_fast_tilize_block(0, cb_out, done, 2, n2)));
            }
        }
        if (sec & 1) {
            const uint32_t t = n2 << 1;
            if constexpr (!(PT_AB & 1)) {
                UNPACK((llk_unpack_fast_tilize_block(cb_in, done + t, 3, 1, block_width)));
                MATH((llk_math_fast_tilize_block_(t, cb_in, 3, 1)));
            }
            if constexpr (!(PT_AB & 2)) {
                PACK((llk_pack_fast_tilize_block(t, cb_out, done + t, 3, 1)));
            }
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        pack_section_done();
        done += sec;
    }
}

// ---- mode 3: DEST sections that span tile-row blocks (block_width <= DEST half / 2)

template <uint32_t block_width>
constexpr uint32_t blocks_per_section() {
    if constexpr (PT_MODE != 3 || block_width * 2 > DEST_HALF_TILES) {
        return 1;
    } else {
        return (DEST_HALF_TILES / block_width) < PT_BPS ? (DEST_HALF_TILES / block_width) : PT_BPS;
    }
}

// One whole block's fast-tilize units at DEST tile offset `dst`; the block's first input row-tile
// is `in_tile` (0, or j * block_width * TILE_R_DIM for the j-th block of a grouped CB wait) and its
// first output tile `out_tile`.
template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void fast_units(uint32_t dst, uint32_t in_tile, uint32_t out_tile) {
    if constexpr (block_width == 1) {
        if constexpr (!(PT_AB & 1)) {
            UNPACK((llk_unpack_fast_tilize_block(cb_in, in_tile, 1, 1, 1)));
            MATH((llk_math_fast_tilize_block_(dst, cb_in, 1, 1)));
        }
        if constexpr (!(PT_AB & 2)) {
            PACK((llk_pack_fast_tilize_block(dst, cb_out, out_tile, 1, 1)));
        }
    } else {
        constexpr uint32_t n2 = (block_width & 1) ? (block_width - 3) >> 1 : block_width >> 1;
        if constexpr (n2 > 0) {
            if constexpr (!(PT_AB & 1)) {
                UNPACK((llk_unpack_fast_tilize_block(cb_in, in_tile, 2, n2, block_width)));
                MATH((llk_math_fast_tilize_block_(dst, cb_in, 2, n2)));
            }
            if constexpr (!(PT_AB & 2)) {
                PACK((llk_pack_fast_tilize_block(dst, cb_out, out_tile, 2, n2)));
            }
        }
        if constexpr (block_width & 1) {
            constexpr uint32_t t = n2 << 1;
            if constexpr (!(PT_AB & 1)) {
                UNPACK((llk_unpack_fast_tilize_block(cb_in, in_tile + t, 3, 1, block_width)));
                MATH((llk_math_fast_tilize_block_(dst + t, cb_in, 3, 1)));
            }
            if constexpr (!(PT_AB & 2)) {
                PACK((llk_pack_fast_tilize_block(dst + t, cb_out, out_tile + t, 3, 1)));
            }
        }
    }
}

// Blocks [b, b + k) in one DEST section. Per-block CB ops (default), or -- PT_GROUP_CB and a CB
// whose capacity is a multiple of the group (so the group never wraps) -- one wait/reserve for
// the group. Each thread decides for its own CB only; the DEST sectioning is identical on all.
template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, uint32_t cb_in_odd, bool split_reader>
FORCE_INLINE void fast_section(
    DataflowBuffer& in_even,
    DataflowBuffer& in_odd,
    DataflowBuffer& out,
    uint32_t b,
    uint32_t k,
    bool grp_in,
    bool grp_out) {
    MATH((llk_math_wait_for_dest_available()));
    PACK((llk_packer_wait_for_math_done()));
    if (grp_in) {
        in_even.wait_front(k * block_width);
    }
    if (grp_out) {
        out.reserve_back(k * block_width);
    }
    for (uint32_t j = 0; j < k; ++j) {
        const bool odd = split_reader && ((b + j) & 1);
        DataflowBuffer& in = odd ? in_odd : in_even;
        if (!grp_in) {
            in.wait_front(block_width);
        }
        if (!grp_out) {
            out.reserve_back(block_width);
        }
        const uint32_t in_tile = grp_in ? j * block_width * TILE_R_DIM : 0;
        const uint32_t out_tile = grp_out ? j * block_width : 0;
        if (odd) {
            fast_units<block_width, cb_in_odd, cb_out>(j * block_width, in_tile, out_tile);
        } else {
            fast_units<block_width, cb_in, cb_out>(j * block_width, in_tile, out_tile);
        }
        if (!grp_out) {
            out.push_back(block_width);
        }
        if (!grp_in) {
            in.pop_front(block_width);
        }
    }
    if (grp_out) {
        out.push_back(k * block_width);
    }
    if (grp_in) {
        in_even.pop_front(k * block_width);
    }
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    pack_section_done();
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void raw_std_block() {
    if constexpr (!(PT_AB & 1)) {
        UNPACK((llk_unpack_tilize_block(cb_in, block_width, 0)));
    }
    uint32_t done = 0;
    while (done < block_width) {
        const uint32_t rem = block_width - done;
        const uint32_t sec = rem < STD_SECTION ? rem : STD_SECTION;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        for (uint32_t i = 0; i < sec && !(PT_AB & 1); ++i) {
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    i, cb_in)));
        }
        for (uint32_t i = 0; i < sec && !(PT_AB & 2); ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, cb_out, done + i)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        pack_section_done();
        done += sec;
    }
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void raw_init() {
    if constexpr (PT_NOINIT) {
        return;
    }
    constexpr uint32_t m = raw_mode<block_width, cb_in, cb_out, fp32_mode>();
    if constexpr (PT_RECONFIG) {
        reconfig_data_format_srca(cb_in);
        if constexpr (m == 1) {
            reconfig_data_format_srcb(cb_in);
        }
        pack_reconfig_data_format(cb_out);
    }
    if constexpr (m == 1) {
        fast_tilize_init(cb_in, block_width, cb_out);
    } else {
        tilize_init(cb_in, block_width, cb_out);
    }
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void raw_uninit() {
    if constexpr (PT_UNINIT && !PT_NOINIT) {
        if constexpr (raw_mode<block_width, cb_in, cb_out, fp32_mode>() == 1) {
            fast_tilize_uninit(cb_in, cb_out, block_width);
        } else {
            tilize_uninit(cb_in, cb_out);
        }
    }
}

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void raw_one_block(DataflowBuffer& in, DataflowBuffer& out) {
    in.wait_front(block_width);
    out.reserve_back(block_width);
    if constexpr (raw_mode<block_width, cb_in, cb_out, fp32_mode>() == 1) {
        raw_fast_block<block_width, cb_in, cb_out>();
    } else {
        raw_std_block<block_width, cb_in, cb_out>();
    }
    out.push_back(block_width);
    in.pop_front(block_width);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width = get_compile_time_arg_val(2);
    constexpr bool split_reader = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_input_sticks_odd = get_compile_time_arg_val(4);
    constexpr Fp32Mode fp32_mode = get_compile_time_arg_val(5) != 0 ? Fp32Mode::Lossless : Fp32Mode::Fast;

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(1);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t num_blocks = core_row_tiles * num_col_blocks;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    MaybeDeviceZoneScope("compute_tilize");
    constexpr uint32_t m = raw_mode<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>();
    if constexpr (m == 0) {
        if constexpr (!split_reader) {
            compute_kernel_lib::tilize<
                block_width,
                cb_input_sticks,
                cb_output_tiles,
                FULL_MODE,
                WaitMode::WaitBlock,
                RECONFIG,
                fp32_mode>(num_blocks);
        } else {
            for (uint32_t seq = 0; seq < num_blocks; ++seq) {
                const bool first = seq == 0;
                const bool last = seq + 1 == num_blocks;
                if (seq & 1) {
                    tilize_one_row<block_width, cb_input_sticks_odd, cb_output_tiles, fp32_mode>(first, last);
                } else {
                    tilize_one_row<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>(first, last);
                }
            }
        }
    } else {
        // Both input CBs share format and tile geometry: one init configures both.
        raw_init<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>();
        DataflowBuffer in_even(cb_input_sticks);
        DataflowBuffer out(cb_output_tiles);
        constexpr uint32_t bps = blocks_per_section<block_width>();
        if constexpr (m == 1 && bps > 1) {
            DataflowBuffer in_odd(split_reader ? cb_input_sticks_odd : cb_input_sticks);
            constexpr uint32_t grp_pages = bps * block_width;
            // grouped CB ops only for one input CB (the split reader alternates two)
            bool grp_in = false;
            bool grp_out = false;
            if constexpr (PT_GROUP_CB && !split_reader) {
                UNPACK((grp_in = compute_kernel_lib::get_dfb_num_pages(cb_input_sticks) % grp_pages == 0));
            }
            if constexpr (PT_GROUP_CB) {
                PACK((grp_out = compute_kernel_lib::get_dfb_num_pages(cb_output_tiles) % grp_pages == 0));
            }
            for (uint32_t b = 0; b < num_blocks; b += bps) {
                const uint32_t k = num_blocks - b < bps ? num_blocks - b : bps;
                fast_section<block_width, cb_input_sticks, cb_output_tiles, cb_input_sticks_odd, split_reader>(
                    in_even, in_odd, out, b, k, grp_in, grp_out);
            }
        } else if constexpr (!split_reader) {
            for (uint32_t b = 0; b < num_blocks; ++b) {
                raw_one_block<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>(in_even, out);
            }
        } else {
            DataflowBuffer in_odd(cb_input_sticks_odd);
            for (uint32_t seq = 0; seq < num_blocks; ++seq) {
                if (seq & 1) {
                    raw_one_block<block_width, cb_input_sticks_odd, cb_output_tiles, fp32_mode>(in_odd, out);
                } else {
                    raw_one_block<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>(in_even, out);
                }
            }
        }
        raw_uninit<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>();
    }
}
