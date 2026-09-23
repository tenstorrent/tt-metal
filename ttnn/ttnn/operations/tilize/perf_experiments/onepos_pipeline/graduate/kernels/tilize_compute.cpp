// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute (TRISC0/1/2) — the `tilize_block` block operation (op_design.md).
//
// Default: ONE compute_kernel_lib::tilize call covers every block of this Tensix
// core, so tilize init / reconfig / uninit happen once per kernel. The helper
// processes core_row_tiles * num_col_blocks helper-blocks of block_width tiles;
// every quantum is the nominal block_width (the ragged last column block
// tilizes stale tail columns that the writer never writes).
//
// Split reader (CT `split_reader`): the input arrives through two CBs (even
// sequence positions from NCRISC, odd from BRISC — one producer each), which a
// single helper call cannot alternate between. The same helper is called once
// per tile-row with its documented back-to-back lifecycle: init (and the
// unpack/pack reconfig) on the first call only, InitUninitMode::Neither +
// NoReconfigure in the middle, uninit on the last. Both input CBs have identical
// format and tile geometry, so the one init configures both.
//
// No data-format reconfig: compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles) is the
// only prior configuration and it programs srcA / srcB / pack for exactly these two CBs, so the
// helper's UnpackAndPackReconfigure re-issued identical config (STALLWAITs + cfg writes) on the
// critical path. Measured (WH B0, perf_experiments/tilize_pack_throughput): HEIGHT-resident
// [1,1,2048,512] 1929 -> 1874 ns, BLOCK-resident [1,1,512,512] 1964 -> 1809 ns, 2-tile rows -15 %;
// DRAM-bound shapes flat; bit-exact on every dtype pair.
//
// Numeric formats (CT `fp32_lossless`): cb_input_sticks carries the input dtype and
// cb_output_tiles the output dtype; the value-preserving cast happens at pack. A 32-bit
// input (Float32 / Int32 / UInt32) is tagged UnpackToDestFp32 on the host and runs with
// fp32_dest_acc_en=true, so the helper is asked for Fp32Mode::Lossless: tilize IS the final
// consumer here, so the fast path's fp32 -> tf32 truncation would corrupt the output.
//
// Column sub-blocks (define TILIZE_SUB_BLOCK_TILES, Perf 2 onepos_pipeline; host knob
// SUB_BLOCK_TILES): each tile-row is tilized as column sub-blocks of that many tiles in the
// writer's rotated production order (tilize_sub_blocks.hpp), pushing each sub-block's output pages
// as soon as it is packed, so the writer's tile writes start after the first sub-block instead of
// after the whole tile-row. RT arg 2 = stick_rotation (the writer's first column). Raw WH LLK: see
// tilize_cols_fast below for what the helper cannot express and the measured helper-vs-raw ns.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

using namespace compute_kernel_lib::tilize_config;

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void tilize_one_row(bool first, bool last) {
    // compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles) already configured srcA/srcB and
    // the packer for exactly these formats (the odd CB matches the even one): no reconfig.
    constexpr auto reconfig = ReconfigureRegisterDatatypeMode::NoReconfigure;
    if (first && last) {
        compute_kernel_lib::
            tilize<block_width, cb_in, cb_out, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, reconfig, fp32_mode>(
                1);
    } else if (first) {
        compute_kernel_lib::
            tilize<block_width, cb_in, cb_out, InitUninitMode::InitOnly, WaitMode::WaitBlock, reconfig, fp32_mode>(1);
    } else if (last) {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::UninitOnly,
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

}  // namespace

#if defined(TILIZE_SUB_BLOCK_TILES) && defined(ARCH_WORMHOLE)
#include "tilize_sub_blocks.hpp"
namespace tilize_sub_blocks {

// RAW LLK — HELPER BYPASS (compute_kernel_lib::tilize, tilize_helpers.hpp). Kind: compute, tilize
// of a COLUMN SLICE of a wider tile-row. What the helper cannot express: it (and the compute API
// fast_tilize_block / tilize_block under it, tt_metal/hw/inc/api/compute/tilize.h) ties the
// unpacker's row stride to the width being tilized (WH fast_tilize_block: `full_dim = block`;
// llk_unpack_tilize_block(icb, block_c_tiles, ...) passes block_c_tiles as the stride) and takes no
// column offset inside the row. A tile-row's sticks in cb_input_sticks (or in a resident input shard)
// have stride block_width tiles, so a 2-tile sub-block needs stride = block_width with width = 2 at
// column offset `in_col`. The WH unpack LLK takes them separately:
// llk_unpack_fast_tilize_block(icb, tile_index, unit_dim, num_units, full_dim) and
// llk_unpack_tilize(icb, tile_index, block_ct_dim). tilize_cols_fast / tilize_cols_slow are the WH
// branches of fast_tilize_block / tilize_block with full_dim decoupled from the sub-block width;
// init / uninit and the fast-vs-lossless selection mirror the helper's InitAndUninit call
// (NoReconfigure, as the helper call below: compute_kernel_hw_startup configured these CBs).
// Measured (WH B0 n150, 64 Tensix cores, DEVICE KERNEL DURATION, same-session medians, bit-exact):
//   LOOSE_CASES[7] [1,1,2048,512] HEIGHT_SHARDED L1 -> DRAM: helper (whole row) 16997 ns vs raw
//   2-tile sub-blocks 15568 ns (-8.4 %, n=6); 16950 vs 15856 ns (-6.5 %, n=8). TRISC_2 kernel span
//   1672 (helper) -> 1878 cycles (raw), hidden under the writes: writer_wait 1361 -> 385 cycles.
//   Do not revert to the helper without re-measuring (perf_experiments/onepos_pipeline/README.md).
// WH only: the BH fast-tilize LLK has another signature (unit chunks, row begin / end); the host
// engages this path on Wormhole only.
ALWI void tilize_cols_fast(uint32_t icb, uint32_t block, uint32_t full_dim, uint32_t ocb, uint32_t in_col) {
    uint32_t packed_tiles = 0;
    uint32_t remaining_tiles = block;
    constexpr uint32_t dest_size = DST_ACCUM_MODE ? 4 : 8;
    const uint32_t unit_dim = full_dim == 1 ? 1 : 2;  // what fast_tilize_init(icb, full_dim, ocb) programmed
    uint32_t num_units = dest_size / unit_dim;
    while (packed_tiles < block) {
        const uint32_t read_tile_index = in_col + packed_tiles;
        const uint32_t write_tile_index = packed_tiles;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        if (remaining_tiles > 2 * dest_size) {
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += dest_size;
            remaining_tiles -= dest_size;
        } else if (remaining_tiles > dest_size) {
            const uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
            num_units = even_remainder / unit_dim;
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += even_remainder;
            remaining_tiles -= even_remainder;
        } else {
            if (remaining_tiles % 2 == 0 || unit_dim == 1) {
                num_units = remaining_tiles / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            } else if (remaining_tiles == 3) {
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, 3, 1)));
            } else {
                num_units = (remaining_tiles - 3) / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index + remaining_tiles - 3, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(remaining_tiles - 3, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(
                    remaining_tiles - 3, ocb, write_tile_index + remaining_tiles - 3, 3, 1)));
            }
            packed_tiles += remaining_tiles;
            remaining_tiles = 0;
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// The slow (lossless fp32 / tiny tile / non-fast-format) tilize of a column slice: tilize_block's WH
// body with llk_unpack_tilize(icb, column, full_dim) per tile instead of llk_unpack_tilize_block.
ALWI void tilize_cols_slow(uint32_t icb, uint32_t block, uint32_t full_dim, uint32_t ocb, uint32_t in_col) {
    for (uint32_t t = 0; t < block; ++t) {
        UNPACK((llk_unpack_tilize(icb, in_col + t, full_dim)));
    }
    for (uint32_t t = 0; t < block; ++t) {
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, icb)));
        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0, ocb, t)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// num_blocks tile-rows (the walk's positions, column block outer: Walker order), each block_width
// pages in cb_in, tilized as SubBlocks<block_width, sb_tiles> column sub-blocks in production order.
template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode, uint32_t sb_tiles>
ALWI void tilize_rows(uint32_t num_blocks, uint32_t core_row_tiles, uint32_t core_col_tiles, uint32_t stick_rotation) {
    using SB = SubBlocks<block_width, sb_tiles>;
    constexpr bool lossless = fp32_mode == Fp32Mode::Lossless && compute_kernel_lib::is_fp32_input_format<cb_in>();
    constexpr bool use_fast = compute_kernel_lib::can_use_fast_tilize<block_width, cb_in, cb_out>() && !lossless;
    if constexpr (use_fast) {
        fast_tilize_init(cb_in, block_width, cb_out);
    } else {
        tilize_init(cb_in, block_width, cb_out);
    }
    uint32_t row_in_block = 0;
    uint32_t cols_left = core_col_tiles;  // tile-columns of this column block onwards
    for (uint32_t seq = 0; seq < num_blocks; ++seq) {
        const uint32_t valid_width = cols_left < block_width ? cols_left : block_width;
        const uint32_t j0 = SB::start(stick_rotation, valid_width);
        cb_wait_front(cb_in, block_width);
        for (uint32_t p = 0; p < SB::n; ++p) {
            const uint32_t k = SB::at(j0, p);
            const uint32_t w = SB::width(k);
            cb_reserve_back(cb_out, w);
            if constexpr (use_fast) {
                tilize_cols_fast(cb_in, w, block_width, cb_out, SB::first(k));
            } else {
                tilize_cols_slow(cb_in, w, block_width, cb_out, SB::first(k));
            }
            cb_push_back(cb_out, w);
        }
        cb_pop_front(cb_in, block_width);
        if (++row_in_block == core_row_tiles) {
            row_in_block = 0;
            cols_left -= valid_width;
        }
    }
    if constexpr (use_fast) {
        fast_tilize_uninit(cb_in, cb_out, block_width);
    } else {
        tilize_uninit(cb_in, cb_out);
    }
}

}  // namespace tilize_sub_blocks
#endif  // TILIZE_SUB_BLOCK_TILES && ARCH_WORMHOLE

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
    // OCCUPANCY, not payload: the helper does its own per-block cb_wait_front (unpack) and
    // cb_reserve_back (pack), so this zone includes starvation on the reader and back-pressure
    // from the writer.
    MaybeDeviceZoneScope("compute_tilize");
#if defined(TILIZE_SUB_BLOCK_TILES) && defined(ARCH_WORMHOLE)
    // Host-engaged (SUB_BLOCK_TILES) where the writer streams the output with store_rows (not
    // resident, no split reader) and block_width >= 4 (n > 1), on every walk: the writer makes the
    // same decision from the same define.
    static_assert(!split_reader, "sub-blocks: one input CB");
    static_assert(tilize_sub_blocks::SubBlocks<block_width, TILIZE_SUB_BLOCK_TILES>::n > 1, "host: block_width >= 4");
    tilize_sub_blocks::tilize_rows<block_width, cb_input_sticks, cb_output_tiles, fp32_mode, TILIZE_SUB_BLOCK_TILES>(
        num_blocks, core_row_tiles, core_col_tiles, get_arg_val<uint32_t>(2));
#else
    if constexpr (!split_reader) {
        compute_kernel_lib::tilize<
            block_width,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
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
#endif  // TILIZE_SUB_BLOCK_TILES && ARCH_WORMHOLE
}
