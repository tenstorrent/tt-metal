// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row reader (NCRISC / NoC0) of the ROW_MAJOR -> TILE regimes: load_rows_block.
//
// A block's tile-rows land in cb_row_stage as 32-row tile-rows (block_cols_tiles tile pages each),
// the tilize input of the compute kernel: one read per live row carrying the block's live bytes of
// that row. Per block: one reserve of the nominal block, one read barrier, one push.
// condition_for_bfp8 (packer_tilize): before the push the live rows are conditioned for the bfp8
// packer (condition_block_for_bfp8). A bf16 output (row_major_to_tile) is a byte copy: the reader
// never touches the bytes, so its RISC needs no cache maintenance either.
//
// Zones (profiler builds only): per block, the reserve wait and the load each have a zone; the read
// barrier is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_row_issue.hpp"

using namespace g6_to_layout;

namespace {

constexpr uint32_t cb_row_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
constexpr bool condition_for_bfp8 = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
// Minimum cycles between two row reads (row_issue::ReadPacer); the host sets it for an L1 source.
constexpr uint32_t row_read_gap = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
constexpr uint32_t input_args_offset = Geometry::next_ct_arg + 3;
// One tile-row of reads may go out unpaced.
using Pacer = row_issue::ReadPacer<row_read_gap, TILE_HW>;

constexpr uint32_t block_tiles = Geometry::block_rows_tiles * Geometry::block_cols_tiles;
// A bfp8 exponent group: the 16 values of one face row.
constexpr uint32_t GROUP_BYTES = FACE_ROW_BYTES;
constexpr bool zero_group_tails = condition_for_bfp8 && Geometry::width % FACE_HW != 0;

// Live rows of block tile-row i -> stage rows 0.., one read each (g6_to_layout_row_issue.hpp).
template <typename Accessor>
FORCE_INLINE void load_tile_row(const Block& block, uint32_t i, uint32_t stage, const Accessor& input, Pacer& pacer) {
    const uint32_t g = block.tile_row + i;
    row_issue::read_rows(
        input,
        first_row_page_of(g),
        live_rows_of(g),
        block.col * TILE_ROW_BYTES,
        row_segment_bytes(block),
        stage,
        Geometry::stage_row_bytes,
        pacer);
}

// Bit 7 of each half of a bf16 pair set where that datum's 7 mantissa bits are all ones; XOR-ing
// the mask >> 7 lowers those datums to the next value toward zero.
// The bfp8 packer rounds such a datum up into the next binade when it holds its group's largest
// exponent, and then requantizes the whole group one exponent higher. The host saturates that
// datum to the largest mantissa instead and keeps the exponent (tech_reports/data_formats/
// data_formats.md, Mantissa Rounding), which is exactly the lowered value. A lowered datum below
// its group's exponent moves by at most a quarter of the group's quantization step.
FORCE_INLINE uint32_t all_ones_mantissas(uint32_t pair) { return ((pair & 0x007F007Fu) + 0x00010001u) & 0x00800080u; }

// Words per test: their loads are independent, so they issue back to back, and one OR-reduced test
// skips the common case (no all-ones mantissa) without a store.
constexpr uint32_t CONDITION_UNROLL = 8;

FORCE_INLINE void lower_saturating_mantissas(tt_l1_ptr uint32_t* words, uint32_t count) {
    uint32_t k = 0;
    for (; k + CONDITION_UNROLL <= count; k += CONDITION_UNROLL) {
        uint32_t pairs[CONDITION_UNROLL];
        uint32_t any = 0;
#pragma GCC unroll 8
        for (uint32_t u = 0; u < CONDITION_UNROLL; ++u) {
            pairs[u] = words[k + u];
            any |= all_ones_mantissas(pairs[u]);
        }
        if (any == 0) {
            continue;
        }
#pragma GCC unroll 8
        for (uint32_t u = 0; u < CONDITION_UNROLL; ++u) {
            words[k + u] = pairs[u] ^ (all_ones_mantissas(pairs[u]) >> 7);
        }
    }
    for (; k < count; ++k) {
        words[k] ^= all_ones_mantissas(words[k]) >> 7;
    }
}

// Prepares the block's live rows for the bfp8 packer: lowers all-ones mantissas (above), then zeroes
// the bytes after each row's last live value up to the end of its exponent group, so a group's
// shared exponent is set by live values only.
FORCE_INLINE void condition_block_for_bfp8(const Block& block, uint32_t stage) {
    const uint32_t segment_bytes = row_segment_bytes(block);
    const uint32_t segment_words = (segment_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    const bool zero_tail = zero_group_tails && block.col + block.cols == Geometry::tensor_cols_tiles;
    const uint32_t group_end = (segment_bytes + GROUP_BYTES - 1) / GROUP_BYTES * GROUP_BYTES;
    for (uint32_t i = 0; i < block.rows; ++i) {
        const uint32_t live_rows = live_rows_of(block.tile_row + i);
        uint32_t row = stage + i * Geometry::stage_tile_row_bytes;
        for (uint32_t r = 0; r < live_rows; ++r) {
            lower_saturating_mantissas(reinterpret_cast<tt_l1_ptr uint32_t*>(row), segment_words);
            if (zero_tail) {
                tt_l1_ptr uint16_t* datums = reinterpret_cast<tt_l1_ptr uint16_t*>(row);
                for (uint32_t b = segment_bytes; b < group_end; b += DATUM_BYTES) {
                    datums[b / DATUM_BYTES] = 0;
                }
            }
            row += Geometry::stage_row_bytes;
        }
    }
}

template <typename Accessor>
FORCE_INLINE void load_for_tilize(uint32_t tile_begin, uint32_t num_tiles, const Accessor& input) {
    BlockWalk blocks{tile_begin, tile_begin + num_tiles};
    Block block;
    Pacer pacer;
    while (blocks.next(block)) {
        {
            DeviceZoneScopedN("reserve_row_stage");
            cb_reserve_back(cb_row_stage, block_tiles);
        }
        const uint32_t stage = get_write_ptr(cb_row_stage);
        {
            DeviceZoneScopedN("load_rows_block");
            for (uint32_t i = 0; i < block.rows; ++i) {
                load_tile_row(block, i, stage + i * Geometry::stage_tile_row_bytes, input, pacer);
            }
            {
                DeviceZoneScopedSumN1("row_stage_read_barrier");
                noc_async_read_barrier();
            }
            if constexpr (condition_for_bfp8) {
                // The NoC wrote the stage; this RISC may hold lines of an earlier block at these addresses.
                invalidate_l1_cache();
                {
                    DeviceZoneScopedN("condition_block_for_bfp8");
                    condition_block_for_bfp8(block, stage);
                }
                // Drain the stores before the unpacker reads the stage.
                invalidate_l1_cache();
            }
        }
        // The nominal block every time: the compute kernel and the ring stay block-aligned.
        cb_push_back(cb_row_stage, block_tiles);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<input_args_offset>();
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto input = TensorAccessor(input_args, input_addr);

    DeviceZoneScopedN("g6_row_reader");
    load_for_tilize(tile_begin, num_tiles, input);
}
