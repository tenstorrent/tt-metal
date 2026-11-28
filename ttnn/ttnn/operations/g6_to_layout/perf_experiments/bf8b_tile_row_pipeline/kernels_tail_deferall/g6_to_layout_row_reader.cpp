// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row reader (NCRISC / NoC0) of the ROW_MAJOR -> TILE regimes: load_rows_block (+ reorder_block).
//
// A block's tile-row is loaded as one read per live row, carrying the block's live bytes of that
// row, into a stage of tile_row_stage_rows rows x block_cols_tiles * 64 B.
//
// row_major_to_tile (reorder): the stage is a ring of row_window_depth tile-row slots
// (cb_row_stage). While tile-row k is reordered, tile-rows k+1 .. k+depth-1 are on the wire under
// their own trids. The reorder is the inverse of staged_reorder's: 32-B self-aimed copies of live
// face rows into the block's tile slots of cb_tile_block (slot i * block_cols_tiles + j), pushed once
// the block's copies have retired (g6_to_layout_face_copies.hpp). Bytes move as bytes, so every bf16
// pattern survives; tile padding is left as it is (it is never a logical value).
// split_reorder: the reader copies the top faces (rows 0-15) and pushes each landed slot to the
// writer, which copies the bottom faces (rows 16-31) on its own NoC and pops the slot; otherwise
// the stage is reader-private (never pushed) and the reader copies both halves.
//
// packer_tilize (!reorder): the stage is cb_row_stage itself, the tilize input of the compute
// kernel (32 rows per tile-row = block_cols_tiles tile pages). Per block: one reserve of the nominal
// block, one read barrier, one conditioning pass over the live rows (condition_block_for_bfp8), one
// push.
//
// Zones (profiler builds only): per block, the reserve wait and the load (+ reorder) work each have
// a zone; trid and barrier waits inside a block are sum zones (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_face_copies.hpp"

using namespace g6_to_layout;
namespace dkl = dataflow_kernel_lib;

namespace {

constexpr uint32_t cb_row_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
constexpr uint32_t cb_tile_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
constexpr bool reorder = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
constexpr uint32_t row_window_depth = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
constexpr bool split_reorder = get_compile_time_arg_val(Geometry::next_ct_arg + 4) != 0;
constexpr uint32_t input_args_offset = Geometry::next_ct_arg + 5;
// Face halves this RISC copies: [0, reader_end_half).
constexpr uint32_t reader_end_half = split_reorder ? 1 : 2;

constexpr uint32_t block_tiles = Geometry::block_rows_tiles * Geometry::block_cols_tiles;
// A bfp8 exponent group: the 16 values of one face row.
constexpr uint32_t GROUP_BYTES = FACE_ROW_BYTES;
constexpr bool zero_group_tails = !reorder && Geometry::width % FACE_HW != 0;
// Trids 1..depth tag the tile-row loads; two more alternate per tile over the reorder copies.
constexpr uint32_t reorder_trid_base = row_window_depth + 1;
static_assert(!reorder || (row_window_depth >= 1 && reorder_trid_base + 1 <= NOC_MAX_TRANSACTION_ID));
// A pushed-ahead slot needs a second free page while the current one is still unpushed.
static_assert(!split_reorder || row_window_depth >= 2);

// Live rows of block tile-row i -> stage rows 0.., one read each.
template <typename Accessor>
FORCE_INLINE void load_tile_row(const Block& block, uint32_t i, uint32_t stage, const Accessor& input) {
    const uint32_t g = block.tile_row + i;
    const uint32_t first_page = first_row_page_of(g);
    const uint32_t live_rows = live_rows_of(g);
    const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
    const uint32_t segment_bytes = row_segment_bytes(block);
    for (uint32_t r = 0; r < live_rows; ++r) {
        noc_async_read(input.get_noc_addr(first_page + r, segment_offset), stage, segment_bytes);
        stage += Geometry::stage_row_bytes;
    }
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

// Tile-rows of the core's blocks in order, one per load.
struct TileRow {
    Block block;
    uint32_t i;
    bool last_of_block;
};

struct TileRowCursor {
    BlockWalk walk;
    Block block;
    uint32_t i;
    uint32_t rows_left;

    FORCE_INLINE bool next(TileRow& tile_row) {
        if (rows_left == 0) {
            if (!walk.next(block)) {
                return false;
            }
            i = 0;
            rows_left = block.rows;
        }
        tile_row = TileRow{block, i, rows_left == 1};
        ++i;
        --rows_left;
        return true;
    }
};

template <typename Accessor>
FORCE_INLINE void load_and_reorder(uint32_t tile_begin, uint32_t num_tiles, const Accessor& input) {
    Noc noc;
    constexpr uint32_t slot_bytes = Geometry::stage_tile_row_bytes;
    const uint32_t window_base = get_write_ptr(cb_row_stage);
    TileRowCursor issue_cursor{{tile_begin, tile_begin + num_tiles}, {}, 0, 0};
    TileRow in_flight[row_window_depth];
    uint32_t issued = 0;
    uint32_t done = 0;
    uint32_t parity = 0;

    // Tile-row n lands in slot n % depth, issued depth - 1 loads ahead of its reorder, into the slot
    // whose previous tile-row was reordered (and its copies retired) one iteration earlier.
    auto issue_next = [&]() {
        TileRow tile_row;
        if (!issue_cursor.next(tile_row)) {
            return;
        }
        if constexpr (split_reorder) {
            // Slots issued ahead are reserved but not yet pushed; this one needs one more page.
            DeviceZoneScopedSumN1("reserve_row_stage");
            cb_reserve_back(cb_row_stage, issued - done + 1);
        }
        const uint32_t slot = issued % row_window_depth;
        dkl::set_read_trid(noc, 1 + slot);
        load_tile_row(tile_row.block, tile_row.i, window_base + slot * slot_bytes, input);
        in_flight[slot] = tile_row;
        ++issued;
    };
    for (uint32_t d = 0; d + 1 < row_window_depth; ++d) {
        issue_next();
    }
    // The tile-row at the top of each iteration opens a block.
    for (;;) {
        {
            DeviceZoneScopedN("issue_row_load");
            issue_next();
        }
        if (done == issued) {
            break;
        }
        {
            DeviceZoneScopedN("reserve_tile_block");
            cb_reserve_back(cb_tile_block, block_tiles);
        }
        {
            DeviceZoneScopedN("load_reorder_block");
            const uint32_t tiles_base = get_write_ptr(cb_tile_block);
            for (;;) {
                const uint32_t slot = done % row_window_depth;
                const TileRow tile_row = in_flight[slot];
                {
                    DeviceZoneScopedSumN1("wait_row_load");
                    dkl::async_read_barrier_with_trid(noc, 1 + slot);
                }
                if constexpr (split_reorder) {
                    cb_push_back(cb_row_stage, 1);
                }
                copy_tile_row_faces<0, reader_end_half>(
                    noc,
                    tile_row.block,
                    tile_row.i,
                    window_base + slot * slot_bytes,
                    tiles_base,
                    reorder_trid_base,
                    parity);
                ++done;
                if (tile_row.last_of_block) {
                    break;
                }
                issue_next();
            }
        }
        cb_push_back(cb_tile_block, block_tiles);
    }
    dkl::set_read_trid(noc, 0);
}

template <typename Accessor>
FORCE_INLINE void load_for_tilize(uint32_t tile_begin, uint32_t num_tiles, const Accessor& input) {
    BlockWalk blocks{tile_begin, tile_begin + num_tiles};
    Block block;
    while (blocks.next(block)) {
        {
            DeviceZoneScopedN("reserve_row_stage");
            cb_reserve_back(cb_row_stage, block_tiles);
        }
        const uint32_t stage = get_write_ptr(cb_row_stage);
        {
            DeviceZoneScopedN("load_rows_block");
            for (uint32_t i = 0; i < block.rows; ++i) {
                load_tile_row(block, i, stage + i * Geometry::stage_tile_row_bytes, input);
            }
            {
                DeviceZoneScopedSumN1("row_stage_read_barrier");
                noc_async_read_barrier();
            }
            // The NoC wrote the stage; this RISC may hold lines of an earlier block at these addresses.
            invalidate_l1_cache();
            {
                DeviceZoneScopedN("condition_block_for_bfp8");
                condition_block_for_bfp8(block, stage);
            }
            // Drain the stores before the unpacker reads the stage.
            invalidate_l1_cache();
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
    if constexpr (reorder) {
        load_and_reorder(tile_begin, num_tiles, input);
    } else {
        load_for_tilize(tile_begin, num_tiles, input);
    }
    {
        DeviceZoneScopedN("final_read_barrier");
        noc_async_read_barrier();
    }
}
