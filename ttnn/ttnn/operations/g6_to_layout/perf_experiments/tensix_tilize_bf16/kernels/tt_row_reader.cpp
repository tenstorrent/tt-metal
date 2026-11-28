// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Bake-off candidate (perf_experiments/tensix_tilize_bf16): row reader (NCRISC / NoC0) of a bf16
// ROW_MAJOR -> TILE conversion whose reorder runs on the Tensix (unpack-tilize -> MOVA2D -> pack).
//
// It only loads: each live row segment of a tile-row lands as one stage row of cb_row_stage, 32 rows
// per tile-row = block_cols_tiles bf16 tile pages (the tilize unpacker's input). No 32-B self-aimed
// copies and no bfp8 conditioning (that pass exists only for the bfp8 packer). Rows past H and
// columns past W are left as they are: they only reach output tile padding.
//
// read_window == 0: the op's R6 schedule (load_for_tilize without conditioning): one reserve of the
//   nominal block, every row read, one read barrier, one push of the block.
// read_window >= 1: a trid ring per tile-row. Tile-row n lands in ring slot n % stage_tile_rows under
//   trid 1 + n % read_window; tile-row n + read_window - 1 is issued before tile-row n is waited, and
//   every landed tile-row is pushed on its own (block_cols_tiles pages), so the tilize starts on it
//   while the next ones are on the wire. A ragged block still pushes the nominal block_rows_tiles
//   tile-rows (the compute counts nominal tile-rows; the writer waits the nominal block).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;
namespace dkl = dataflow_kernel_lib;

#ifndef TT_ABLATE
#define TT_ABLATE 0  // ablation only (never a result): bit 0 = no row reads
#endif

namespace {

constexpr uint32_t cb_row_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
constexpr uint32_t read_window = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
constexpr uint32_t stage_tile_rows = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
constexpr uint32_t input_args_offset = Geometry::next_ct_arg + 3;

constexpr uint32_t block_tiles = Geometry::block_rows_tiles * Geometry::block_cols_tiles;
constexpr uint32_t tile_row_pages = Geometry::block_cols_tiles;
static_assert(read_window <= stage_tile_rows, "a reserved-ahead tile-row needs its own ring slot");
static_assert(read_window <= NOC_MAX_TRANSACTION_ID, "one trid per tile-row in flight");
// Divisor of the ring's trid / slot arithmetic (the ring is not instantiated for read_window == 0).
constexpr uint32_t ring_trids = read_window == 0 ? 1 : read_window;

// Live rows of block tile-row i -> stage rows 0.., one read each.
template <typename Accessor>
FORCE_INLINE void load_tile_row(const Block& block, uint32_t i, uint32_t stage, const Accessor& input) {
    const uint32_t g = block.tile_row + i;
    const uint32_t first_page = first_row_page_of(g);
    const uint32_t live_rows = live_rows_of(g);
    const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
    const uint32_t segment_bytes = row_segment_bytes(block);
    for (uint32_t r = 0; r < live_rows; ++r) {
#if !(TT_ABLATE & 1)
        noc_async_read(input.get_noc_addr(first_page + r, segment_offset), stage, segment_bytes);
#endif
        stage += Geometry::stage_row_bytes;
    }
}

template <typename Accessor>
FORCE_INLINE void load_blocks_barrier(uint32_t tile_begin, uint32_t num_tiles, const Accessor& input) {
    BlockWalk blocks{tile_begin, tile_begin + num_tiles};
    Block block;
    while (blocks.next(block)) {
        cb_reserve_back(cb_row_stage, block_tiles);
        const uint32_t stage = get_write_ptr(cb_row_stage);
        for (uint32_t i = 0; i < block.rows; ++i) {
            load_tile_row(block, i, stage + i * Geometry::stage_tile_row_bytes, input);
        }
        noc_async_read_barrier();
        cb_push_back(cb_row_stage, block_tiles);
    }
}

// Nominal tile-rows of the core's blocks in order; `live` is false for the padding tile-rows of a
// ragged block (pushed, never loaded).
struct NominalTileRow {
    Block block;
    uint32_t i;
    bool live;
};

struct NominalCursor {
    BlockWalk walk;
    Block block;
    uint32_t i;
    bool open;

    FORCE_INLINE bool next(NominalTileRow& row) {
        if (!open || i == Geometry::block_rows_tiles) {
            if (!walk.next(block)) {
                return false;
            }
            i = 0;
            open = true;
        }
        row = NominalTileRow{block, i, i < block.rows};
        ++i;
        return true;
    }
};

template <typename Accessor>
FORCE_INLINE void load_ring(uint32_t tile_begin, uint32_t num_tiles, const Accessor& input) {
    Noc noc;
    const uint32_t ring_base = get_write_ptr(cb_row_stage);
    NominalCursor cursor{{tile_begin, tile_begin + num_tiles}, {}, 0, false};
    uint32_t issued = 0;
    uint32_t done = 0;

    auto issue_next = [&]() -> bool {
        NominalTileRow row;
        if (!cursor.next(row)) {
            return false;
        }
        // Tile-rows issued ahead are reserved but not pushed yet; this one needs one more tile-row.
        cb_reserve_back(cb_row_stage, (issued - done + 1) * tile_row_pages);
        dkl::set_read_trid(noc, 1 + issued % ring_trids);
        if (row.live) {
            load_tile_row(
                row.block, row.i, ring_base + (issued % stage_tile_rows) * Geometry::stage_tile_row_bytes, input);
        }
        ++issued;
        return true;
    };

    for (uint32_t d = 0; d + 1 < ring_trids; ++d) {
        issue_next();
    }
    for (;;) {
        issue_next();
        if (done == issued) {
            break;
        }
        dkl::async_read_barrier_with_trid(noc, 1 + done % ring_trids);
        cb_push_back(cb_row_stage, tile_row_pages);
        ++done;
    }
    dkl::set_read_trid(noc, 0);
}

}  // namespace

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<input_args_offset>();
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto input = TensorAccessor(input_args, input_addr);

    DeviceZoneScopedN("tt_row_reader");
    if constexpr (read_window == 0) {
        load_blocks_barrier(tile_begin, num_tiles, input);
    } else {
        load_ring(tile_begin, num_tiles, input);
    }
    noc_async_read_barrier();
}
