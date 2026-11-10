// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Down compute (TRISC): the down projection on a down core. For each sub-block of the schedule (a run of one expert's
// token rows):
//   y[sub-block rows, this core's y columns] = h[sub-block rows, :] @ W_down[:, this core's y columns]
// one row tile at a time, accumulated in DST over all K tiles of h. h arrives K-tile-major ([K tiles][row tiles]: a
// gate/up core's slice of it is one contiguous run). An expert's down weights stay in the in1 ring (filled by
// down_weight_reader.cpp) for all its sub-blocks and are freed after the last one.
//
// y leaves as tiles (row tile after row tile of the sub-block) or, with flat_options::y_row_major, pack-untilized row
// tile by row tile (y_row_major.hpp writes the rows).
//
// Compile-time args: 0 row tiles per sub-block, 1 K tiles (the intermediate size / 32), 2 K tiles per weight block,
// 3 y tile columns of this core, 4 local experts, 5 (unused), 6 tiles per weight block, 7 the in1 ring's blocks
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "flat_options.hpp"
#include "schedule_meta.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace {

constexpr uint32_t weight_cb = tt::CBIndex::c_1;
constexpr uint32_t h_cb = tt::CBIndex::c_2;
constexpr uint32_t schedule_cb = tt::CBIndex::c_6;
constexpr uint32_t out_cb = tt::CBIndex::c_16;

constexpr uint32_t row_tiles = get_compile_time_arg_val(0);
constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(2);
constexpr uint32_t columns = get_compile_time_arg_val(3);
constexpr uint32_t block_tiles = get_compile_time_arg_val(6);
constexpr uint32_t ring_blocks = get_compile_time_arg_val(7);
constexpr uint32_t blocks_per_expert = k_tiles / k_tiles_per_block;
constexpr uint32_t h_tiles = k_tiles * row_tiles;
// matmul_block's in0 block width (kt_dim)
constexpr uint32_t in0_block_k_tiles = 8;
constexpr uint32_t dest_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
constexpr bool reads_per_load = flat_options::down_weights_per_load;
static_assert(columns <= 16 && k_tiles % k_tiles_per_block == 0 && k_tiles % in0_block_k_tiles == 0);
static_assert(ring_blocks >= blocks_per_expert);
static_assert(!reads_per_load || ring_blocks == flat_options::gate_up_ring_regions * blocks_per_expert);

constexpr uint32_t largest_divisor(uint32_t value, uint32_t cap) {
    for (uint32_t divisor = cap < value ? cap : value; divisor > 1; --divisor) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

uint32_t blocks_freed = 0;

// The weight CB tile offset of stream block `block`, once it has landed. The blocks sit in the ring in stream order
// (block b in ring slot b % ring_blocks) and are freed in that order.
FORCE_INLINE uint32_t weight_block(uint32_t block) {
    cb_wait_front(weight_cb, (block - blocks_freed + 1) * block_tiles);
    return static_cast<uint32_t>(
               static_cast<int32_t>(block % ring_blocks) - static_cast<int32_t>(blocks_freed % ring_blocks)) *
           block_tiles;
}

// With down_weights_per_load: the weight CB tile offset of stream block `block` in ring slot `ring_slot`, once it has
// landed. A load's blocks sit in its ring region and are freed by count after the load's last use (as
// gate_up_compute.cpp's gate/up weight ring).
FORCE_INLINE uint32_t weight_block_in_region(uint32_t block, uint32_t ring_slot) {
    if (block + 1 > blocks_freed) {
        cb_wait_front(weight_cb, (block + 1 - blocks_freed) * block_tiles);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(ring_slot) - static_cast<int32_t>(blocks_freed % ring_blocks)) *
           block_tiles;
}

template <bool row_major_y>
void compute_schedule() {
    // y columns per pass (DST holds dest_tiles outputs; the K loop runs once per pass over the same h and weights).
    // Row-major y: pack_untilize places pass c's columns at c x the pass width, so every pass has the same width.
    constexpr uint32_t pass_columns =
        row_major_y ? largest_divisor(columns, dest_tiles) : (columns < dest_tiles ? columns : dest_tiles);
    static_assert(!row_major_y || columns % pass_columns == 0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(h_cb, weight_cb, out_cb);
    matmul_block_init(h_cb, weight_cb, false, pass_columns, 1, in0_block_k_tiles);
    if constexpr (row_major_y) {
        pack_untilize_dest_init<pass_columns, columns>(out_cb);
    }
    // The schedule page (schedule_meta.hpp, from down_core_writer.cpp)
    cb_wait_front(schedule_cb, 1);
    auto page_word = [](uint32_t word) { return read_tile_value(schedule_cb, 0, word); };
    const uint32_t entries = page_word(schedule_page::kEntries);
    const uint32_t total_sub_blocks = page_word(schedule_page::kTotalSubBlocks);
    // the current entry and its sub-block
    uint32_t entry = 0, sub_block = 0;
    uint32_t entry_sub_blocks = 0, entry_load_word = 0, entry_last_row_tiles = row_tiles;
    auto read_entry = [&]() {
        entry_sub_blocks = entry < entries ? page_word(schedule_page::kSubBlocks + entry) : 0;
        entry_load_word = entry < entries ? page_word(schedule_page::kLoadWord + entry) : 0;
        entry_last_row_tiles = entry < entries ? page_word(schedule_page::kLastRowTiles + entry) : row_tiles;
    };
    read_entry();
    for (uint32_t done = 0; done < total_sub_blocks; ++done) {
        const bool last_sub_block = sub_block + 1 == entry_sub_blocks;
        // row tiles holding tokens (the rest are never written out)
        const uint32_t rows = last_sub_block ? entry_last_row_tiles : row_tiles;
        // after this sub-block the weights go: the entry's last sub-block (per load: the load's last use)
        const bool frees_weights =
            last_sub_block && (!reads_per_load || schedule_page::load_word_last_use(entry_load_word));
        const uint32_t load = schedule_page::load_word_load(entry_load_word);
        const uint32_t region_first_slot = schedule_page::load_word_ring_region(entry_load_word) * blocks_per_expert;
        cb_wait_front(h_cb, h_tiles);
        if constexpr (!row_major_y) {
            cb_reserve_back(out_cb, row_tiles * columns);
        }
        for (uint32_t row = 0; row < rows; ++row) {
            if constexpr (row_major_y) {
                cb_reserve_back(out_cb, columns);
            }
            for (uint32_t first_column = 0; first_column < columns; first_column += pass_columns) {
                const uint32_t width = columns - first_column < pass_columns ? columns - first_column : pass_columns;
                const bool last_pass = first_column + width == columns;
                if constexpr (columns > pass_columns) {
                    matmul_block_init(h_cb, weight_cb, false, width, 1, in0_block_k_tiles);
                }
                tile_regs_acquire();
                uint32_t block_offset = 0;
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    const uint32_t block = k / k_tiles_per_block;
                    if (k % k_tiles_per_block == 0) {
                        block_offset = reads_per_load ? weight_block_in_region(
                                                            load * blocks_per_expert + block, region_first_slot + block)
                                                      : weight_block(entry * blocks_per_expert + block);
                    }
                    matmul_block(
                        h_cb,
                        weight_cb,
                        row + k * row_tiles,
                        block_offset + (k % k_tiles_per_block) * columns + first_column,
                        0,
                        false,
                        width,
                        1,
                        in0_block_k_tiles);
                    // the weights' last row (final column pass): each block goes as soon as it is used, so the next
                    // expert's blocks stream into the ring while this row still runs
                    if (frees_weights && last_pass && row == rows - 1 &&
                        k % k_tiles_per_block == k_tiles_per_block - 1) {
                        cb_pop_front(weight_cb, block_tiles);
                        ++blocks_freed;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (row_major_y) {
                    pack_untilize_dest<pass_columns, columns>(out_cb, 1, first_column / pass_columns);
                } else {
                    for (uint32_t column = 0; column < width; ++column) {
                        // [row tiles x columns], row tile major
                        pack_tile<true>(column, out_cb, row * columns + first_column + column);
                    }
                }
                tile_regs_release();
            }
            if constexpr (row_major_y) {
                cb_push_back(out_cb, columns);
            }
        }
        cb_pop_front(h_cb, h_tiles);
        if constexpr (!row_major_y) {
            cb_push_back(out_cb, row_tiles * columns);
        }
        if (++sub_block == entry_sub_blocks) {
            sub_block = 0;
            ++entry;
            read_entry();
        }
    }
    cb_pop_front(schedule_cb, 1);
}

}  // namespace

void kernel_main() { compute_schedule<flat_options::y_row_major>(); }
