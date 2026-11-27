// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage_tiles reader (NCRISC / NoC0): reads the core's staged input pages into
// cb_staged_tiles a block at a time, then overwrites the part of every edge tile that lies
// outside the logical input with c, copied over the NoC from this core's fill page at the
// same in-tile offset.
//
// load_block uses raw noc_async_read + TensorAccessor: kernel_lib has no interleaved batch reader.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_pad_common.hpp"

namespace {

// Self-aimed L1 -> L1 copies of [off, off + len) from the fill page into a staged tile.
// Runs arrive in increasing offset order; adjacent runs are merged into one transfer.
class MaskCopier {
public:
    MaskCopier(uint32_t fill_page) : fill_page_(fill_page) {}

    FORCE_INLINE void add(uint32_t tile_addr, uint32_t off, uint32_t len) {
        if (len_ != 0 && tile_ == tile_addr && off == off_ + len_) {
            len_ += len;
            return;
        }
        flush();
        tile_ = tile_addr;
        off_ = off;
        len_ = len;
    }

    FORCE_INLINE void flush() {
        if (len_ == 0) {
            return;
        }
        UnicastEndpoint self;
        noc_.async_read(
            self,
            CoreLocalMem<uint32_t>(tile_ + off_),
            len_,
            dataflow_kernel_lib::local_addr(fill_page_ + off_, noc_.get_noc_id()),
            {});
        len_ = 0;
    }

private:
    Noc noc_;
    uint32_t fill_page_;
    uint32_t tile_ = 0, off_ = 0, len_ = 0;
};

// Mask of one edge tile: every element outside [0, row_mask) x [0, col_mask) takes c.
template <uint32_t es>
FORCE_INLINE void mask_tile(MaskCopier& copier, uint32_t tile_addr, uint32_t row_mask, uint32_t col_mask) {
    constexpr uint32_t row_bytes = g6_pad::FACE * es;
    constexpr uint32_t face_bytes = g6_pad::FACE * row_bytes;
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_base = face * face_bytes;
        const uint32_t row0 = g6_pad::FACE * (face >> 1);
        const uint32_t col0 = g6_pad::FACE * (face & 1);
        const uint32_t rows = row_mask > row0 ? (row_mask - row0 < g6_pad::FACE ? row_mask - row0 : g6_pad::FACE) : 0;
        const uint32_t cols = col_mask > col0 ? (col_mask - col0 < g6_pad::FACE ? col_mask - col0 : g6_pad::FACE) : 0;
        if (rows == 0 || cols == 0) {
            copier.add(tile_addr, face_base, face_bytes);
            continue;
        }
        if (cols < g6_pad::FACE) {
            for (uint32_t r = 0; r < rows; ++r) {
                copier.add(tile_addr, face_base + r * row_bytes + cols * es, (g6_pad::FACE - cols) * es);
            }
        }
        if (rows < g6_pad::FACE) {
            copier.add(tile_addr, face_base + rows * row_bytes, (g6_pad::FACE - rows) * row_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_staged_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fill_page_reader = get_compile_time_arg_val(1);
    constexpr uint32_t es = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr bool fill_is_zero = get_compile_time_arg_val(4) != 0;
    constexpr auto in_args = TensorAccessorArgs<5>();

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t ht_in = get_common_arg_val<uint32_t>(2);
    const uint32_t wt_in = get_common_arg_val<uint32_t>(3);
    const uint32_t row_mask = get_common_arg_val<uint32_t>(4);
    const uint32_t col_mask = get_common_arg_val<uint32_t>(5);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(6);

    const uint32_t staged_start = get_arg_val<uint32_t>(0);
    const uint32_t staged_count = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const bool has_masks = row_mask < g6_pad::TILE || col_mask < g6_pad::TILE;
    const uint32_t fill_page = get_write_ptr(cb_fill_page_reader);
    MaskCopier copier(fill_page);

    uint32_t page_id = staged_start;
    // The lead indices are irrelevant here: lead1 = 1 makes the walk track (ht, wt) only.
    g6_pad::StagedWalk walk(staged_start, ht_in, wt_in, 1);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t done = block_idx * block_pages;
        const uint32_t block_extent = staged_count - done < block_pages ? staged_count - done : block_pages;

        cb_reserve_back(cb_staged_tiles, block_pages);
        const uint32_t block_base = get_write_ptr(cb_staged_tiles);
        {
            DeviceZoneScopedN("G6PAD-READ-BLOCK");
            uint32_t slot = block_base;
            for (uint32_t i = 0; i < block_extent; ++i) {
                noc_async_read(in.get_noc_addr(page_id + i), slot, page_bytes);
                slot += page_bytes;
            }
            if (block_idx == 0 && has_masks) {
                DeviceZoneScopedN("G6PAD-FILL-PAGE");
                g6_pad::build_fill_page<cb_fill_page_reader, es, fill_is_zero>(0, page_bytes, c_bits);
            }
            noc_async_read_barrier();
        }
        page_id += block_extent;

        if (has_masks) {
            DeviceZoneScopedN("G6PAD-MASK");
            uint32_t slot = block_base;
            for (uint32_t i = 0; i < block_extent; ++i) {
                const uint32_t tile_row_mask = walk.ht == ht_in - 1 ? row_mask : g6_pad::TILE;
                const uint32_t tile_col_mask = walk.wt == wt_in - 1 ? col_mask : g6_pad::TILE;
                if (tile_row_mask < g6_pad::TILE || tile_col_mask < g6_pad::TILE) {
                    mask_tile<es>(copier, slot, tile_row_mask, tile_col_mask);
                }
                walk.advance(ht_in, wt_in, 1);
                slot += page_bytes;
            }
            copier.flush();
            noc_async_read_barrier();
        }
        cb_push_back(cb_staged_tiles, block_pages);
    }
}
