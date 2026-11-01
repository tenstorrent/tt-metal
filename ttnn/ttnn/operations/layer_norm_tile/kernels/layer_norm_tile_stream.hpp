// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared tile-stream pieces of the layer_norm reader and writer.
//
// read_block_chunks: a trid ring over one block's chunks. Chunk k is tagged with id (k % depth) + 1;
// once `depth` chunks are outstanding the oldest is retired with a per-id barrier, so the remaining
// chunks stay on the wire while the retired one is pushed to compute. The whole block is reserved
// up front by the caller, so chunk k lands at a fixed offset and no slot arithmetic is needed.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace layer_norm_tile {

// Issues `count` page reads/writes of one chunk starting at its `rotation`-th tile, so cores with
// different rotations start their bursts on different DRAM banks (bank_stagger).
template <typename Op>
FORCE_INLINE void for_each_rotated(uint32_t count, uint32_t rotation, Op op) {
    for (uint32_t step = 0; step < count; ++step) {
        uint32_t tile = step + rotation;
        tile = tile >= count ? tile - count : tile;
        op(tile);
    }
}

// One chunk of `chunk_tiles` consecutive pages between DRAM/L1 pages and contiguous local L1.
template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void read_chunk(const Accessor& accessor, uint32_t first_page, uint32_t l1_addr, uint32_t rotation) {
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        noc_async_read(accessor.get_noc_addr(first_page + tile), l1_addr + tile * tile_bytes, tile_bytes);
    });
}

template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void write_chunk(const Accessor& accessor, uint32_t first_page, uint32_t l1_addr, uint32_t rotation) {
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        noc_async_write(l1_addr + tile * tile_bytes, accessor.get_noc_addr(first_page + tile), tile_bytes);
    });
}

// Work units (BAND_ROWS, op_design.md R1 / R3). A compute tile-row is filled from two 16-row bands: the top
// half of its tiles (faces 0-1, bytes [0, half)) and the bottom half (faces 2-3, bytes [half, 2 half)).
//   R1 (half_band = false): the unit is a whole tile-row, i.e. both bands of one source tile-row.
//   R3 (half_band = true) : the unit is a 16-row band; a core owns a contiguous run of the valid bands (an
//                           image's all-padding last band, H % 32 in [1, 16], is not counted), paired in order.
// A pair (row r top, row r bottom) is `whole`: one full-tile transfer per tile. Any other pair or a lone band
// moves half tiles. With an odd band count, a run that starts on a bottom band takes it alone first, so the
// rest pairs whole; otherwise the last band is alone. A lone band leaves the bottom half of its compute
// tile unread: rows are independent, so those rows are computed on stale L1 and never written.
struct BandSlot {
    uint32_t page;    // first page of the source tile-row
    uint32_t offset;  // byte offset of the band in each page: 0 or half a tile
};

struct RowSource {
    BandSlot top;
    BandSlot bottom;
    bool has_bottom;
    bool whole;
};

template <bool half_band, uint32_t tensor_w_tiles, uint32_t image_row_tiles, uint32_t image_bands, uint32_t half_bytes>
class RowSources {
public:
    RowSources(uint32_t unit_start, uint32_t unit_count) : remaining_(unit_count) {
        if constexpr (half_band) {
            image_row_ = (unit_start / image_bands) * image_row_tiles;
            within_ = unit_start % image_bands;
            lone_first_ = (unit_count % 2 != 0) && (within_ % 2 != 0);
            if (unit_count % 2 != 0) {
                lone_row_ = lone_first_ ? 0 : compute_rows(unit_count) - 1;
            }
        } else {
            image_row_ = unit_start;
        }
    }

    // Compute tile-row holding the lone band (NO_LONE_ROW: none).
    static constexpr uint32_t NO_LONE_ROW = 0xFFFFFFFF;
    uint32_t lone_row() const { return lone_row_; }

    // Compute tile-rows of a run of `unit_count` units.
    static constexpr uint32_t compute_rows(uint32_t unit_count) {
        return half_band ? (unit_count + 1) / 2 : unit_count;
    }

    RowSource next() {
        RowSource src{};
        if constexpr (half_band) {
            src.top = take();
            src.has_bottom = remaining_ > 0 && !lone_first_;
            lone_first_ = false;
            if (src.has_bottom) {
                src.bottom = take();
            }
            src.whole =
                src.has_bottom && src.top.page == src.bottom.page && src.top.offset == 0 && src.bottom.offset != 0;
        } else {
            src.top = {image_row_ * tensor_w_tiles, 0};
            src.bottom = {src.top.page, half_bytes};
            src.has_bottom = true;
            src.whole = true;
            ++image_row_;
        }
        return src;
    }

private:
    BandSlot take() {
        const BandSlot slot{(image_row_ + within_ / 2) * tensor_w_tiles, (within_ % 2) * half_bytes};
        --remaining_;
        if (++within_ == image_bands) {
            within_ = 0;
            image_row_ += image_row_tiles;
        }
        return slot;
    }

    uint32_t remaining_;
    uint32_t lone_row_ = NO_LONE_ROW;
    uint32_t image_row_ = 0;  // first tile-row of the cursor's image (R1: the next tile-row)
    uint32_t within_ = 0;     // valid band index inside the image
    bool lone_first_ = false;
};

// One chunk of `chunk_tiles` tiles of a compute tile-row starting at tile column `col`.
template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void read_row_chunk(
    const Accessor& accessor, const RowSource& src, uint32_t col, uint32_t l1_addr, uint32_t rotation) {
    if (src.whole) {
        read_chunk<chunk_tiles, tile_bytes>(accessor, src.top.page + col, l1_addr, rotation);
        return;
    }
    constexpr uint32_t half = tile_bytes / 2;
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        const uint32_t l1 = l1_addr + tile * tile_bytes;
        noc_async_read(accessor.get_noc_addr(src.top.page + col + tile, src.top.offset), l1, half);
        if (src.has_bottom) {
            noc_async_read(accessor.get_noc_addr(src.bottom.page + col + tile, src.bottom.offset), l1 + half, half);
        }
    });
}

// Folded lone band (lone_band_fold): source columns 2j / 2j + 1 of the band land in the top / bottom half of
// compute tile j, so a chunk of `chunk_tiles` source columns fills chunk_tiles / 2 tiles. Same reads as the
// unfolded lone band (one half tile per source column), half the compute tiles. `col` is even.
template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void read_folded_chunk(const Accessor& accessor, const RowSource& src, uint32_t col, uint32_t l1_addr) {
    constexpr uint32_t half = tile_bytes / 2;
    for (uint32_t tile = 0; tile < chunk_tiles; ++tile) {
        noc_async_read(
            accessor.get_noc_addr(src.top.page + col + tile, src.top.offset),
            l1_addr + (tile / 2) * tile_bytes + (tile % 2) * half,
            half);
    }
}

template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void write_folded_chunk(const Accessor& accessor, const RowSource& src, uint32_t col, uint32_t l1_addr) {
    constexpr uint32_t half = tile_bytes / 2;
    for (uint32_t tile = 0; tile < chunk_tiles; ++tile) {
        noc_async_write(
            l1_addr + (tile / 2) * tile_bytes + (tile % 2) * half,
            accessor.get_noc_addr(src.top.page + col + tile, src.top.offset),
            half);
    }
}

template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void write_row_chunk(
    const Accessor& accessor, const RowSource& src, uint32_t col, uint32_t l1_addr, uint32_t rotation) {
    if (src.whole) {
        write_chunk<chunk_tiles, tile_bytes>(accessor, src.top.page + col, l1_addr, rotation);
        return;
    }
    constexpr uint32_t half = tile_bytes / 2;
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        const uint32_t l1 = l1_addr + tile * tile_bytes;
        noc_async_write(l1, accessor.get_noc_addr(src.top.page + col + tile, src.top.offset), half);
        if (src.has_bottom) {
            noc_async_write(l1 + half, accessor.get_noc_addr(src.bottom.page + col + tile, src.bottom.offset), half);
        }
    });
}

// gamma/beta multicast block, one per tensor: `group_cols` tile columns per 2 * FACE_BYTES group, column c's 64 B
// entry at 64 * (c % group_cols) and its face-1 row 0 FACE_BYTES above its face-0 row 0. One read of
// FACE_BYTES + 32 B then fills a tile slot's row 0 of faces 0 and 1 (face-0 rows 1-15 are don't-care for the row
// broadcast). TILE: the face-1 read lands there directly. ROW_MAJOR: the [face 0 | face 1] spans land at the
// entries and one local copy per group, shifted by FACE_BYTES - 32 B, moves each face-1 half into place; source
// and copy must not overlap, which bounds the group to 7 columns. Every source read is 64 B-aligned (BH DRAM).
template <bool affine_is_tile, uint32_t tensor_w_tiles, uint32_t group_cols, uint32_t page_elems, uint32_t elem_bytes>
struct AffineBlock {
    static constexpr uint32_t TILE_DIM = 32;
    static constexpr uint32_t FACE_DIM = 16;
    static constexpr uint32_t row_bytes = TILE_DIM * elem_bytes;
    static constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    static constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    static constexpr uint32_t tile_bytes = 4 * face_bytes;
    static constexpr uint32_t group_bytes = 2 * face_bytes;
    static constexpr uint32_t rm_shift = face_bytes - face_row_bytes;
    static constexpr uint32_t tensor_bytes = (tensor_w_tiles + group_cols - 1) / group_cols * group_bytes;
    static_assert(
        group_cols >= 1 && group_cols * row_bytes <= (affine_is_tile ? face_bytes : rm_shift),
        "affine group overlaps its face-1 rows");
    static_assert(affine_is_tile || page_elems % TILE_DIM == 0, "RM affine stick must hold whole tile rows");

    // fn(first column, columns, block offset of the group)
    template <typename Fn>
    static FORCE_INLINE void for_each_group(Fn fn) {
        uint32_t group = 0;
        for (uint32_t col = 0; col < tensor_w_tiles; col += group_cols, group += group_bytes) {
            fn(col, tensor_w_tiles - col < group_cols ? tensor_w_tiles - col : group_cols, group);
        }
    }

    // ROW_MAJOR: a group's entries inside one stick page are contiguous, so they land in one read.
    template <typename Accessor>
    static FORCE_INLINE void issue_reads(const Accessor& accessor, uint32_t block) {
        for_each_group([&](uint32_t first_col, uint32_t cols, uint32_t group) {
            uint32_t entry = block + group;
            if constexpr (affine_is_tile) {
                for (uint32_t col = first_col; col < first_col + cols; ++col, entry += row_bytes) {
                    noc_async_read(accessor.get_noc_addr(col, 0), entry, row_bytes);
                    noc_async_read(accessor.get_noc_addr(col, face_bytes), entry + face_bytes, row_bytes);
                }
            } else {
                constexpr uint32_t page_cols = page_elems / TILE_DIM;
                for (uint32_t col = first_col; col < first_col + cols;) {
                    const uint32_t page_col = page_cols == 1 ? 0 : col % page_cols;
                    uint32_t span_cols = page_cols - page_col;
                    span_cols = span_cols < first_col + cols - col ? span_cols : first_col + cols - col;
                    noc_async_read(
                        accessor.get_noc_addr(page_cols == 1 ? col : col / page_cols, page_col * row_bytes),
                        entry,
                        span_cols * row_bytes);
                    col += span_cols;
                    entry += span_cols * row_bytes;
                }
            }
        });
    }

    // ROW_MAJOR, after the entries landed.
    static FORCE_INLINE void shift_face1_rows(uint32_t block) {
        for_each_group([&](uint32_t, uint32_t cols, uint32_t group) {
            noc_async_read(get_noc_addr(block + group), block + group + rm_shift, cols * row_bytes);
        });
    }

    // Folded operand (lone_band_fold): row 0 of faces 2/3 of slot 2j <- column 2j + 1, so a per-face row broadcast
    // of slot 2j serves folded tile j (top half column 2j, bottom half column 2j + 1). The standard row broadcast
    // never reads faces 2/3, so the unfolded rows are unaffected.
    static FORCE_INLINE void scatter_folded(uint32_t block, uint32_t slot) {
        constexpr uint32_t bottom = 2 * face_bytes;
        for (uint32_t col = 1; col < tensor_w_tiles; col += 2, slot += 2 * tile_bytes) {
            const uint32_t entry = block + (col / group_cols) * group_bytes + (col % group_cols) * row_bytes;
            noc_async_read(get_noc_addr(entry), slot + bottom, face_bytes + face_row_bytes);
        }
    }

    // Row 0 of faces 0/1 of `tensor_w_tiles` consecutive tile slots from `slot`.
    static FORCE_INLINE void scatter(uint32_t block, uint32_t slot) {
        for_each_group([&](uint32_t, uint32_t cols, uint32_t group) {
            const uint64_t entry = get_noc_addr(block + group);
            for (uint32_t col = 0; col < cols; ++col, slot += tile_bytes) {
                noc_async_read(entry + col * row_bytes, slot, face_bytes + face_row_bytes);
            }
        });
    }
};

template <uint32_t depth, typename Issue, typename Retire>
FORCE_INLINE void read_block_chunks(Noc noc, uint32_t num_chunks, Issue issue, Retire retire) {
    static_assert(depth >= 1 && depth <= NOC_MAX_TRANSACTION_ID, "trid ring depth out of the NoC id range");
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t trid = (chunk % depth) + 1;
        if (chunk >= depth) {
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
            retire();
        }
        dataflow_kernel_lib::set_read_trid(noc, trid);
        issue(chunk);
    }
    const uint32_t outstanding = num_chunks < depth ? num_chunks : depth;
    for (uint32_t chunk = num_chunks - outstanding; chunk < num_chunks; ++chunk) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, (chunk % depth) + 1);
        retire();
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);
}

}  // namespace layer_norm_tile
