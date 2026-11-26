// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement helpers shared by the qkv_causal_conv1d_silu kernels.
#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

namespace qkv_conv {

static_assert(kda_chronology::history_rows == history_rows);

// The first word of `accessor`'s page 0, read through the aligned L1 slot `slot`.
template <typename Accessor>
inline uint32_t read_word(Noc& noc, const Accessor& accessor, uint32_t slot) {
    noc.async_read(accessor, CoreLocalMem<uint32_t>(slot), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot)[0];
}

// A convolution tail (`history`, `final_history`, or a bare predecessor carry) holds `history_rows` rows of every
// channel in one of two layouts; `page_channels` is the channels one page holds. Token-paged: page `r` is row `r`, all
// channels. Channel-paged: page `c / page_channels` is the `history_rows` rows of its `page_channels` channels, row
// after row. The page holding tail row `row` at channel tile `ct`, and that tile row's byte offset in the page.
struct TailLocation {
    uint32_t page;
    uint32_t offset;
};

template <uint32_t token_paged, uint32_t page_channels>
inline TailLocation tail_location(uint32_t row, uint32_t ct) {
    if constexpr (token_paged) {
        return {row, ct * tile_row_bytes};
    } else {
        constexpr uint32_t page_tiles = page_channels / tile_width;
        return {ct / page_tiles, row * page_channels * element_bytes + ct % page_tiles * tile_row_bytes};
    }
}

// The channel tiles one page holds of a tail row whose channel tiles end at `row_end`: the whole row when token-paged.
template <uint32_t token_paged, uint32_t page_channels>
inline uint32_t tail_page_tiles(uint32_t row_end) {
    if constexpr (token_paged) {
        return row_end;
    } else {
        return page_channels / tile_width;
    }
}

// Row `row` of a convolution tail's channel tiles [ct_start, ct_start + ct_count) into `destination` at byte
// `destination_offset`: one read per page the range touches, the tail's pages from `page_base`.
template <uint32_t token_paged, uint32_t page_channels, typename Tail, typename Destination>
inline void read_tail_row(
    Noc& noc,
    const Tail& tail,
    const Destination& destination,
    uint32_t row,
    uint32_t ct_start,
    uint32_t ct_count,
    uint32_t destination_offset,
    uint32_t page_base = 0) {
    const uint32_t ct_end = ct_start + ct_count;
    const uint32_t page_tiles = tail_page_tiles<token_paged, page_channels>(ct_end);
    for (uint32_t ct = ct_start; ct < ct_end;) {
        const uint32_t to_page_end = page_tiles - ct % page_tiles;
        const uint32_t count = to_page_end < ct_end - ct ? to_page_end : ct_end - ct;
        const auto [page, offset] = tail_location<token_paged, page_channels>(row, ct);
        noc.async_read(
            tail,
            destination,
            count * tile_row_bytes,
            {.page_id = page_base + page, .offset_bytes = offset},
            {.offset_bytes = destination_offset + (ct - ct_start) * tile_row_bytes});
        ct += count;
    }
}

// write_tail's page map: the tail page a full tail's page goes to, or no_page to skip it.
constexpr uint32_t no_page = ~0U;
struct SamePage {
    uint32_t operator()(uint32_t page) const { return page; }
};

// The mirror for a whole tail: its `history_rows` rows of `channel_tiles` tiles, row after row in local `scratch` at
// byte `rows_offset`, into `tail`, one write per page. Token-paged pages are those rows. Channel-paged pages are first
// gathered, through local reads, into as many bytes of `scratch` at `pages_offset`. Page p is written to tail page
// page_base + page_map(p), or skipped where page_map returns no_page (an SP-sharded slab stores some pages only).
template <uint32_t token_paged, uint32_t page_channels, typename Scratch, typename Tail, typename PageMap = SamePage>
inline void write_tail(
    Noc& noc,
    const Scratch& scratch,
    const Tail& tail,
    uint32_t channel_tiles,
    uint32_t rows_offset,
    uint32_t pages_offset,
    uint32_t page_base = 0,
    const PageMap& page_map = {}) {
    const uint32_t row_bytes = channel_tiles * tile_row_bytes;
    const uint32_t page_tiles = tail_page_tiles<token_paged, page_channels>(channel_tiles);
    const uint32_t pages = token_paged ? history_rows : channel_tiles / page_tiles;
    const uint32_t page_bytes = history_rows * row_bytes / pages;
    uint32_t pages_source = rows_offset;
    if constexpr (!token_paged) {
        for (uint32_t row = 0; row < history_rows; ++row) {
            for (uint32_t ct = 0; ct < channel_tiles; ct += page_tiles) {
                const auto [page, offset] = tail_location<token_paged, page_channels>(row, ct);
                noc.async_read(
                    scratch,
                    scratch,
                    page_tiles * tile_row_bytes,
                    {.offset_bytes = rows_offset + row * row_bytes + ct * tile_row_bytes},
                    {.offset_bytes = pages_offset + page * page_bytes + offset});
            }
        }
        noc.async_read_barrier();
        pages_source = pages_offset;
    }
    for (uint32_t page = 0; page < pages; ++page) {
        const uint32_t target = page_map(page);
        if (target == no_page) {
            continue;
        }
        noc.async_write(
            scratch,
            tail,
            page_bytes,
            {.offset_bytes = pages_source + page * page_bytes},
            {.page_id = page_base + target});
    }
}

// A history slab's carry: its first page, whether it reads as zeros (actual_start == 0 or past the slab), and whether
// the replacement is written (not past the slab). Scalars land in the aligned L1 `slot`.
struct SlabCarry {
    uint32_t page_base;
    bool zeros;
    bool writes;
};

template <uint32_t slab, uint32_t slab_index_limit, uint32_t slab_scale, uint32_t slab_offset, uint32_t entry_pages>
inline SlabCarry read_slab_carry(Noc& noc, uint32_t slot) {
    if constexpr (!slab) {
        return {0, false, true};
    } else {
        const uint32_t start = read_word(noc, TensorAccessor(tensor::actual_start), slot);
        const uint32_t index = read_word(noc, TensorAccessor(*tensor::get_token_if_present<"slab_index">()), slot);
        const bool in_slab = index < slab_index_limit;
        return {in_slab ? (index * slab_scale + slab_offset) * entry_pages : 0, start == 0 || !in_slab, in_slab};
    }
}

// This rank's chronology over [actual_start, actual_end), or the full capacity without actual_end.
template <uint32_t has_actual_end, uint32_t sp_size, uint32_t local_rows>
inline kda_chronology::Topology read_chronology(Noc& noc, uint32_t slot, uint32_t sp_rank) {
    const uint32_t start = read_word(noc, TensorAccessor(tensor::actual_start), slot);
    if constexpr (has_actual_end) {
        const uint32_t end = read_word(noc, TensorAccessor(*tensor::get_token_if_present<"actual_end">()), slot);
        return kda_chronology::derive_interval(start, end, sp_rank, sp_size, local_rows);
    } else {
        return kda_chronology::derive(start, sp_rank, sp_size, local_rows);
    }
}

}  // namespace qkv_conv
