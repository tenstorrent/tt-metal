// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Regime logical_rows_partial_tile, shared by the g6_clone reader and writer: which byte ranges of
// a TILE page carry logical elements, so a partial page moves only those instead of its padding.

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"

namespace g6_clone {

// At most two (offset, length) byte ranges of one page; length1 == 0 means one range. Offsets are
// face bases and lengths are rounded up to the DRAM alignment on the host, so a range lands at the
// same in-page offset in the CB slot and shares its alignment with the DRAM source.
struct PagePieces {
    uint32_t offset0;
    uint32_t length0;
    uint32_t offset1;
    uint32_t length1;
};

// CT layout from `ct_base`: w_tiles, h_tiles, then the PagePieces of the four page kinds, indexed
// by kind = (tile row is the image's last) | (tile column is the image's last) << 1.
// Walks consecutive page ids, tracking the page's tile column and tile row so no page id is divided.
template <uint32_t ct_base>
class PartialTileWalk {
public:
    static constexpr uint32_t w_tiles = get_compile_time_arg_val(ct_base);
    static constexpr uint32_t h_tiles = get_compile_time_arg_val(ct_base + 1);
    static_assert(w_tiles >= 1 && h_tiles >= 1, "g6_clone: tile counts must be at least 1");

    explicit PartialTileWalk(uint32_t first_page) {
        const uint32_t tile_row = first_page / w_tiles;
        wt_ = first_page - tile_row * w_tiles;
        ht_ = tile_row % h_tiles;
    }

    FORCE_INLINE const PagePieces& pieces() const {
        const uint32_t kind = (ht_ == h_tiles - 1 ? 1u : 0u) | (wt_ == w_tiles - 1 ? 2u : 0u);
        return kinds[kind];
    }

    FORCE_INLINE void advance() {
        if (++wt_ == w_tiles) {
            wt_ = 0;
            if (++ht_ == h_tiles) {
                ht_ = 0;
            }
        }
    }

private:
    template <uint32_t kind>
    static constexpr PagePieces kind_pieces() {
        return {
            get_compile_time_arg_val(ct_base + 2 + 4 * kind),
            get_compile_time_arg_val(ct_base + 3 + 4 * kind),
            get_compile_time_arg_val(ct_base + 4 + 4 * kind),
            get_compile_time_arg_val(ct_base + 5 + 4 * kind)};
    }

    static constexpr PagePieces kinds[4] = {kind_pieces<0>(), kind_pieces<1>(), kind_pieces<2>(), kind_pieces<3>()};

    uint32_t wt_;
    uint32_t ht_;
};

// CT args the walk consumes after ct_base.
constexpr uint32_t PARTIAL_TILE_CT_ARGS = 2 + 4 * 4;

}  // namespace g6_clone
