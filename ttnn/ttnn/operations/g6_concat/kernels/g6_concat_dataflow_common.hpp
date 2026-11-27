// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pieces shared by the g6_concat readers and writer: the output-page cursor that yields the valid
// row count of each page, the row-trimmed transfer layout, and the per-input accessor that is
// rebased (base address + buffer type) when the walk moves to another input.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace g6_concat {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

// Tile row R and tile column c of the current output page, advanced page by page so the
// valid-row count v(p) never needs a division inside a block.
struct OutputPageCursor {
    uint32_t out_Wt;
    uint32_t out_Ht;
    uint32_t v_last;
    uint32_t R;
    uint32_t c;

    FORCE_INLINE void init(uint32_t start_page) {
        c = start_page % out_Wt;
        R = (start_page / out_Wt) % out_Ht;
    }

    FORCE_INLINE uint32_t valid_rows() const { return R == out_Ht - 1 ? v_last : TILE_ROWS; }

    FORCE_INLINE void advance() {
        if (++c == out_Wt) {
            c = 0;
            if (++R == out_Ht) {
                R = 0;
            }
        }
    }
};

// Rows [0, v) of a tile, v <= FACE_ROWS, live in faces 0 and 1 at the same in-face offsets:
// v*rb bytes at 0 and v*rb bytes at face_bytes. They merge into one range when v == FACE_ROWS.
// The callback receives (byte offset inside the tile, byte count).
template <typename Fn>
FORCE_INLINE void for_each_leading_rows_range(uint32_t v, uint32_t rb, uint32_t face_bytes, Fn&& fn) {
    const uint32_t bytes = v * rb;
    if (bytes == face_bytes) {
        fn(0, 2 * face_bytes);
    } else {
        fn(0, bytes);
        fn(face_bytes, bytes);
    }
}

// One input of the concatenation as an interleaved page source. The accessor's buffer type is a
// template parameter of TensorAccessor, so the reader keeps one DRAM and one L1 accessor-args set
// and rebases the matching one onto input k's base address. The per-input buffer type comes from
// a compile-time bit mask (K <= 64, two words).
template <typename DramArgsT, typename L1ArgsT, uint32_t l1_mask_lo, uint32_t l1_mask_hi>
class InputPages {
public:
    static constexpr bool any_l1 = (l1_mask_lo | l1_mask_hi) != 0;

    InputPages(const DramArgsT& dram_args, const L1ArgsT& l1_args, uint32_t page_bytes) :
        dram_args_(dram_args), l1_args_(l1_args), page_bytes_(page_bytes) {}

    static FORCE_INLINE bool is_l1(uint32_t k) {
        if constexpr (!any_l1) {
            return false;
        } else {
            return ((k < 32 ? (l1_mask_lo >> k) : (l1_mask_hi >> (k - 32))) & 1u) != 0;
        }
    }

    // NoC read alignment of input k's buffer: source and destination must agree modulo this.
    static FORCE_INLINE uint32_t read_alignment(uint32_t k) {
        return is_l1(k) ? NOC_L1_READ_ALIGNMENT_BYTES : NOC_DRAM_READ_ALIGNMENT_BYTES;
    }

    FORCE_INLINE void rebase(uint32_t k, uint32_t base_addr) {
        base_ = base_addr;
        cur_is_l1_ = is_l1(k);
    }

    FORCE_INLINE uint64_t noc_addr(uint32_t page_id, uint32_t offset) const {
        if constexpr (any_l1) {
            if (cur_is_l1_) {
                return TensorAccessor(l1_args_, base_, page_bytes_).get_noc_addr(page_id, offset);
            }
        }
        return TensorAccessor(dram_args_, base_, page_bytes_).get_noc_addr(page_id, offset);
    }

private:
    DramArgsT dram_args_;
    L1ArgsT l1_args_;
    uint32_t page_bytes_;
    uint32_t base_ = 0;
    bool cur_is_l1_ = false;
};

}  // namespace g6_concat
