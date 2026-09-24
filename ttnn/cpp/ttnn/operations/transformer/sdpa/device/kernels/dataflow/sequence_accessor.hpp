// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// A virtual per-head concatenation of tile-padded segments. Segment boundaries
// change addresses and padding masks, never softmax-state lifetime.
// Pages are 32x32 tiles, DHt per 32-row sequence tile.
template <uint32_t PrimaryRows, uint32_t JointRows, uint32_t ChunkRows, typename Primary, typename Joint, uint32_t DHt = 4>
struct SequenceAccessor {
    static constexpr bool has_partial_rows = PrimaryRows % 32 != 0 || JointRows % 32 != 0;
    static constexpr uint32_t primary_rows = ((PrimaryRows + 31) / 32) * 32;
    static constexpr uint32_t joint_rows = ((JointRows + 31) / 32) * 32;
    static constexpr uint32_t rows_per_page = 32 / DHt;
    static constexpr uint32_t head_pages =
        ((primary_rows + joint_rows + ChunkRows - 1) / ChunkRows) * ChunkRows / rows_per_page;
    Primary primary;
    Joint joint;

    FORCE_INLINE uint32_t valid_rows(uint32_t page) const {
        const uint32_t row = (page % head_pages) / DHt * 32;
        const uint32_t remaining = row < primary_rows                ? PrimaryRows - row
                                   : row < primary_rows + joint_rows ? JointRows - (row - primary_rows)
                                                                     : 0;
        return remaining < 32 ? remaining : 32;
    }

    template <typename Function>
    FORCE_INLINE bool visit(uint32_t page, Function&& function) const {
        if constexpr (JointRows == 0 && PrimaryRows % ChunkRows == 0) {
            function(primary, page);
        } else {
            const uint32_t head = page / head_pages;
            const uint32_t offset = page % head_pages;
            if (offset < primary_rows / rows_per_page) {
                function(primary, head * primary_rows / rows_per_page + offset);
            } else if (offset < (primary_rows + joint_rows) / rows_per_page) {
                function(joint, head * joint_rows / rows_per_page + offset - primary_rows / rows_per_page);
            } else {
                return false;
            }
        }
        return true;
    }
};

template <uint32_t PrimaryRows, uint32_t JointRows, uint32_t ChunkRows, uint32_t DHt = 4, typename Primary, typename Joint>
auto sequence_accessor(Primary primary, Joint joint) {
    return SequenceAccessor<PrimaryRows, JointRows, ChunkRows, Primary, Joint, DHt>{primary, joint};
}

template <uint32_t PrimaryRows, uint32_t ChunkRows, uint32_t DHt = 4, typename Primary>
auto sequence_accessor(Primary primary) {
    return sequence_accessor<PrimaryRows, 0, ChunkRows, DHt>(primary, primary);
}
