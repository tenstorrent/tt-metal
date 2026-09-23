// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// A virtual per-head concatenation of tile-padded segments. Segment boundaries
// change addresses and padding masks, never softmax-state lifetime.
template <uint32_t PrimaryRows, uint32_t JointRows, uint32_t ChunkRows, typename Primary, typename Joint>
struct SequenceAccessor {
    static constexpr bool has_partial_rows = PrimaryRows % 32 != 0 || JointRows % 32 != 0;
    static constexpr uint32_t primary_rows = ((PrimaryRows + 31) / 32) * 32;
    static constexpr uint32_t joint_rows = ((JointRows + 31) / 32) * 32;
    static constexpr uint32_t head_pages = ((primary_rows + joint_rows + ChunkRows - 1) / ChunkRows) * ChunkRows / 8;
    Primary primary;
    Joint joint;

    FORCE_INLINE uint32_t valid_rows(uint32_t page) const {
        const uint32_t row = (page % head_pages) / 4 * 32;
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
            if (offset < primary_rows / 8) {
                function(primary, head * primary_rows / 8 + offset);
            } else if (offset < (primary_rows + joint_rows) / 8) {
                function(joint, head * joint_rows / 8 + offset - primary_rows / 8);
            } else {
                return false;
            }
        }
        return true;
    }
};

template <uint32_t PrimaryRows, uint32_t JointRows, uint32_t ChunkRows, typename Primary, typename Joint>
auto sequence_accessor(Primary primary, Joint joint) {
    return SequenceAccessor<PrimaryRows, JointRows, ChunkRows, Primary, Joint>{primary, joint};
}

template <uint32_t PrimaryRows, uint32_t ChunkRows, typename Primary>
auto sequence_accessor(Primary primary) {
    return sequence_accessor<PrimaryRows, 0, ChunkRows>(primary, primary);
}
