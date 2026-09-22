// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// A virtual per-head concatenation. Compute sees the same tile order as dense
// attention; segment boundaries change addresses, never softmax-state lifetime.
template <uint32_t PrimaryPages, uint32_t JointPages, typename Primary, typename Joint>
struct SequenceAccessor {
    Primary primary;
    Joint joint;

    template <typename Function>
    FORCE_INLINE void visit(uint32_t page, Function&& function) const {
        if constexpr (JointPages == 0) {
            function(primary, page);
        } else {
            constexpr uint32_t pages_per_head = PrimaryPages + JointPages;
            const uint32_t head = page / pages_per_head;
            const uint32_t offset = page % pages_per_head;
            if (offset < PrimaryPages) {
                function(primary, head * PrimaryPages + offset);
            } else {
                function(joint, head * JointPages + offset - PrimaryPages);
            }
        }
    }
};

template <uint32_t PrimaryPages, uint32_t JointPages, typename Primary, typename Joint>
auto sequence_accessor(Primary primary, Joint joint) {
    return SequenceAccessor<PrimaryPages, JointPages, Primary, Joint>{primary, joint};
}

template <typename Primary>
auto sequence_accessor(Primary primary) {
    return SequenceAccessor<0, 0, Primary, Primary>{primary, primary};
}
