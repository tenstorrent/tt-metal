// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// A core's items are the input's items wi = first + k * stride, k < count (wi = (b * H + h) * Mt + mt). The core runs
// those whose gate still needs its sigmoid (heads [Ha, H)) first, then those whose gate arrives activated (heads
// [0, Ha)), each in wi order. With Ha = 0 the sequence is the core's items in wi order.
template <uint32_t H, uint32_t Ha, uint32_t Mt, uint32_t max_items>
struct NormItems {
    struct Item {
        uint32_t b;
        uint32_t h;
        uint32_t mt;
        uint32_t wi() const { return (b * H + h) * Mt + mt; }
    };

    uint32_t wi_[max_items > 0 ? max_items : 1];
    uint32_t count = 0;
    uint32_t raw_count = 0;

    static bool activated(uint32_t wi) { return wi / Mt % H < Ha; }

    NormItems(uint32_t first, uint32_t stride, uint32_t n) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t wi = first + k * stride;
            if (!activated(wi)) {
                wi_[count++] = wi;
            }
        }
        raw_count = count;
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t wi = first + k * stride;
            if (activated(wi)) {
                wi_[count++] = wi;
            }
        }
    }

    Item operator[](uint32_t k) const {
        const uint32_t bh = wi_[k] / Mt;
        return {bh / H, bh % H, wi_[k] % Mt};
    }

    // How many of the group [first, first + items) still need a sigmoid: they lead the group.
    uint32_t raw_in(uint32_t first, uint32_t items) const {
        if (first >= raw_count) {
            return 0;
        }
        return raw_count - first < items ? raw_count - first : items;
    }
};
