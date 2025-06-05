// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val, bool unconditionally = true) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = val | (val << 16);
    if (ptr[0] != value || unconditionally) {
        for (uint32_t i = 0; i < n / 2; ++i) {
            ptr[i] = (value);
        }
    }

    return true;
}
