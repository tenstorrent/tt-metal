// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <api/dataflow/dataflow_api.h>
#include <cstdint>
#include <tt-metalium/constants.hpp>

#define RM_DF_ALWI inline __attribute__((always_inline))

// Same packing convention as pool_kernels_common fill_with_val: repeat uint16 `val` for n uint16 slots.
RM_DF_ALWI void rm_fill_with_val_bf16(uint32_t begin_addr, uint32_t num_u16, uint16_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = static_cast<uint32_t>(val) | (static_cast<uint32_t>(val) << 16);
    uint32_t num_pairs = num_u16 / 2;
    for (uint32_t i = 0; i < num_pairs; ++i) {
        ptr[i] = value;
    }
    if (num_u16 & 1) {
        volatile tt_l1_ptr uint16_t* ptr16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
        ptr16[num_u16 - 1] = val;
    }
}

RM_DF_ALWI void rm_fill_buffer_with_identity_pattern(
    uint32_t begin_addr, uint32_t num_bytes, uint32_t elem_bytes, uint32_t pattern_bits) {
    if (num_bytes == 0) {
        return;
    }
    if (elem_bytes == 2) {
        const uint16_t v = static_cast<uint16_t>(pattern_bits & 0xFFFFu);
        rm_fill_with_val_bf16(begin_addr, num_bytes / 2, v);
    } else if (elem_bytes == 4) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
        const uint32_t n = num_bytes / 4;
        for (uint32_t i = 0; i < n; ++i) {
            p[i] = pattern_bits;
        }
    }
}
