// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"

// W-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
template <typename DFB>
FORCE_INLINE void generate_bcast_col_scalar(DFB& dfb, uint32_t scalar) {
    const uint16_t scalar_val = scalar >> 16;
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb.get_write_ptr());
    for (int k = 0; k < 4; k += 2) {
        uint32_t idx = k << 8;
        for (int j = 0; j < 256; j += 16) {
            ptr[idx + j] = scalar_val;
        }
    }
    dfb.push_back(1);
}

// H-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
template <typename DFB>
FORCE_INLINE void generate_bcast_row_scalar(DFB& dfb, uint32_t scalar) {
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    for (int k = 0; k < 2; ++k) {
        uint32_t idx = k << 7;
        for (int j = 0; j < 8; ++j) {
            ptr[idx + j] = scalar;
        }
    }
    dfb.push_back(1);
}

// HW-bcast scalar
// A 16-bit tile keeps the upper half of the scalar, which the caller double packs (the packed low
// half is never read). A 32-bit tile keeps the whole word, so a float32 scalar keeps every mantissa
// bit. Element 0 is the only datum the HW-broadcast reads, so one word is enough for either width.
template <typename DFB>
FORCE_INLINE void generate_bcast_unary_scalar(DFB& dfb, uint32_t scalar) {
    const DataFormat data_format = dfb.get_dataformat();
    uint32_t scalar_val = scalar >> 16;
    switch (static_cast<uint32_t>(data_format) & 0x1Fu) {
        case static_cast<uint32_t>(DataFormat::Float32):
        case static_cast<uint32_t>(DataFormat::Int32):
#ifndef ARCH_QUASAR  // Quasar has no UInt32 data format
        case static_cast<uint32_t>(DataFormat::UInt32):
#endif
            scalar_val = scalar;
            break;
        default: break;
    }
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    ptr[0] = scalar_val;
    dfb.push_back(1);
}
