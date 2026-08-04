// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

// Format-aware tile fill. Reserves one tile of cb, fills num_of_elems elements with
// `value` (interpreted per the CB's data format), and pushes the tile.
//
//   Float32 / Int32 / UInt32 -> write `value` as-is (32-bit element).
//   Float16_b (default)      -> write upper 16 bits of `value` (bf16 element).
FORCE_INLINE void fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t num_of_elems = 1024) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    const DataFormat data_format = cb.get_dataformat();
    switch (static_cast<uint32_t>(data_format) & 0x1F) {
        case static_cast<uint8_t>(DataFormat::Float32):
        case static_cast<uint8_t>(DataFormat::Int32):
        case static_cast<uint8_t>(DataFormat::UInt32): {
            CoreLocalMem<volatile uint32_t> ptr(cb.get_write_ptr());
            for (int32_t j = 0; j < num_of_elems; ++j) {
                ptr[j] = value;
            }
            break;
        }
        case static_cast<uint8_t>(DataFormat::Float16_b):
        default: {
            CoreLocalMem<volatile uint16_t> ptr(cb.get_write_ptr());
            const uint16_t packed = static_cast<uint16_t>(value >> 16);
            for (int32_t j = 0; j < num_of_elems; ++j) {
                ptr[j] = packed;
            }
            break;
        }
    }
    cb.push_back(1);
}
