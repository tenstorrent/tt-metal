// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"

// Tracks a logical sequence row within a page-bundle table. The table contains
// uint16 physical bundle ids; layer and head offsets are applied by the caller.
struct PagedKVBundleCursor {
    uint32_t bundle_ids_l1_addr = 0;
    uint32_t page_size_rows = 1;
    uint32_t logical_bundle = 0;
    uint32_t row_in_bundle = 0;

    void reset(uint32_t table_l1_addr, uint32_t logical_row, uint32_t rows_per_page) {
        bundle_ids_l1_addr = table_l1_addr;
        page_size_rows = rows_per_page;
        logical_bundle = logical_row / rows_per_page;
        row_in_bundle = logical_row % rows_per_page;
    }

    uint32_t physical_bundle() const { return CoreLocalMem<volatile uint16_t>(bundle_ids_l1_addr)[logical_bundle]; }

    // Returns true when advancing enters the next logical bundle.
    bool advance_row() {
        if (++row_in_bundle == page_size_rows) {
            row_in_bundle = 0;
            ++logical_bundle;
            return true;
        }
        return false;
    }
};
