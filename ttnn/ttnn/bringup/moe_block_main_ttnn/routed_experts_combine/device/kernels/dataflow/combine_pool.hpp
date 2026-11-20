// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The landing pool on the flat cells (CombineLanding::FlatCores): the L1 the flat expert leaves free on its cells
// (FlatRoutedExpertPlan::free_l1), described by runtime args at a0 (combine_program.cpp pool_args): [words, classes,
// then a class: cells | spans << 8 | slots a cell << 16, its spans (begin / 64 | slots << 16), its cells (x | y << 8,
// two a word)]. Pool slot g: the class's cell g % cells, the (g / cells)-th slot over that cell's spans - consecutive
// slots on different cells.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

inline uint64_t rec_pool_noc(uint32_t a0, uint32_t area, uint32_t slot_bytes, uint32_t g) {
    const uint32_t nc = get_arg_val<uint32_t>(a0 + 1);
    uint32_t i = a0 + 2;
    for (uint32_t c = 0; c < nc; ++c) {
        const uint32_t h = get_arg_val<uint32_t>(i);
        const uint32_t cells = h & 0xFF, ns = (h >> 8) & 0xFF, spc = h >> 16;
        if (g < cells * spc) {
            const uint32_t cell = g % cells;
            uint32_t j = g / cells;
            for (uint32_t s = 0; s < ns; ++s) {
                const uint32_t w = get_arg_val<uint32_t>(i + 1 + s), n = w >> 16;
                if (j < n) {
                    const uint32_t xy = (get_arg_val<uint32_t>(i + 1 + ns + cell / 2) >> ((cell & 1) * 16)) & 0xFFFF;
                    return get_noc_addr(xy & 0xFF, xy >> 8, area + (w & 0xFFFF) * 64 + j * slot_bytes);
                }
                j -= n;
            }
        }
        g -= cells * spc;
        i += 1 + ns + (cells + 1) / 2;
    }
    return 0;
}
