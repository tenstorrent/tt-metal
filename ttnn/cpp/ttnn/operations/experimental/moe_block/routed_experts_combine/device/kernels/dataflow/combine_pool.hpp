// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The landing pool on the flat expert cores (CombineLanding::FlatCores): the L1 the flat expert leaves free on its
// cores (FlatRoutedExpertPlan::free_l1), described by runtime args at a0 (combine_program.cpp pool_args): [words,
// classes, then a class (cores with the same free spans): cores | spans << 8 | slots a core << 16, its spans
// (begin / 64 | slots << 16), its cores (x | y << 8, two a word)]. Pool slot g: the class's core g % cores, the
// (g / cores)-th slot over that core's spans - consecutive slots on different cores.
#pragma once
#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "api/dataflow/dataflow_api.h"

inline uint64_t rec_pool_noc(uint32_t a0, uint32_t area, uint32_t slot_bytes, uint32_t g) {
    const uint32_t nc = moe_block_args::arg(a0 + 1);
    uint32_t i = a0 + 2;
    for (uint32_t c = 0; c < nc; ++c) {
        const uint32_t h = moe_block_args::arg(i);
        const uint32_t cores = h & 0xFF, ns = (h >> 8) & 0xFF, spc = h >> 16;
        if (g < cores * spc) {
            const uint32_t core = g % cores;
            uint32_t j = g / cores;
            for (uint32_t s = 0; s < ns; ++s) {
                const uint32_t w = moe_block_args::arg(i + 1 + s), n = w >> 16;
                if (j < n) {
                    const uint32_t xy = (moe_block_args::arg(i + 1 + ns + core / 2) >> ((core & 1) * 16)) & 0xFFFF;
                    return get_noc_addr(xy & 0xFF, xy >> 8, area + (w & 0xFFFF) * 64 + j * slot_bytes);
                }
                j -= n;
            }
        }
        g -= cores * spc;
        i += 1 + ns + (cores + 1) / 2;
    }
    return 0;
}
