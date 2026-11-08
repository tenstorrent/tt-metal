// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The GGUF matmul compute kernels' segment ends (gguf_matmul_compute.cpp, gguf_matmul_tq2_compute.cpp; roles in
// gguf_matmul_program_factory.cpp): c_16 bf16 output tiles, c_17 fp32 partials to other cores, c_18 the partials from
// other cores, c_20 this core's own partial.

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace gguf_matmul {

constexpr uint32_t cb_out = tt::CBIndex::c_16, cb_part = tt::CBIndex::c_17;
constexpr uint32_t cb_remote = tt::CBIndex::c_18, cb_own = tt::CBIndex::c_20;

// Finish: this core's partial into DEST tile 0, the `parts` others' into 1 .. parts, added in fp32, packed bf16. SrcA
// must be configured for c_20's format.
ALWI void finish_segment(const uint32_t parts) {
    copy_init(cb_own);
    tile_regs_acquire();
    cb_wait_front(cb_own, 1);
    copy_tile(cb_own, 0, 0);
    cb_wait_front(cb_remote, parts);
    for (uint32_t p = 0; p < parts; p++) {
        copy_tile(cb_remote, p, 1 + p);
    }
    add_binary_tile_init();
    for (uint32_t p = 0; p < parts; p++) {
        add_binary_tile(0, 1 + p, 0);
    }
    cb_pop_front(cb_own, 1);
    cb_pop_front(cb_remote, parts);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

// After a segment's K work in DEST tile 0 (acquired): a whole N tile packs bf16, a head or tail an fp32 partial.
ALWI void pack_segment(const bool whole) {
    tile_regs_commit();
    const uint32_t cb = whole ? cb_out : cb_part;
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    if (!whole) {
        pack_reconfig_data_format(cb_part);
    }
    pack_tile(0, cb);
    if (!whole) {
        pack_reconfig_data_format(cb_out);
    }
    tile_regs_release();
    cb_push_back(cb, 1);
}

}  // namespace gguf_matmul
