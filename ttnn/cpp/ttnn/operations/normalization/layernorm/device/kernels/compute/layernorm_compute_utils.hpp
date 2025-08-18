// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"

namespace layernorm::compute::utils {
// Transpose a tile that is in a single-tile CB
// and push it back into the CB
// icb: Circular buffer ID
// idst: Destination register through which transpose is performed
inline void transpose_single_tile_cb(uint32_t icb, uint32_t idst = 0) {
    tile_regs_acquire();
    tile_regs_wait();

    cb_wait_front(icb, 1);
    transpose_wh_init_short(icb);
    reconfig_data_format_srca(icb);
    transpose_wh_tile(icb, 0, idst);
    tile_regs_commit();

    cb_pop_front(icb, 1);
    cb_reserve_back(icb, 1);
    pack_reconfig_data_format(icb);
    pack_tile(idst, icb);
    cb_push_back(icb, 1);
    tile_regs_release();
}

}  // namespace layernorm::compute::utils
