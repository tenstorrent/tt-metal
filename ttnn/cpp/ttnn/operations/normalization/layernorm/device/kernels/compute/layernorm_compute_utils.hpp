// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
namespace layernorm::compute::utils {

ALWI void MATH_ACQUIRE(const char* msg, uint32_t j, uint32_t wt, uint32_t ncht) {
    MATH(DPRINT << "Math acquiring at: " << msg << " " << j << " " << wt << " " << ncht << ENDL());
    tile_regs_acquire();
}

ALWI void MATH_COMMIT(const char* msg, uint32_t j, uint32_t wt, uint32_t ncht) {
    MATH(DPRINT << "Math committing after: " << msg << " " << j << " " << wt << " " << ncht << ENDL());
    tile_regs_commit();
}

ALWI void PACK_WAIT(const char* msg, uint32_t j, uint32_t wt, uint32_t ncht) {
    PACK(DPRINT << "Pack waiting at: " << msg << " " << j << " " << wt << " " << ncht << ENDL());
    tile_regs_wait();
}

ALWI void PACK_RELEASE(const char* msg, uint32_t j, uint32_t wt, uint32_t ncht) {
    PACK(DPRINT << "Pack releasing after: " << msg << " " << j << " " << wt << " " << ncht << ENDL());
    tile_regs_release();
}

// Transpose and pack mean and variance from regs to CBs
ALWI void transpose_pack_mean_and_variance(uint32_t cb_ex, uint32_t cb_ex2, uint32_t ex_dst, uint32_t ex2_dst) {
    constexpr auto onetile = 1;

    tile_regs_acquire();
    tile_regs_wait();

    cb_reserve_back(cb_ex, onetile);
    cb_reserve_back(cb_ex2, onetile);

    pack_reconfig_data_format(cb_ex);
    pack_tile(ex_dst, cb_ex);
    pack_reconfig_data_format(cb_ex2);
    pack_tile(ex2_dst, cb_ex2);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_ex, onetile);
    cb_push_back(cb_ex2, onetile);

    cb_wait_front(cb_ex, onetile);
    cb_wait_front(cb_ex2, onetile);
    transpose_wh_init_short(cb_ex);
    reconfig_data_format_srca(cb_ex);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_tile(cb_ex, 0, ex_dst);

    tile_regs_commit();
    transpose_wh_init_short(cb_ex2);
    reconfig_data_format_srca(cb_ex2);
    tile_regs_acquire();
    transpose_wh_tile(cb_ex2, 0, ex2_dst);

    tile_regs_commit();

    cb_pop_front(cb_ex, onetile);
    cb_pop_front(cb_ex2, onetile);

    cb_reserve_back(cb_ex, onetile);
    cb_reserve_back(cb_ex2, onetile);
    pack_reconfig_data_format(cb_ex);

    pack_tile(ex_dst, cb_ex);
    pack_reconfig_data_format(cb_ex2);
    pack_tile(ex2_dst, cb_ex2);

    tile_regs_release();

    cb_push_back(cb_ex, onetile);
    cb_push_back(cb_ex2, onetile);
}

}  // namespace layernorm::compute::utils
