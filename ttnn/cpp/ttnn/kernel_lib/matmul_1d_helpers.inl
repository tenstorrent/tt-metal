// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/cb_helpers.hpp"

/**
 * @file matmul_1d_helpers.inl
 * @brief Implementation of matmul_1d helper function.
 *
 * This file contains the implementation details for the matmul_1d() function.
 * It should only be included by matmul_1d_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    matmul_1d_config::InitUninitMode init_uninit_mode,
    matmul_1d_config::WaitMode wait_mode,
    matmul_1d_config::ReconfigureRegisterDatatypeMode reconfig_mode>
ALWI void matmul_1d(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_1d: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_1d: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_1d: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_1d: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_1d: out_cb must be less than 32");

    // Runtime parameter validation
    ASSERT(Mt > 0);
    ASSERT(Nt > 0);
    ASSERT(Kt > 0);
    ASSERT(batch > 0);
    PACK(ASSERT(get_cb_num_pages(out_cb) >= 1));

    // Data format reconfiguration (applied before init)
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == matmul_1d_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == matmul_1d_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == matmul_1d_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == matmul_1d_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    if constexpr (use_unpack_reconfig) {
        reconfig_data_format_srca(in0_cb);
        reconfig_data_format_srcb(in1_cb);
    }

    if constexpr (use_pack_reconfig) {
        pack_reconfig_data_format(out_cb);
    }

    // Init: mm_init must be called once before matmul_tiles.
    // UninitOnly and Neither are no-ops — there is no mm_uninit in the LLK API.
    if constexpr (
        init_uninit_mode == matmul_1d_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == matmul_1d_config::InitUninitMode::InitOnly) {
        mm_init(in0_cb, in1_cb, out_cb);
    }

    // Main loop: batch × Mt × Nt × Kt.
    // This order must match the CB production order from read_matmul_tiles():
    // for each (b, mt, nt, kt), the reader pushes A[b,mt,kt] then B[b,kt,nt].
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            // WaitUpfront: pre-wait for the full Mt-row block before the Nt loop.
            // in0_cb holds Kt tiles for this (b, mt) row; in1_cb holds Kt*Nt tiles
            // (the full B block for this mt row). All Nt outputs are computed before
            // any input tiles are popped (so the same A tiles can be reused for each nt).
            if constexpr (wait_mode == matmul_1d_config::WaitMode::WaitUpfront) {
                cb_wait_front(in0_cb, Kt);
                cb_wait_front(in1_cb, Kt * Nt);
            }

            for (uint32_t nt = 0; nt < Nt; ++nt) {
                tile_regs_acquire();

                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    // WaitPerTile: wait for one A tile and one B tile per Kt step.
                    // Tile index is always 0 because tiles are popped immediately after use;
                    // the reader re-pushes the same A[mt,kt] tile once per Nt iteration.
                    if constexpr (wait_mode == matmul_1d_config::WaitMode::WaitPerTile) {
                        cb_wait_front(in0_cb, 1);
                        cb_wait_front(in1_cb, 1);
                    }

                    matmul_tiles(in0_cb, in1_cb, 0, 0, 0);

                    if constexpr (wait_mode == matmul_1d_config::WaitMode::WaitPerTile) {
                        cb_pop_front(in0_cb, 1);
                        cb_pop_front(in1_cb, 1);
                    }
                }

                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(out_cb, 1);
                pack_tile(0, out_cb);
                cb_push_back(out_cb, 1);

                tile_regs_release();
            }

            // WaitUpfront: pop the full Mt-row block after all Nt output tiles are packed.
            if constexpr (wait_mode == matmul_1d_config::WaitMode::WaitUpfront) {
                cb_pop_front(in0_cb, Kt);
                cb_pop_front(in1_cb, Kt * Nt);
            }
        }
    }

    // UninitOnly and Neither: no-op — there is no mm_uninit in the LLK API.
}

}  // namespace compute_kernel_lib
