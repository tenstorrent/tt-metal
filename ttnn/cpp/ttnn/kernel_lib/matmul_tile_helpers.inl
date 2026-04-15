// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file matmul_tile_helpers.inl
 * @brief Implementation of matmul_tile helper function.
 *
 * Wraps mm_init + matmul_tiles LLK for tile-by-tile matrix multiplication.
 * Caller must call mm_init before invoking this helper.
 * This file should only be included by matmul_tile_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    bool transpose,
    typename PostComputeFn>
ALWI void matmul_tile(
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t batch,
    PostComputeFn post_compute) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_tile: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_tile: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_tile: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_tile: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_tile: out_cb must be less than 32");

    // Runtime validation
    ASSERT(Mt > 0);
    ASSERT(Nt > 0);
    ASSERT(Kt > 0);
    ASSERT(batch > 0);
    PACK(ASSERT(get_cb_num_pages(out_cb) >= 1));

    constexpr uint32_t onetile = 1;

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                tile_regs_acquire();

                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    cb_wait_front(in0_cb, onetile);
                    cb_wait_front(in1_cb, onetile);

                    ckernel::matmul_tiles(in0_cb, in1_cb, 0, 0, 0);

                    cb_pop_front(in0_cb, onetile);
                    cb_pop_front(in1_cb, onetile);
                }

                // PostComputeFn fires after Kt accumulation, before packing
                post_compute(onetile);

                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(out_cb, onetile);
                pack_tile(0, out_cb);
                cb_push_back(out_cb, onetile);

                tile_regs_release();
            }
        }
    }
}

}  // namespace compute_kernel_lib
