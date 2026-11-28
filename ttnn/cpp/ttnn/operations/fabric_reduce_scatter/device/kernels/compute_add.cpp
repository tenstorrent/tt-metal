// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// run_pages tiles per chunk. First n_copy chunks: OUT = OWN (a ring port's leading block: it starts that block's
// chain). Then n chunks: OUT = OWN + IN [+ IN2], or OWN + IN2 (a final core without a forward upstream).
// Nothing (n_copy = n = 0): a line end's port, whose reader feeds the sender directly.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_args.hpp"

using namespace fabric_reduce_scatter;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(compute_ct::RUN_PAGES);
    constexpr uint32_t DST = 4;  // tiles per DEST batch
    const uint32_t n = get_arg_val<uint32_t>(compute_rt::N);
    const uint32_t has_b = get_arg_val<uint32_t>(compute_rt::HAS_B);  // second input IN
    const uint32_t has_c = get_arg_val<uint32_t>(compute_rt::HAS_C);  // third input IN2
    const uint32_t n_copy = get_arg_val<uint32_t>(compute_rt::N_COPY);
    if (n == 0 && n_copy == 0) {
        return;
    }
    const uint32_t cb_b = has_b ? cb::IN : cb::IN2;
    const bool three = has_b && has_c;
    compute_kernel_hw_startup(cb::OWN, cb_b, cb::OUT);
    if (n_copy > 0) {
        copy_init(cb::OWN);
        for (uint32_t i = 0; i < n_copy; ++i) {
            cb_wait_front(cb::OWN, R);
            cb_reserve_back(cb::OUT, R);
            for (uint32_t j0 = 0; j0 < R; j0 += DST) {
                const uint32_t m = R - j0 < DST ? R - j0 : DST;
                tile_regs_acquire();
                for (uint32_t j = 0; j < m; ++j) {
                    copy_tile(cb::OWN, j0 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < m; ++j) {
                    pack_tile(j, cb::OUT);
                }
                tile_regs_release();
            }
            cb_push_back(cb::OUT, R);
            cb_pop_front(cb::OWN, R);
        }
    }
    add_init(cb::OWN, cb_b);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb::OWN, R);
        cb_wait_front(cb_b, R);
        if (three) {
            cb_wait_front(cb::IN2, R);
        }
        cb_reserve_back(cb::OUT, R);
        for (uint32_t j0 = 0; j0 < R; j0 += DST) {
            const uint32_t m = R - j0 < DST ? R - j0 : DST;
            tile_regs_acquire();
            if (three) {
                add_init(cb::OWN, cb_b);
            }
            for (uint32_t j = 0; j < m; ++j) {
                add_tiles(cb::OWN, cb_b, j0 + j, j0 + j, j);
            }
            if (three) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb::IN2);
                for (uint32_t j = 0; j < m; ++j) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb::IN2, j0 + j, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile(j, cb::OUT);
            }
            tile_regs_release();
        }
        cb_push_back(cb::OUT, R);
        cb_pop_front(cb::OWN, R);
        cb_pop_front(cb_b, R);
        if (three) {
            cb_pop_front(cb::IN2, R);
        }
    }
}
