// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// exact_fp32_add: the same chunk schedule and arguments as compute_add.cpp, but every operand is unpacked straight
// into an fp32 DEST (UnpackToDestFp32 on OWN / IN / IN2, fp32_dest_acc_en) and added on the SFPU, so each add is one
// IEEE fp32 rounding (torch's fp32 a + b) instead of the FPU's add_tiles, which reads fp32 at about TF32. A
// three-input chunk is (OWN + IN) + IN2, as compute_add.cpp. Two tiles per DEST batch (4 fp32 DEST tiles, half sync).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_args.hpp"

using namespace fabric_reduce_scatter;

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(compute_ct::RUN_PAGES);
    constexpr uint32_t PAIR = 2;  // tiles per DEST batch: slots 2 j (sum) and 2 j + 1 (addend)
    const uint32_t n = get_arg_val<uint32_t>(compute_rt::N);
    const uint32_t has_b = get_arg_val<uint32_t>(compute_rt::HAS_B);
    const uint32_t has_c = get_arg_val<uint32_t>(compute_rt::HAS_C);
    const uint32_t n_copy = get_arg_val<uint32_t>(compute_rt::N_COPY);
    if (n == 0 && n_copy == 0) {
        return;
    }
    const uint32_t cb_b = has_b ? cb::IN : cb::IN2;
    const bool three = has_b && has_c;
    compute_kernel_hw_startup(cb::OWN, cb::OUT);
    for (uint32_t i = 0; i < n_copy; ++i) {
        cb_wait_front(cb::OWN, R);
        cb_reserve_back(cb::OUT, R);
        copy_init(cb::OWN);
        for (uint32_t j0 = 0; j0 < R; j0 += PAIR) {
            const uint32_t m = R - j0 < PAIR ? R - j0 : PAIR;
            tile_regs_acquire();
            for (uint32_t j = 0; j < m; ++j) {
                copy_tile(cb::OWN, j0 + j, 2 * j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile(2 * j, cb::OUT);
            }
            tile_regs_release();
        }
        cb_push_back(cb::OUT, R);
        cb_pop_front(cb::OWN, R);
    }
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb::OWN, R);
        cb_wait_front(cb_b, R);
        if (three) {
            cb_wait_front(cb::IN2, R);
        }
        cb_reserve_back(cb::OUT, R);
        for (uint32_t j0 = 0; j0 < R; j0 += PAIR) {
            const uint32_t m = R - j0 < PAIR ? R - j0 : PAIR;
            tile_regs_acquire();
            copy_init(cb::OWN);
            for (uint32_t j = 0; j < m; ++j) {
                copy_tile(cb::OWN, j0 + j, 2 * j);
            }
            copy_init(cb_b);
            for (uint32_t j = 0; j < m; ++j) {
                copy_tile(cb_b, j0 + j, 2 * j + 1);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < m; ++j) {
                add_binary_tile(2 * j, 2 * j + 1, 2 * j);
            }
            if (three) {
                copy_init(cb::IN2);
                for (uint32_t j = 0; j < m; ++j) {
                    copy_tile(cb::IN2, j0 + j, 2 * j + 1);
                }
                add_binary_tile_init();
                for (uint32_t j = 0; j < m; ++j) {
                    add_binary_tile(2 * j, 2 * j + 1, 2 * j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile(2 * j, cb::OUT);
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
