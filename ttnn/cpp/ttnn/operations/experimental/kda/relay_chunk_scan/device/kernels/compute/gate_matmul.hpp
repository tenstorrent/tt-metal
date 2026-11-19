// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

namespace kda_gate {

// One K step of a 1 x `columns` subblock at `fidelity`, which may differ from the kernel's MATH_FIDELITY: the relay
// runs at HiFi3, the output gate at the input projection's fidelity. Throttled like matmul_block (the Blackhole di/dt
// limit), whose init the shared MOP state tracks.
template <ckernel::MathFidelity fidelity>
ALWI void matmul_step(uint32_t in0, uint32_t in1, uint32_t in0_tile, uint32_t in1_tile, uint32_t columns, uint32_t kb) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, in0_tile, in1_tile, columns, 1, kb)));
#ifdef ARCH_BLACKHOLE
    const uint32_t throttle = *ckernel::throttle_ptr % 2;
    if (ckernel::throttled_mop_status != throttle) {
        if (throttle) {
            MATH((llk_math_matmul_init<fidelity, MM_THROTTLE_MAX>(in0, in1, 0, columns, 1)));
        } else {
            MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(in0, in1, 0, columns, 1)));
        }
        ckernel::throttled_mop_status = throttle;
    }
    if (throttle) {
        MATH((llk_math_matmul<fidelity, MM_THROTTLE_MAX>(0, columns, 1)));
    } else {
        MATH((llk_math_matmul<fidelity, MM_THROTTLE>(0, columns, 1)));
    }
#else
    MATH((llk_math_matmul<fidelity, MM_THROTTLE>(0, columns, 1)));
#endif
}

// The output gate g = x @ W of `rows` tile rows and `columns` tile columns into cb_out (FP32, pushed whole): K arrives
// in blocks of `kb` tiles, [rows x kb] of x in cb_in0 and [kb x columns] of W in cb_in1, and each block's products
// accumulate in the FP32 DST, then onto the partial sums in cb_out in the packer (L1 accumulation).
template <ckernel::MathFidelity fidelity>
ALWI void matmul(
    uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out, uint32_t rows, uint32_t columns, uint32_t kb, uint32_t blocks) {
    reconfig_data_format<SrcOrder::Reverse>(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, columns, 1, kb)));
    MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(cb_in0, cb_in1, 0, columns, 1)));
#ifdef ARCH_BLACKHOLE
    MATH((ckernel::throttled_mop_status = 0));
#endif
    cb_reserve_back(cb_out, rows * columns);
    for (uint32_t b = 0; b < blocks; ++b) {
        cb_wait_front(cb_in0, rows * kb);
        cb_wait_front(cb_in1, kb * columns);
        for (uint32_t m = 0; m < rows; ++m) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < kb; ++k) {
                matmul_step<fidelity>(cb_in0, cb_in1, m * kb + k, k * columns, columns, kb);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < columns; ++j) {
                pack_tile<true>(j, cb_out, m * columns + j);
            }
            tile_regs_release();
        }
        if (b == 0) {
            pack_reconfig_l1_acc(1);
        }
        cb_pop_front(cb_in0, rows * kb);
        cb_pop_front(cb_in1, kb * columns);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_out, rows * columns);
}

}  // namespace kda_gate
