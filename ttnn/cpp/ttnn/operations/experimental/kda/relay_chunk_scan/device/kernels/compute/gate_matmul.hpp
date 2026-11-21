// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace kda_gate {

// One K step of a 1 x `columns` subblock at `fidelity`, which may differ from the kernel's MATH_FIDELITY: the relay
// runs at the kernel's, the output gate at gate_fidelity. Throttled like matmul_block (the Blackhole di/dt
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

// The output gate of `ways` heads (1, or a pair over half of K): g = x @ W of `rows` tile rows and each head's
// `columns` tile columns, this core's head's into cb_own and a pair's other head's into cb_pair (FP32; cb_pair pushed
// for the pair's core, cb_own pushed alone, else left reserved for add_pair). K arrives in blocks of `kb` tiles,
// [rows x kb] of x in cb_in0 and [kb x ways columns] of W in cb_in1 (a pair's first head's columns, then its
// second's), and each block's products accumulate in the FP32 DST (one head's columns at a time: DST holds 4 FP32
// tiles), then onto the partial sums in the packer (L1 accumulation).
template <ckernel::MathFidelity fidelity>
ALWI void matmul(
    uint32_t cb_in0,
    uint32_t cb_in1,
    uint32_t cb_own,
    uint32_t cb_pair,
    bool own_first,
    uint32_t ways,
    uint32_t rows,
    uint32_t columns,
    uint32_t kb,
    uint32_t blocks) {
    reconfig_data_format<SrcOrder::Reverse>(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_own);
    UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, columns, 1, kb)));
    MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(cb_in0, cb_in1, 0, columns, 1)));
#ifdef ARCH_BLACKHOLE
    MATH((ckernel::throttled_mop_status = 0));
#endif
    cb_reserve_back(cb_own, rows * columns);
    if (ways == 2) {
        cb_reserve_back(cb_pair, rows * columns);
    }
    for (uint32_t b = 0; b < blocks; ++b) {
        cb_wait_front(cb_in0, rows * kb);
        cb_wait_front(cb_in1, kb * ways * columns);
        for (uint32_t m = 0; m < rows; ++m) {
            for (uint32_t head = 0; head < ways; ++head) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kb; ++k) {
                    matmul_step<fidelity>(cb_in0, cb_in1, m * kb + k, (ways * k + head) * columns, columns, kb);
                }
                tile_regs_commit();
                tile_regs_wait();
                const uint32_t cb_out = (head == 0) == own_first ? cb_own : cb_pair;
                for (uint32_t j = 0; j < columns; ++j) {
                    pack_tile<true>(j, cb_out, m * columns + j);
                }
                tile_regs_release();
            }
        }
        if (b == 0) {
            pack_reconfig_l1_acc(1);
        }
        cb_pop_front(cb_in0, rows * kb);
        cb_pop_front(cb_in1, kb * ways * columns);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(ways == 2 ? cb_pair : cb_own, rows * columns);
}

// g of this core's head: the pair's core's partial sums (cb_partial, FP32, unpacked to DST exactly) onto this core's
// in cb_own (L1 accumulation in FP32), which is then pushed.
ALWI void add_pair(uint32_t cb_partial, uint32_t cb_own, uint32_t tiles) {
    cb_wait_front(cb_partial, tiles);
    reconfig_data_format_srca(cb_partial);
    pack_reconfig_data_format(cb_own);
    copy_init(cb_partial);
    pack_reconfig_l1_acc(1);
    for (uint32_t i = 0; i < tiles; ++i) {
        tile_regs_acquire();
        copy_tile(cb_partial, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_own, i);
        tile_regs_release();
    }
    pack_reconfig_l1_acc(0);
    cb_pop_front(cb_partial, tiles);
    cb_push_back(cb_own, tiles);
}

}  // namespace kda_gate
