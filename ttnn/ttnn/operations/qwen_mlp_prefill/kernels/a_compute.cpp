// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"

#include "swiglu_sfpu.hpp"

void kernel_main() {
    constexpr uint32_t k_tiles = CT_ARG("h_tiles");
    constexpr uint32_t columns = CT_ARG("a_columns_per_core");
    constexpr uint32_t cb_n = CT_ARG("cb_n");
    constexpr uint32_t cb_wg = CT_ARG("cb_wg");
    constexpr uint32_t cb_a_out = CT_ARG("cb_a_out");
    CircularBuffer n(cb_n);
    CircularBuffer weights(cb_wg);
    CircularBuffer a_out(cb_a_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_n, cb_wg, cb_a_out);
    PACK((swiglu_tile_init()));
    matmul_block_init(cb_n, cb_wg, false, columns, 1, k_tiles);
    weights.wait_front(k_tiles * columns);

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        n.wait_front(k_tiles);
        a_out.reserve_back(columns / 2);
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_tiles; k++) {
            matmul_block(cb_n, cb_wg, k, k * columns, 0, false, columns, 1, k_tiles);
        }
        tile_regs_commit();
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t pair = 0; pair < columns; pair += 2) {
            PACK((swiglu_tile(pair, pair + 1, pair)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t pair = 0; pair < columns; pair += 2) {
            pack_tile<true>(pair, cb_a_out, pair / 2);
        }
        tile_regs_release();
        a_out.push_back(columns / 2);
        n.pop_front(k_tiles);
    }
}
