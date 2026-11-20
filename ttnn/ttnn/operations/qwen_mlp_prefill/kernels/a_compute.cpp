// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"

#include "swiglu_sfpu.hpp"

void kernel_main() {
    constexpr uint32_t k_tiles = CT_ARG("h_tiles");
    constexpr uint32_t columns = CT_ARG("a_columns_per_core");
    constexpr uint32_t pair = 2;
    constexpr uint32_t scale_tile = 2;
    constexpr uint32_t cb_n = CT_ARG("cb_n");
    constexpr uint32_t cb_wg = CT_ARG("cb_wg");
    constexpr uint32_t cb_a_out = CT_ARG("cb_a_out");
    CircularBuffer n(cb_n);
    CircularBuffer weights(cb_wg);
    CircularBuffer a_out(cb_a_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_n, cb_wg, cb_a_out);
    PACK((swiglu_tile_init()));
    matmul_block_init(cb_n, cb_wg, false, pair, 1, k_tiles);
    weights.wait_front(k_tiles * columns);

    for (uint32_t m = 0; m < CT_ARG("rows"); m++) {
        n.wait_front(CT_ARG("n_row_tiles"));

        a_out.reserve_back(columns / pair);
        for (uint32_t first = 0; first < columns; first += pair) {
            tile_regs_acquire();

            for (uint32_t k = 0; k < k_tiles; k++) {
                matmul_block(cb_n, cb_wg, k, k * columns + first, 0, false, pair, 1, k_tiles);
            }

            reconfig_data_format_srca(cb_n);
            copy_init(cb_n);
            copy_tile(cb_n, k_tiles, scale_tile);
            tile_regs_commit();

            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            PACK((swiglu_tile(0, 1, 0, scale_tile)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile<true>(0, cb_a_out, first / pair);
            tile_regs_release();

            reconfig_data_format(cb_wg, cb_n);
            matmul_block_init(cb_n, cb_wg, false, pair, 1, k_tiles);
        }
        a_out.push_back(columns / pair);

        n.pop_front(CT_ARG("n_row_tiles"));
    }
}
