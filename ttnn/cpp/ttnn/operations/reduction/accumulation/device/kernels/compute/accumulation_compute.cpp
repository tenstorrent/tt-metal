// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#define APPROX false
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "../accumulation_common.hpp"

void kernel_main() {
    constexpr auto default_acc_value = get_arg(args::default_acc_value);

    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t tiles_per_row = get_arg(args::tiles_per_row);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_out_obj(dfb::out);
    DataflowBuffer dfb_acc_obj(dfb::acc);  // note: only used in compute kernel
#ifdef COMPENSATED_SUM
    // Kahan compensation term, kept in its own single-entry FIFO exactly like the accumulator:
    // packed out each step and read straight back at full 32-bit precision on the next.
    DataflowBuffer dfb_comp_obj(dfb::comp);
#endif

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);

    BINARY_OP_INIT();

    constexpr uint32_t DST_IN = 0;
    constexpr uint32_t DST_ACC = 1;
#ifdef COMPENSATED_SUM
    // Dest holds exactly four tiles in this kernel's mode (double-buffered, 32-bit): these are all
    // of them. DST_T receives the new total and is what gets packed; DST_ACC is the old total.
    constexpr uint32_t DST_COMP = 2;
    constexpr uint32_t DST_T = 3;
#endif

    dfb_acc_obj.reserve_back(ONE_TILE);
    dfb_acc_obj.push_back(ONE_TILE);
#ifdef COMPENSATED_SUM
    dfb_comp_obj.reserve_back(ONE_TILE);
    dfb_comp_obj.push_back(ONE_TILE);
#endif

    for (uint32_t i = 0; i < num_rows; i++) {
        // Synchronize unpacker-packer between iterations
        // This is necessary to avoid data-races on the accumulator buffer
        dfb_acc_obj.wait_front(ONE_TILE);
        dfb_acc_obj.pop_front(ONE_TILE);
#ifdef COMPENSATED_SUM
        dfb_comp_obj.wait_front(ONE_TILE);
        dfb_comp_obj.pop_front(ONE_TILE);
#endif

        tile_regs_acquire();
        reconfig_data_format(dfb::acc, dfb::acc);

        fill_tile_init();
        FILL_TILE(DST_ACC, default_acc_value);
#ifdef COMPENSATED_SUM
        FILL_TILE(DST_COMP, 0u);  // compensation starts at +0.0f for every row
#endif

        tile_regs_commit();

        tile_regs_wait();

        // out_of_order_output to keep packing to the accumulator buffer at the same location
        dfb_acc_obj.reserve_back(ONE_TILE);

        pack_reconfig_data_format(dfb::acc);
        pack_tile(DST_ACC, dfb::acc);
#ifdef COMPENSATED_SUM
        dfb_comp_obj.reserve_back(ONE_TILE);
        pack_reconfig_data_format(dfb::comp);
        pack_tile(DST_COMP, dfb::comp);
#endif
        tile_regs_release();

        dfb_acc_obj.push_back(ONE_TILE);
#ifdef COMPENSATED_SUM
        dfb_comp_obj.push_back(ONE_TILE);
#endif

        for (uint32_t j = 0; j < tiles_per_row; j++) {
            // Synchronize unpacker-packer between iterations
            dfb_acc_obj.wait_front(ONE_TILE);
#ifdef COMPENSATED_SUM
            dfb_comp_obj.wait_front(ONE_TILE);
#endif

            tile_regs_acquire();
            dfb_in_obj.wait_front(ONE_TILE);

            reconfig_data_format(dfb::in, dfb::in);
            copy_init(dfb::in);
            copy_tile(dfb::in, 0, DST_IN);

            reconfig_data_format(dfb::acc, dfb::acc);
            copy_init(dfb::acc);
            copy_tile(dfb::acc, 0, DST_ACC);

#ifdef COMPENSATED_SUM
            reconfig_data_format(dfb::comp, dfb::comp);
            copy_init(dfb::comp);
            copy_tile(dfb::comp, 0, DST_COMP);

            // Compensated (Kahan) summation on the running total. The plain path computes
            // acc = acc + in and throws away the rounding error of every add, so over a long scan
            // the error grows as ~T^1.5 and there is no length at which it stops (#55542). This
            // keeps the discarded low-order part in COMP and feeds it back on the next add, which
            // pins the error to O(1) ULP of the result independent of scan length. Four SFPU ops
            // per tile instead of one. Each op is a genuinely rounded fp32 op on the SFPU, which is
            // what the algorithm depends on: (t - acc) - y must NOT be simplified to zero.
            // Kahan step. No re-init between add and sub: the SFPU binary init is op-specific
            // only for DIV/POW/XLOGY (ckernel_sfpu_binary.h), so BINARY_OP_INIT above covers both.
            sub_binary_tile(DST_IN, DST_COMP, DST_IN);    // y = in - c
            add_binary_tile(DST_ACC, DST_IN, DST_T);      // t = acc + y
            sub_binary_tile(DST_T, DST_ACC, DST_COMP);    // (t - acc)
            sub_binary_tile(DST_COMP, DST_IN, DST_COMP);  // c = (t - acc) - y
            constexpr uint32_t DST_RESULT = DST_T;
            dfb_comp_obj.pop_front(ONE_TILE);
#else
            BINARY_OP(DST_IN, DST_ACC, DST_ACC);
            constexpr uint32_t DST_RESULT = DST_ACC;
#endif
            dfb_acc_obj.pop_front(ONE_TILE);

            dfb_in_obj.pop_front(ONE_TILE);

            tile_regs_commit();

            tile_regs_wait();

            dfb_out_obj.reserve_back(ONE_TILE);
            pack_reconfig_data_format(dfb::acc, dfb::out);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_RESULT, dfb::out);
            dfb_out_obj.push_back(ONE_TILE);

            dfb_acc_obj.reserve_back(ONE_TILE);

            pack_reconfig_data_format(dfb::out, dfb::acc);  // Needed for fp32_acc_to_dest=True
            pack_tile(DST_RESULT, dfb::acc);
#ifdef COMPENSATED_SUM
            dfb_comp_obj.reserve_back(ONE_TILE);
            pack_reconfig_data_format(dfb::acc, dfb::comp);
            pack_tile(DST_COMP, dfb::comp);
#endif

            tile_regs_release();

            dfb_acc_obj.push_back(ONE_TILE);
#ifdef COMPENSATED_SUM
            dfb_comp_obj.push_back(ONE_TILE);
#endif
        }
    }

    // Clean-up and empty the accumulator buffer
    dfb_acc_obj.wait_front(ONE_TILE);
    dfb_acc_obj.pop_front(ONE_TILE);
#ifdef COMPENSATED_SUM
    dfb_comp_obj.wait_front(ONE_TILE);
    dfb_comp_obj.pop_front(ONE_TILE);
#endif
}
