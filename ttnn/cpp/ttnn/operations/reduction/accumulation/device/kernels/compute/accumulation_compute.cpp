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

#if defined(COMPENSATED_SUM) && !defined(ARCH_QUASAR) && defined(TRISC_MATH)
namespace ckernel::sfpu {

// One SFPU pass for the Kahan step on four Dest tiles: y = in - c, t = acc + y, c = (t - acc) - y.
// Each op is a genuinely rounded fp32 op, which is what the algorithm depends on: (t - acc) - y must
// not be simplified to zero.
// The new c is set to 0 when t - acc is not finite. Once the total reaches +-inf, or t - acc
// overflows while t is still finite, (t - acc) - y is +-inf or NaN, and y = in - c would turn every
// later element into NaN (#58986). With c = 0 the next step is the plain fp32 add, so the total stays
// +-inf or continues as the uncompensated sum. While t - acc is finite the step is unchanged, so a
// scan whose compensation stays finite is bit-identical to the unguarded one.
// The test is on t - acc rather than on t as suggested in #58986: t stays finite in the second case
// above (test_cumsum_fp32_non_finite[finite_with_overflowing_c]). t - acc is not finite exactly when
// c would not be, and it is ready one op before c, so the test does not wait on the last subtraction.
// The biased exponent of an fp32 value is at most 255, so ">= 255" selects inf and NaN exactly like
// "== 255" does; the >= form lets the compare set the lane flags itself, one SFPU instruction fewer.
template <uint32_t DST_IN, uint32_t DST_ACC, uint32_t DST_COMP, uint32_t DST_T, int ITERATIONS = 8>
inline void calculate_kahan_step() {
    constexpr uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[DST_IN * dst_tile_size_sfpi];
        sfpi::vFloat acc = sfpi::dst_reg[DST_ACC * dst_tile_size_sfpi];
        sfpi::vFloat c = sfpi::dst_reg[DST_COMP * dst_tile_size_sfpi];
        sfpi::vFloat y = in - c;
        sfpi::vFloat t = acc + y;
        sfpi::vFloat diff = t - acc;
        sfpi::vFloat c_new = diff - y;
        v_if(sfpi::exexp(diff, sfpi::ExponentMode::Biased) >= 255) { c_new = 0.0f; }
        v_endif;
        sfpi::dst_reg[DST_COMP * dst_tile_size_sfpi] = c_new;
        sfpi::dst_reg[DST_T * dst_tile_size_sfpi] = t;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

// The four tiles are addressed from Dest tile 0 by the template arguments, so the call starts at 0.
template <uint32_t DST_IN, uint32_t DST_ACC, uint32_t DST_COMP, uint32_t DST_T>
inline void kahan_step_tile() {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_kahan_step,
        (DST_IN, DST_ACC, DST_COMP, DST_T, 8 /* ITERATIONS */),
        0,
        VectorMode::RC);
}
#endif

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

#ifdef COMPENSATED_SUM
            // in, acc and comp are all Float32 here (the factory sets COMPENSATED_SUM only then): one unpack
            // configuration serves the three copies.
            reconfig_data_format(dfb::in, dfb::in);
            copy_init(dfb::in);
            copy_tile(dfb::in, 0, DST_IN);
            copy_tile(dfb::acc, 0, DST_ACC);
            copy_tile(dfb::comp, 0, DST_COMP);
#else
            reconfig_data_format(dfb::in, dfb::in);
            copy_init(dfb::in);
            copy_tile(dfb::in, 0, DST_IN);

            reconfig_data_format(dfb::acc, dfb::acc);
            copy_init(dfb::acc);
            copy_tile(dfb::acc, 0, DST_ACC);
#endif
#ifdef COMPENSATED_SUM
            // Compensated (Kahan) summation on the running total. The plain path computes
            // acc = acc + in and throws away the rounding error of every add, so over a long scan
            // the error grows as ~T^1.5 and there is no length at which it stops (#55542). This
            // keeps the discarded low-order part in COMP and feeds it back on the next add, which
            // pins the error to O(1) ULP of the result independent of scan length.
#ifndef ARCH_QUASAR
            // One SFPU pass, guarded against a non-finite compensation (calculate_kahan_step above).
            MATH((kahan_step_tile<DST_IN, DST_ACC, DST_COMP, DST_T>()));
#else
            // Quasar keeps the four binary SFPU ops without the guard: a non-finite running total
            // still turns later elements into NaN there. No re-init between add and sub: the SFPU
            // binary init is op-specific only for DIV/POW/XLOGY, so BINARY_OP_INIT above covers both.
            sub_binary_tile(DST_IN, DST_COMP, DST_IN);    // y = in - c
            add_binary_tile(DST_ACC, DST_IN, DST_T);      // t = acc + y
            sub_binary_tile(DST_T, DST_ACC, DST_COMP);    // (t - acc)
            sub_binary_tile(DST_COMP, DST_IN, DST_COMP);  // c = (t - acc) - y
#endif
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
