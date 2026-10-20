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

#if defined(COMPENSATED_SUM) && defined(TRISC_MATH)
namespace ckernel::sfpu {

// Single-pass Kahan summation step across the four Dest tiles:
//   y = in - c
//   t = acc + y
//   diff = t - acc
//   c_new = (t - acc) - y
// When the running total reaches +-inf (or diff = t - acc overflows to +-inf while t remains
// finite), (t - acc) - y evaluates to NaN or +-inf, which would poison every subsequent scan
// element via y = in - c (#58986). Zeroing c_new whenever diff is non-finite (biased exponent >= 255)
// keeps subsequent steps on standard IEEE/PyTorch fp32 addition while preserving bit-exact
// compensated summation whenever diff is finite.
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

            reconfig_data_format(dfb::in, dfb::in);
            copy_init(dfb::in);
            copy_tile(dfb::in, 0, DST_IN);

            reconfig_data_format(dfb::acc, dfb::acc);
            copy_init(dfb::acc);
            copy_tile(dfb::acc, 0, DST_ACC);

#ifdef COMPENSATED_SUM
            // dfb::comp shares DataFormat::Float32 and UnpackToDest with dfb::acc.
            copy_tile(dfb::comp, 0, DST_COMP);

            // Compensated (Kahan) summation on the running total in a single SFPU pass,
            // guarded against non-finite (t - acc) so +-inf or overflow does not poison
            // subsequent elements with NaN (#58986).
            MATH((kahan_step_tile<DST_IN, DST_ACC, DST_COMP, DST_T>()));
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
