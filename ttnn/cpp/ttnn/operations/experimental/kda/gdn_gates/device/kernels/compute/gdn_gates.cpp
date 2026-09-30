// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Reproduces, op for op, the binary_ng chain it replaces (bf16 DEST, fp32 accumulation off):
//   bs   = bf16(sigmoid(b))                              SFPU sigmoid, bf16 pack
//   beta = bf16(bs * scale)                              SFPU multiply (mul_binary_tile, RNE into the bf16 DEST),
//   packed to fp32 (exact) sp   = bf16(softplus(a + dt_bias))                   FPU add feeding SFPU softplus directly
//   (no bf16 rounding of
//                                                        the sum, as in the chain), bf16 pack
//   g    = bf16(a_neg * sp)                              SFPU multiply, packed to fp32 (exact)
// dt_bias and a_neg are row-broadcast once per core (dtf / anegf); the scalar scale tile is filled in DEST.
// left_faces_only: num_heads <= 16, so the valid columns are the left faces (0 and 2) of every tile; the SFPU
// sigmoid/softplus then run in VectorMode::C (half the work). The other faces keep their input padding.
template <uint32_t left_faces_only, uint32_t scale_bits>
TT_KERNEL void compute(uint32_t mt_count) {
    constexpr VectorMode vmode = left_faces_only ? VectorMode::C : VectorMode::RC;
    compute_kernel_hw_startup(dfb::a, dfb::dt, dfb::sp);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer dt(dfb::dt);
    DataflowBuffer aneg(dfb::aneg);
    DataflowBuffer dtf(dfb::dtf);
    DataflowBuffer anegf(dfb::anegf);
    DataflowBuffer bs(dfb::bs);
    DataflowBuffer sp(dfb::sp);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer g(dfb::g);
    // Row-broadcast dt_bias and a_neg (row 0 -> all rows) like binary_ng's unary_bcast<ROW> does.
    dt.wait_front(1);
    dtf.reserve_back(1);
    pack_reconfig_data_format(dfb::dtf);
    reconfig_data_format(dfb::dt, dfb::dt);
    unary_bcast_init<BroadcastType::ROW>(dfb::dt);
    tile_regs_acquire();
    unary_bcast<BroadcastType::ROW>(dfb::dt, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::dtf, 0);
    tile_regs_release();
    dtf.push_back(1);
    dt.pop_front(1);
    aneg.wait_front(1);
    anegf.reserve_back(1);
    pack_reconfig_data_format(dfb::anegf);
    reconfig_data_format(dfb::aneg, dfb::aneg);
    unary_bcast_init<BroadcastType::ROW>(dfb::aneg);
    tile_regs_acquire();
    unary_bcast<BroadcastType::ROW>(dfb::aneg, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::anegf, 0);
    tile_regs_release();
    anegf.push_back(1);
    aneg.pop_front(1);
    dtf.wait_front(1);
    anegf.wait_front(1);

    for (uint32_t i = 0; i < mt_count; i++) {
        b.wait_front(1);
        a.wait_front(1);

        // bs = sigmoid(b), rounded to bf16 by the bf16 pack.
        bs.reserve_back(1);
        pack_reconfig_data_format(dfb::bs);
        reconfig_data_format_srca(dfb::b);
        copy_init(dfb::b);
        tile_regs_acquire();
        copy_tile(dfb::b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile<vmode>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::bs, 0);
        tile_regs_release();
        bs.push_back(1);
        b.pop_front(1);

        // beta = bs * scale -> fp32.
        bs.wait_front(1);
        beta.reserve_back(1);
        pack_reconfig_data_format(dfb::beta);
        mul_binary_tile_init();
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::bs);
        copy_init(dfb::bs);
        copy_tile(dfb::bs, 0, 0);
        fill_tile_init();
        fill_tile_bitcast(1, scale_bits);  // the scalar operand tile (bf16-exact scale) built in DEST
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::beta, 0);
        tile_regs_release();
        beta.push_back(1);
        bs.pop_front(1);

        // sp = softplus(a + dt_bias): the FPU add feeds softplus directly, the result is rounded by the bf16 pack.
        sp.reserve_back(1);
        pack_reconfig_data_format(dfb::sp);
        reconfig_data_format(dfb::a, dfb::dtf);
        add_init(dfb::a, dfb::dtf, false);
        tile_regs_acquire();
        add_tiles(dfb::a, dfb::dtf, 0, 0, 0);
        softplus_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_softplus,
            (APPROX, DST_ACCUM_MODE),
            0,
            vmode,
            0x3f800000u,
            0x3f800000u,
            0x41a00000u));  // = softplus_tile(0, beta=1, 1/beta=1, threshold=20) with a vector mode
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::sp, 0);
        tile_regs_release();
        sp.push_back(1);
        a.pop_front(1);

        // g = a_neg * sp -> fp32 (a_neg is the left operand).
        sp.wait_front(1);
        g.reserve_back(1);
        pack_reconfig_data_format(dfb::g);
        mul_binary_tile_init();
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::anegf);
        copy_init(dfb::anegf);
        copy_tile(dfb::anegf, 0, 0);
        reconfig_data_format_srca(dfb::sp);
        copy_init(dfb::sp);
        copy_tile(dfb::sp, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::g, 0);
        tile_regs_release();
        g.push_back(1);
        sp.pop_front(1);
    }
}
