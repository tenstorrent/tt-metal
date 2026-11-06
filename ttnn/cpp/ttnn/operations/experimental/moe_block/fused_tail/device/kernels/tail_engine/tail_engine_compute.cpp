// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailEngine engine core compute: my own rows' shares tilized into their landing slots; per step my share's squares,
// all of my half's K in DEST (my finished columns kept, the partner's sent), the half's sum of squares, and the
// previous step's out = (mine + the partner's) x rsqrt(sum / Hr + eps) + se.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop.hpp"
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/tilize.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif

constexpr uint32_t KH = get_compile_time_arg_val(0), GX = get_compile_time_arg_val(1),
                   SHMAX = get_compile_time_arg_val(2);
constexpr uint32_t INV_HR = get_compile_time_arg_val(3), NORM_EPS = get_compile_time_arg_val(4),
                   XT = get_compile_time_arg_val(5);
constexpr uint32_t CB_W = 0, CB_A = 1, CB_PART = 2, CB_PMY = 3, CB_ONES = 4, CB_ACCM = 5, CB_XSEND = 6, CB_SQM = 7;
constexpr uint32_t CB_XRECV = 8, CB_SE = 9, CB_OUT = 10, CB_L0 = 11, CB_RM = 12;

// (the squares' row sums at HiFi4 whatever the kernel's fidelity: HiFi2 truncates them ~0.8% low)
ALWI void mm4_init(uint32_t in0, uint32_t in1) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, 1, 1, 1)));
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(in0, in1, 0, 1, 1)));
}
ALWI void mm4(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, 1, 1, 1)));
    MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(idst, 1, 1)));
}

MB_LOOP_BODY {
    // args: pn, my finished columns' offset in mine (fo) and count (fn), my share [sk0, sk1), steps, own steps, the W'
    // K range bounds 0, .., KH
    const uint32_t pn = get_arg_val<uint32_t>(0), fo = get_arg_val<uint32_t>(1), fn = get_arg_val<uint32_t>(2);
    const uint32_t sk0 = get_arg_val<uint32_t>(3), sk1 = get_arg_val<uint32_t>(4), nst = get_arg_val<uint32_t>(5);
    const uint32_t nown = get_arg_val<uint32_t>(6);
    uint32_t wr = 0, wnext = 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_A, CB_W, CB_ACCM);
    for (uint32_t j = 0; j < nown; ++j) {  // my shares of my own rows tilized into their landing slots (no separate
                                           // tilize: the combine's rows read once)
        DeviceZoneScopedN("TL-TILIZE");
        const uint32_t n = sk1 - sk0;
        cb_wait_front(CB_RM, n);
        cb_reserve_back(CB_L0, SHMAX);
        reconfig_data_format_srca(CB_RM);
        pack_reconfig_data_format(CB_L0);
        tilize_init(CB_RM, n, CB_L0);
        tilize_block(CB_RM, n, CB_L0);
        tilize_uninit(CB_RM, CB_L0);
        cb_push_back(CB_L0, SHMAX);
        cb_pop_front(CB_RM, n);
    }
    cb_wait_front(CB_ONES, 1);
    // out = (mine + the partner's) x s + se for the oldest step not yet out - run behind the next step's squares,
    // matmul and sums (the partner's half and its sum of squares land meanwhile: the exchange's round trip off the
    // step's critical path)
    auto out_step = [&]() {
        DeviceZoneScopedN("TL-OUT");
        cb_wait_front(CB_XRECV, XT);
        cb_wait_front(CB_SQM, 1);
        cb_wait_front(CB_ACCM, fn);
        cb_wait_front(CB_SE, fn);
        cb_reserve_back(CB_OUT, fn);
        tile_regs_acquire();
        // (DEST: s 0, mine 1 .., the partner's sum of squares 4, its partials 5 .., then se 4 ..)
        reconfig_data_format_srca(CB_SQM);
        copy_init(CB_SQM);
        copy_tile(CB_SQM, 0, 0);
        reconfig_data_format_srca(CB_XRECV);
        copy_init(CB_XRECV);
        copy_tile(CB_XRECV, XT - 1, 4);
        for (uint32_t j = 0; j < fn; ++j) {
            copy_tile(CB_XRECV, j, 5 + j);
        }
        add_binary_tile_init();
        add_binary_tile(0, 4, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, INV_HR);
        add_unary_tile(0, NORM_EPS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        reconfig_data_format_srca(CB_ACCM);
        copy_init(CB_ACCM);
        for (uint32_t j = 0; j < fn; ++j) {
            copy_tile(CB_ACCM, j, 1 + j);
        }
        add_binary_tile_init();
        for (uint32_t j = 0; j < fn; ++j) {
            add_binary_tile(1 + j, 5 + j, 1 + j);
        }
        mul_binary_tile_init();
        for (uint32_t j = 0; j < fn; ++j) {
            mul_binary_tile(1 + j, 0, 1 + j);
        }
        reconfig_data_format_srca(CB_SE);
        copy_init(CB_SE);
        for (uint32_t j = 0; j < fn; ++j) {
            copy_tile(CB_SE, j, 4 + j);
        }
        add_binary_tile_init();
        for (uint32_t j = 0; j < fn; ++j) {
            add_binary_tile(1 + j, 4 + j, 1 + j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        for (uint32_t j = 0; j < fn; ++j) {
            pack_tile(1 + j, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, fn);
        cb_pop_front(CB_XRECV, XT);
        cb_pop_front(CB_SQM, 1);
        cb_pop_front(CB_ACCM, fn);
        cb_pop_front(CB_SE, fn);
    };
    for (uint32_t s = 0; s < nst; ++s) {
        cb_wait_front(CB_A, KH);
        {  // my share's squares accumulated elementwise in DEST (fp32, SFPU): E = sum_k z_k * z_k
            DeviceZoneScopedN("TL-SQ");
            cb_reserve_back(CB_PMY, 1);
            reconfig_data_format_srca(CB_A);
            pack_reconfig_data_format(CB_PMY);
            tile_regs_acquire();
            for (uint32_t k0 = sk0; k0 < sk1; k0 += 7) {  // (DEST 1 .. 7 a batch; 0 the sum)
                const uint32_t n = sk1 - k0 < 7 ? sk1 - k0 : 7;
                copy_init(CB_A);
                for (uint32_t i = 0; i < n; ++i) {
                    copy_tile(CB_A, k0 + i, 1 + i);
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < n; ++i) {
                    mul_binary_tile(1 + i, 1 + i, 1 + i);
                }
                if (k0 == sk0) {
                    copy_dest_values<DataFormat::Float32>(1, 0);
                }
                add_binary_tile_init();
                for (uint32_t i = k0 == sk0 ? 1 : 0; i < n; ++i) {
                    add_binary_tile(0, 1 + i, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_PMY);
            tile_regs_release();
            cb_push_back(CB_PMY, 1);
        }
        {  // all of my half's K in DEST: my finished columns to ACCM, the partner's to XSEND (fp32)
            DeviceZoneScopedN("TL-MM");
            reconfig_data_format(CB_W, CB_A);
            matmul_block_init(CB_A, CB_W, 0, pn, 1, KH);
            cb_reserve_back(CB_ACCM, fn);
            cb_reserve_back(CB_XSEND, XT);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KH; ++k) {
                if (s == 0 && k == wnext) {  // (step 0: W' K range by range, as each lands)
                    wnext = get_arg_val<uint32_t>(8 + wr++);
                    cb_wait_front(CB_W, wnext * pn);
                }
                matmul_block(CB_A, CB_W, k, k * pn, 0, false, pn, 1, KH);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_ACCM);
            for (uint32_t j = 0; j < fn; ++j) {
                pack_tile<true>(fo + j, CB_ACCM, j);
            }
            for (uint32_t j = 0, i = 0; j < pn; ++j) {
                if (j < fo || j >= fo + fn) {
                    pack_tile<true>(j, CB_XSEND, i++);
                }
            }
            tile_regs_release();
            cb_push_back(CB_ACCM, fn);
            cb_pop_front(CB_A, KH);
        }
        {  // my half's sum of squares (the row's E tiles @ ones, HiFi4, accumulated in DEST): to the partner and mine
            DeviceZoneScopedN("TL-S");
            cb_wait_front(CB_PART, GX);
            cb_reserve_back(CB_SQM, 1);
            reconfig_data_format(CB_ONES, CB_PART);
            mm4_init(CB_PART, CB_ONES);
            tile_regs_acquire();
            for (uint32_t i = 0; i < GX; ++i) {
                mm4(CB_PART, CB_ONES, i, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_XSEND);
            pack_tile<true>(0, CB_XSEND, XT - 1);
            pack_tile(0, CB_SQM);
            tile_regs_release();
            cb_push_back(CB_XSEND, XT);
            cb_push_back(CB_SQM, 1);
            cb_pop_front(CB_PART, GX);
        }
        if (s > 0) {
            out_step();  // (step s - 1's)
        }
    }
    if (nst) {
        out_step();
    }
}

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop_main.hpp"
