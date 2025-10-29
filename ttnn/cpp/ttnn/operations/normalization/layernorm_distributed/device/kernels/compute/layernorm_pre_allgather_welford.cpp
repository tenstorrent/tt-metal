// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes larnorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/welford.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "debug/dprint_pages.h"
#include "dprint_tensix.h"
// read dest reg
#include "debug/dprint.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#include "compute_kernel_api/transpose_wh_dest.h"

namespace NAMESPACE {
template <typename To, typename From>
inline To _bit_cast_(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Types must have same size");
    static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");

    union {
        From f;
        To t;
    } u;

    u.f = from;
    return u.t;
}
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_x2);

    DPRINT << "WELFORDS!" << ENDL();
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        transpose_wh_init_short(cb_inp);
        welford_init();

        tile_regs_acquire();
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(cb_inp, 1);  // cumulative wait
            transpose_wh_tile(cb_inp, 0, dst0);
            welford_tile<dst0, dst1, dst2, true, 0>((wt) * 32, W, 0, {});
            cb_pop_front(cb_inp, 1);
        }
        // tt-llk/issues/549
        // BUG: using transpose_dest here causes a bug. where the kernel hangs
        //  transpose_wh_dest_init_short();
        //  transpose_wh_dest(dst1);
        //  transpose_wh_dest(dst2);
        cb_reserve_back(cb_x2, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, cb_x2);
        pack_tile(dst2, cb_x2);
        cb_push_back(cb_x2, 2);
        tile_regs_release();
        reconfig_data_format(cb_x2, cb_x2);
        pack_reconfig_data_format(cb_out);
        transpose_wh_init_short(cb_x2);
        tile_regs_acquire();
        cb_wait_front(cb_x2, 2);  // cumulative wait
        transpose_wh_tile(cb_x2, 0, dst0);
        transpose_wh_tile(cb_x2, 1, dst1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        pack_tile(dst1, cb_out);
        cb_push_back(cb_out, 2);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
