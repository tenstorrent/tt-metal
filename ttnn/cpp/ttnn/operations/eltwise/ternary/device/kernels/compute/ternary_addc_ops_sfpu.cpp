// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
struct AddcMacroOp : compute_kernel_lib::TernaryOp<
                         AddcMacroOp,
                         compute_kernel_lib::Dst::D0,
                         compute_kernel_lib::Dst::D1,
                         compute_kernel_lib::Dst::D2,
                         compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;
    uint32_t scalar;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) const {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx, scalar);
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "addc_ops_sfpu path runs one tile per chain invocation");

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(cb_in0, cb_out);
    ckernel::init_sfpu(cb_in0, cb_out);

    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    eltwise_pipeline<cb_out>(
        num_tiles,
        eltwise_chain(
            CopyTile<cb_in0, Dst::D0>{},
            CopyTile<cb_in1, Dst::D1>{},
            CopyTile<cb_in2, Dst::D2>{},
            AddcMacroOp{{}, scalar_arg}));
}
