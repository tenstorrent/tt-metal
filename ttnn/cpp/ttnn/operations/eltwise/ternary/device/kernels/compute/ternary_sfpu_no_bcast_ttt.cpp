// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

// Kernel-local CRTP struct that defers the actual ternary op to the
// program-factory-provided TERNARY_SFPU_OP_INIT / TERNARY_SFPU_OP_FUNC macros.
// Helper does not need to know which specific ternary primitive is dispatched;
// the chain combinator handles CB lifecycle + DEST acquire/commit/wait/pack/release.
namespace {
struct TernaryMacroOp : compute_kernel_lib::TernaryOp<
                            TernaryMacroOp,
                            compute_kernel_lib::Dst::D0,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::Dst::D2,
                            compute_kernel_lib::Dst::D0> {
    // Most ternary ops (where, lerp, addcmul, addcdiv) program SFPU LUT via init.
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx);
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    static_assert(num_tiles_per_cycle == 1, "TTT path runs one tile per chain invocation");

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(cb_pre_in1, cb_out);
    ckernel::init_sfpu(cb_pre_in1, cb_out);

    compute_kernel_lib::eltwise_pipeline<cb_out>(
        num_tiles,
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::CopyTile<cb_pre_in1, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::CopyTile<cb_pre_in2, compute_kernel_lib::Dst::D1>{},
            compute_kernel_lib::CopyTile<cb_pre_in3, compute_kernel_lib::Dst::D2>{},
            TernaryMacroOp{}));
}
