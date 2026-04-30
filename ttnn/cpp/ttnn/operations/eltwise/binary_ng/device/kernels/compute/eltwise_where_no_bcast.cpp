// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
template <compute_kernel_lib::Dst Slot>
struct FillMacroOp : compute_kernel_lib::UnaryOp<FillMacroOp<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;
    uint32_t value;

    ALWI static void init() { ckernel::fill_tile_init(); }
    ALWI void call(uint32_t dst) const {
#ifdef FILL_WITH_VALUE_FLOAT
        const auto fval = reinterpret_cast<const float*>(&value);
        FILL_LLK(dst, *fval);
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(dst, value);
#endif
    }
};

struct WhereMacroOp : compute_kernel_lib::TernaryOp<
                          WhereMacroOp,
                          compute_kernel_lib::Dst::D0,
                          compute_kernel_lib::Dst::D1,
                          compute_kernel_lib::Dst::D2,
                          compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() {}
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        BINARY_SFPU_OP(i0, i1, i2, out_idx);
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "where_no_bcast path runs one tile per chain invocation");

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_in0, cb_out);
    ckernel::init_sfpu(cb_in0, cb_out);
    BINARY_SFPU_INIT

    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if WHERE_TTS
        eltwise_pipeline<cb_out>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<cb_in0, Dst::D0>{},
                CopyTile<cb_in1, Dst::D1>{},
                FillMacroOp<Dst::D2>{{}, scalar_value},
                WhereMacroOp{}));
#endif
#if WHERE_TST
        eltwise_pipeline<cb_out>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<cb_in0, Dst::D0>{},
                FillMacroOp<Dst::D1>{{}, scalar_value},
                CopyTile<cb_in1, Dst::D2>{},
                WhereMacroOp{}));
#endif
    }
}
