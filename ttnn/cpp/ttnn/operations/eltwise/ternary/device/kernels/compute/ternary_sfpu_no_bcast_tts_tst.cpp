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
#include "api/compute/eltwise_unary/fill.h"

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

struct TernaryMacroOp : compute_kernel_lib::TernaryOp<
                            TernaryMacroOp,
                            compute_kernel_lib::Dst::D0,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::Dst::D2,
                            compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx);
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "TTS/TST path runs one tile per chain invocation");
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);  // 1=TST, 0=TTS

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(cb_pre_in1, cb_out);
    ckernel::init_sfpu(cb_pre_in1, cb_out);

    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    if constexpr (scalar_is_true) {
        // TST: predicate=D0, true=fill(scalar) at D1, false=tensor at D2
        eltwise_pipeline<cb_out>(
            num_tiles,
            eltwise_chain(
                CopyTile<cb_pre_in1, Dst::D0>{},
                CopyTile<cb_pre_in2, Dst::D2>{},
                FillMacroOp<Dst::D1>{{}, scalar_value},
                TernaryMacroOp{}));
    } else {
        // TTS: predicate=D0, true=tensor at D1, false=fill(scalar) at D2
        eltwise_pipeline<cb_out>(
            num_tiles,
            eltwise_chain(
                CopyTile<cb_pre_in1, Dst::D0>{},
                CopyTile<cb_pre_in2, Dst::D1>{},
                FillMacroOp<Dst::D2>{{}, scalar_value},
                TernaryMacroOp{}));
    }
}
