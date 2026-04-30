// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-stage logit: clamp -> intermediate CB -> 1/(1-x) divide -> log.

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/rsub.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"

namespace {
template <compute_kernel_lib::Dst Slot>
struct ClampOp : compute_kernel_lib::UnaryOp<ClampOp<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;
    uint32_t min_val;
    uint32_t max_val;
    ALWI static void init() { ckernel::clamp_tile_init(); }
    ALWI void call(uint32_t dst) const { ckernel::clamp_tile(dst, min_val, max_val); }
};

template <compute_kernel_lib::Dst Slot>
struct RsubOp : compute_kernel_lib::UnaryOp<RsubOp<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;
    uint32_t scalar;
    ALWI static void init() { ckernel::rsub_tile_init(); }
    ALWI void call(uint32_t dst) const { ckernel::rsub_tile(dst, scalar); }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::DivBinary;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;
    using compute_kernel_lib::Log;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Stage 1: optional clamp -> cb_tmp0 (single-tile pipeline).
#ifdef CLAMP
        eltwise_pipeline<cb_tmp0>(
            1, eltwise_chain(CopyTile<cb_input>{}, ClampOp<Dst::D0>{{}, packed_scalar1, packed_scalar2}));
#else
        eltwise_pipeline<cb_tmp0>(1, eltwise_chain(CopyTile<cb_input>{}));
#endif

        // Stage 2: D0 = rsub(x, 1.0) = 1-x; D1 = x; D0 = div(D1, D0) = x/(1-x); log(D0).
        eltwise_pipeline<cb_output>(
            1,
            eltwise_chain(
                CopyTile<cb_tmp0, Dst::D0>{},
                CopyTile<cb_tmp0, Dst::D1>{},
                RsubOp<Dst::D0>{{}, 0x3F800000u},  // 1.0f
                DivBinary<Dst::D1, Dst::D0, Dst::D0>{},
                Log<>{}));
    }
}
