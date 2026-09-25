// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Small element scenarios that do not justify separate kernel files:
//   0: ternary Where over three input CBs;
//   1: one ReLU pack plus one unmodified pack.
//   2: int32 CopyDest from D0 to D1.
//   3-5: reciprocal and the two rsqrt modes.
//   6: precise reciprocal overriding the kernel's approximation mode.
//
// CT args: [n, mode].

#include <cstdint>
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"

struct PreciseRecip : compute_kernel_lib::UnaryOp<PreciseRecip, compute_kernel_lib::Dst::D0> {
    static constexpr auto dest_acc =
        DST_ACCUM_MODE ? ckernel::ReciprocalDestAcc::FP32 : ckernel::ReciprocalDestAcc::BF16;
    static ALWI void init() { ckernel::recip_tile_init<dest_acc, ckernel::ReciprocalApproxMode::Precise>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        ckernel::recip_tile<dest_acc, ckernel::ReciprocalApproxMode::Precise>(slot_offset);
    }
};

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    static_assert(mode < 7);

    using namespace compute_kernel_lib;
    if constexpr (mode == 0) {
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_a)>{},
            CopyTile<input(cb_b), Dst::D1>{},
            CopyTile<input(cb_c), Dst::D2>{},
            Where<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
            PackTile<output(cb_out)>{});
    } else if constexpr (mode == 1) {
        constexpr uint32_t cb_linear = tt::CBIndex::c_17;
        compute_kernel_hw_startup(cb_a, cb_out);
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_a)>{},
            PackTile<output(
                cb_out,
                ReservePolicy::PerTile,
                PushPolicy::PerTile,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::Disabled,
                L1Accumulation::Disabled,
                PackRelu::Zero)>{},
            PackTile<output(cb_linear)>{});
    } else if constexpr (mode == 2) {
        compute_kernel_hw_startup(cb_a, cb_out);
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_a), Dst::D0>{},
            CopyDest<Dst::D0, Dst::D1, DataFormat::Int32>{},
            PackTile<output(cb_out), Dst::D1>{});
    } else {
        compute_kernel_hw_startup(cb_a, cb_out);
        using RootOp = std::conditional_t<
            mode == 6,
            PreciseRecip,
            std::conditional_t<mode == 3, Recip<>, Rsqrt<mode == 5 ? Approx::Fast : Approx::Exact>>>;
        eltwise_chain(IterationShape::tiles(n), CopyTile<input(cb_a)>{}, RootOp{}, PackTile<output(cb_out)>{});
    }
}
