// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"  // FillScalar
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_extended.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"  // Ltz
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"     // Where, LgammaStirlingFloat, LgammaAdjusted
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"        // Sin
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rounding.hpp"    // Floor, Frac
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"        // Abs
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"        // Log
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr float M_PI = 3.14159265358979323846f;

    init_sfpu(cb_input, cb_output);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        // x -> D0 (owns the wait), x -> D1.
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::HeldStream, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        // D2 = 0.5 ; D1 = x - 0.5 ; D1 = (x-0.5 < 0)
        ckl::FillScalar<ckl::Dst::D2>{0.5f},
        ckl::SubBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::Ltz<ckl::Dst::D1>{},
        // D2 = 1.0 ; D2 = 1 - x
        ckl::FillScalar<ckl::Dst::D2>{1.0f},
        ckl::SubBinary<ckl::Dst::D2, ckl::Dst::D0, ckl::Dst::D2>{},
        // D1 = z = where(cond=D1, a=1-x, b=x)
        ckl::Where<DataFormat::Float32, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0, ckl::Dst::D1>{},
        // D1 = log z
        ckl::Log<ckl::Approx::Exact, ckl::Dst::D1>{},
        // D0 = stirling(x=D0, log_z=D1)
        ckl::LgammaStirlingFloat<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        // D2 = M_PI ; reload x -> D1 ; D1 = sin(frac(x) * M_PI)
        ckl::FillScalar<ckl::Dst::D2>{M_PI},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::Frac<ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::Sin<ckl::Dst::D1>{},
        // reload x -> D2, D3 ; D3 = floor(x) ; D2 = (x == floor(x))
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D3>{},
        ckl::Floor<ckl::Dst::D3>{},
        ckl::EqBinary<ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D2>{},
        // D3 = 0 ; D1 = where(cond=D2, a=0, b=sin) -> 0 at integers else sin
        ckl::FillScalar<ckl::Dst::D3>{0.0f},
        ckl::Where<DataFormat::Float32, ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D1>{},
        // D1 = log|D1|
        ckl::Abs<ckl::Dst::D1>{},
        ckl::Log<ckl::Approx::Exact, ckl::Dst::D1>{},
        // reload x -> D2 (owns the pop) ; D0 = adjusted(stirling=D0, logsin=D1, x=D2)
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::NoWaitPop, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::LgammaAdjusted<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_output, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}
