// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Wraps the host-defined TERNARY_SFPU_OP_FUNC macro (where/lerp/addcmul/addcdiv).
template <
    compute_kernel_lib::Dst In0 = compute_kernel_lib::Dst::D0,
    compute_kernel_lib::Dst In1 = compute_kernel_lib::Dst::D1,
    compute_kernel_lib::Dst In2 = compute_kernel_lib::Dst::D2,
    compute_kernel_lib::Dst Out = compute_kernel_lib::Dst::D0>
struct TernarySfpuOp
    : compute_kernel_lib::TernaryOp<TernarySfpuOp<In0, In1, In2, Out>, In0, In1, In2, Out> {
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    static ALWI void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t o) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, o);
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_pre_in1, cb_pre_in2, cb_out);

    eltwise_chain(
        num_tiles,
        CopyTile<cb_pre_in1, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_pre_in2, Dst::D1, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_pre_in3, Dst::D2, CopyTilePolicy::WaitAndPop>{},
        TernarySfpuOp<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
