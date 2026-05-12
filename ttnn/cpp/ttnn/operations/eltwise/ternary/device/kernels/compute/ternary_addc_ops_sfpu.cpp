// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Wraps the host-defined TERNARY_SFPU_OP_FUNC(in0, in1, in2, out, scalar) — addcmul/addcdiv.
// Carries the scalar arg as a runtime instance member. Inherits FillTileTag so the chain
// dispatches the instance-based `exec(i)` path (allowing the runtime scalar to be captured).
template <
    compute_kernel_lib::Dst In0 = compute_kernel_lib::Dst::D0,
    compute_kernel_lib::Dst In1 = compute_kernel_lib::Dst::D1,
    compute_kernel_lib::Dst In2 = compute_kernel_lib::Dst::D2,
    compute_kernel_lib::Dst Out = compute_kernel_lib::Dst::D0>
struct TernarySfpuOpScalar : compute_kernel_lib::FillTileTag {
    uint32_t scalar;
    constexpr explicit TernarySfpuOpScalar(uint32_t s) noexcept : scalar(s) {}
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI void exec(uint32_t /*i*/) const {
        TERNARY_SFPU_OP_FUNC(
            compute_kernel_lib::to_u32(In0),
            compute_kernel_lib::to_u32(In1),
            compute_kernel_lib::to_u32(In2),
            compute_kernel_lib::to_u32(Out),
            scalar);
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    (void)num_tiles_per_cycle;

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    // D5/D8: caller-side BIG init at the top of MAIN(). The chain reads from cb_in0
    // first; SFPU ternary op runs in DEST. Boot for the first reader's CB.
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_in0, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_in1, Dst::D1, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_in2, Dst::D2, CopyTilePolicy::WaitAndPop>{},
        TernarySfpuOpScalar<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{scalar_arg},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
