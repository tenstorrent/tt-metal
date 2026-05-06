// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/add_int_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"

namespace {

// DEST-DEST integer multiply where in0/in1/out may alias.
// Inherits DestOnlyTag directly (BinaryOp's static_assert forbids slot aliasing).
template <DataFormat DF, compute_kernel_lib::Dst In0, compute_kernel_lib::Dst In1, compute_kernel_lib::Dst Out>
struct MulIntInPlace : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() { mul_int_tile_init<DF>(); }
    static ALWI void exec() {
        mul_int_tile<DF>(compute_kernel_lib::to_u32(In0), compute_kernel_lib::to_u32(In1), compute_kernel_lib::to_u32(Out));
    }
};

template <DataFormat DF, compute_kernel_lib::Dst In0, compute_kernel_lib::Dst In1, compute_kernel_lib::Dst Out>
struct AddIntInPlace : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() { add_int_tile_init(); }
    static ALWI void exec() {
        add_int_tile<DF>(compute_kernel_lib::to_u32(In0), compute_kernel_lib::to_u32(In1), compute_kernel_lib::to_u32(Out));
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

    // out = input_a + scalar * input_b * input_c, integer dtype.
    // Mirrors the original 4-DST-slot layout (D3 reused as scratch).
    using Chain = EltwiseChain<
        CopyTile<cb_in0, Dst::D0, CopyTilePolicy::WaitAndPop>,
        CopyTile<cb_in1, Dst::D1, CopyTilePolicy::WaitAndPop>,
        CopyTile<cb_in2, Dst::D2, CopyTilePolicy::WaitAndPop>,
        FillInt<ADDCMUL_DATA_FORMAT, Dst::D3>,
        MulIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D1, Dst::D3>,
        MulIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D2, Dst::D2>,
        AddIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D0, Dst::D2, Dst::D0>,
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_in0, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_in1, Dst::D1, CopyTilePolicy::WaitAndPop>{},
        CopyTile<cb_in2, Dst::D2, CopyTilePolicy::WaitAndPop>{},
        FillInt<ADDCMUL_DATA_FORMAT, Dst::D3>{scalar_arg},
        MulIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D1, Dst::D3>{},
        MulIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D2, Dst::D2>{},
        AddIntInPlace<ADDCMUL_DATA_FORMAT, Dst::D0, Dst::D2, Dst::D0>{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
