// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"

namespace {

// Wraps the host-defined SFPU_OP_CHAIN_0 (where(D0, D1, D2) → D0).
struct WhereSfpu : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() {}
    ALWI void exec(uint32_t /*i*/) const { SFPU_OP_CHAIN_0; }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);
    const auto true_value = reinterpret_cast<const float*>(&packed_scalar1);
    const auto false_value = reinterpret_cast<const float*>(&packed_scalar2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_input, cb_input, cb_output);

#if defined(INP_INT32) || defined(INP_UINT32)
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        FillInt<DataFormat::Int32, Dst::D1>{packed_scalar1},
        FillInt<DataFormat::Int32, Dst::D2>{packed_scalar2},
        WhereSfpu{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
#if defined(INP_FLOAT) || defined(INP_FLOAT32)
    eltwise_chain(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        FillScalar<Dst::D1>{*true_value},
        FillScalar<Dst::D2>{*false_value},
        WhereSfpu{},
        PackTile<cb_output, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
#endif
}
