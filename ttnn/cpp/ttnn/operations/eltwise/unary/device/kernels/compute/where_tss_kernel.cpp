// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // SFPU_OP_CHAIN_0 — where_tile / where_tile_init etc.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

// U5: collapse #ifdef-gated chain branches into a single chain definition where
// one of two OptionalChainElement pairs is live. The fold's `<false, Inner>`
// specialisation inherits Inner's tag and emits no-ops.
#if defined(INP_INT32) || defined(INP_UINT32)
constexpr bool USE_INT_FILL = true;
#else
constexpr bool USE_INT_FILL = false;
#endif

namespace {

// Wraps the host-defined SFPU_OP_CHAIN_0 (where(D0, D1, D2) → D0).
struct WhereSfpu : compute_kernel_lib::DestOnlyTag {
    static ALWI void init() {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const { SFPU_OP_CHAIN_0; }
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

    // U5: single chain — exactly one of the two OptionalChainElement pairs is live.
    // The `<false, Inner>` specialisation is tag-only no-op (variadic ctor swallows
    // the args); dead branches cost zero at runtime. After this rewrite where_tss
    // is single-stage and uses U4's deduced wrapper.
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::None>{},
        OptionalChainElement<USE_INT_FILL, FillInt<DataFormat::Int32, Dst::D1>>{packed_scalar1},
        OptionalChainElement<USE_INT_FILL, FillInt<DataFormat::Int32, Dst::D2>>{packed_scalar2},
        OptionalChainElement<!USE_INT_FILL, FillScalar<Dst::D1>>{*true_value},
        OptionalChainElement<!USE_INT_FILL, FillScalar<Dst::D2>>{*false_value},
        WhereSfpu{},
        PackTile<
            cb_output,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
