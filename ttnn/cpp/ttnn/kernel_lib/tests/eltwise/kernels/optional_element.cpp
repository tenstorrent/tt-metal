// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// OptionalChainElement validation kernel.
//
// Tests OptionalChainElement<COND, Inner> for both COND=true (forwards to Inner)
// and COND=false (tag-only no-op). The test harness flips OPTIONAL_COND at
// compile time and supplies `Inner` via OPTIONAL_INNER.
//
// Reference:
//   - When OPTIONAL_COND=true the chain runs `CopyTile + Inner + Pack` (golden:
//     apply Inner to input).
//   - When OPTIONAL_COND=false the chain runs `CopyTile + (no-op) + Pack`
//     (golden: identity copy).

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"  // pulls SFPU op-struct families

#ifndef OPTIONAL_COND
#define OPTIONAL_COND 1
#endif

// OPTIONAL_INNER_TYPE is host-injected — defaults to Eqz<Dst::D0> so the test
// kernel compiles standalone.
#ifndef OPTIONAL_INNER_TYPE
#define OPTIONAL_INNER_TYPE compute_kernel_lib::Eqz<compute_kernel_lib::Dst::D0>
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    constexpr bool kCond = (OPTIONAL_COND != 0);
    using InnerT = OPTIONAL_INNER_TYPE;

    // U5 / Q5 — exercise both <true, Inner> and <false, Inner> paths over each
    // tag the conditional ladder dispatches to.
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::None>{},
        OptionalChainElement<kCond, InnerT>{},
        PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::None>{});
}
