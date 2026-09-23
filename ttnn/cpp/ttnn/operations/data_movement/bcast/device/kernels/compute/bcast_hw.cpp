// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as bcast_hw_metal2.cpp. Ops ported to Metal 2.0
// bind the fork; this file serves the consumers still on the legacy API. Until the last of them migrates
// and this file is retired, changes here likely belong in the fork too.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto dfb_lhs_id = tt::CBIndex::c_0;
    constexpr auto dfb_rhs_id = tt::CBIndex::c_1;
    constexpr auto dfb_out_id = tt::CBIndex::c_16;

    compute_kernel_hw_startup(dfb_lhs_id, dfb_rhs_id, dfb_out_id);

#ifdef BCAST_SCALAR
    constexpr auto rhs_wait = ckl::WaitPolicy::Upfront;
    constexpr auto rhs_pop = ckl::PopPolicy::None;
#else
    constexpr auto rhs_wait = ckl::WaitPolicy::PerTile;
    constexpr auto rhs_pop = ckl::PopPolicy::PerTile;
#endif

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(B * Ht * Wt),
        ckl::BinaryFpu<
            CHAIN_BCAST_OP,
            ckl::input(
                dfb_lhs_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::input(dfb_rhs_id, CHAIN_BCAST_DIM, rhs_wait, rhs_pop, ckl::DataFormatReconfig::Disabled)>{},
        ckl::PackTile<ckl::output(
            dfb_out_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
