// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as bcast_hw_metal2.cpp. Ops ported to Metal 2.0
// bind the fork; this file serves the consumers still on the legacy API. Until the last of them migrates
// and this file is retired, changes here likely belong in the fork too.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_lhs, cb_rhs, cb_out);

#ifdef BCAST_SCALAR
    constexpr auto rhs_pop = ckl::PopPolicy::None;
#else
    constexpr auto rhs_pop = ckl::PopPolicy::PerTile;
#endif

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(B * Ht * Wt),
        ckl::BinaryFpu<
            ckl::input(cb_lhs, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::input(cb_rhs, ckl::WaitPolicy::PerTile, rhs_pop, ckl::DataFormatReconfig::Disabled),
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM>{},
        ckl::PackTile<ckl::output(
            cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
