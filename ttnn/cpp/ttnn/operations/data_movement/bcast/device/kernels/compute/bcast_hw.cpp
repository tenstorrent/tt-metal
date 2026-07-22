// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_lhs, cb_rhs, cb_out);

#ifdef BCAST_SCALAR
    constexpr auto rhs_lifecycle = ckl::InputLifecycle::HeldStream;
#else
    constexpr auto rhs_lifecycle = ckl::InputLifecycle::Streaming;
#endif

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(B * Ht * Wt),
        ckl::BinaryFpu<
            ckl::input(cb_lhs, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
            ckl::input(cb_rhs, rhs_lifecycle, ckl::DataFormatReconfig::Disabled),
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM>{},
        ckl::PackTile<ckl::output(cb_out, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}
