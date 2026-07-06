// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
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

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(B * Ht, Wt),
        ckl::BinaryFpu<
            cb_lhs,
            cb_rhs,
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM,
            ckl::InputLifecycle::Streaming,    // cb_lhs: one tile per (row,col)
            ckl::InputLifecycle::OuterStream,  // cb_rhs: streamed broadcast, one per row
            ckl::BinaryDataFormatReconfig::None,
            ckl::Dst::D0,
            ckl::OperandKind::Scalar,     // cb_lhs reads the front
            ckl::OperandKind::Scalar>{},  // cb_rhs reads the front (advances per row)
        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
