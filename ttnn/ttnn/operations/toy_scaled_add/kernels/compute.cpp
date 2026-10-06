// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// toy_scaled_add compute: out = a + alpha * (b * gamma), gamma broadcast down the rows.
// Without gamma: out = a + alpha * b.
//
// One element-wise chain walks this core's (num_rows x Wt) tile grid, per tile:
//   b * gamma[c] -> DEST   (FPU multiply, gamma's row 0 broadcast over the tile; or a plain copy of b)
//   DEST * alpha -> DEST   (SFPU, alpha from a common runtime arg)
//   a + DEST     -> DEST   (FPU add with DEST as srcA)
//   pack         -> out
// gamma is waited for once (Wt tiles) and indexed by column, so each gamma tile serves every row.
//
// The same kernel serves both programs: on the sharded one a / b / out are CBs over the shards, and
// the chain's per-tile waits and pops walk them in place.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"

namespace ckl = compute_kernel_lib;
using namespace toy_scaled_add;

void kernel_main() {
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    const uint32_t num_rows = get_arg_val<uint32_t>(core_arg::NUM_ROWS);
    const uint32_t alpha_bits = get_common_arg_val<uint32_t>(compute_arg::ALPHA_BITS);

    compute_kernel_hw_startup(cb::A, cb::B, cb::OUT);

    const auto shape = ckl::IterationShape::grid(num_rows, Wt);
#ifdef TOY_SCALED_ADD_HAS_GAMMA
    ckl::eltwise_chain(
        shape,
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(cb::B),
            ckl::input(
                cb::GAMMA,
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Row)>{},
        ckl::MulUnary<>{alpha_bits},
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, ckl::input(cb::A), ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<ckl::output(cb::OUT)>{});
#else
    ckl::eltwise_chain(
        shape,
        ckl::CopyTile<ckl::input(cb::B)>{},
        ckl::MulUnary<>{alpha_bits},
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, ckl::input(cb::A), ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<ckl::output(cb::OUT)>{});
#endif
}
