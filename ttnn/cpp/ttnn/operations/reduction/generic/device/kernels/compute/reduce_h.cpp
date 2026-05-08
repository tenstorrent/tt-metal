// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the multi-core H reduction primitive (no negation).
//
// Why this file is distinct from reduce.cpp:
//   - The W factory varies Ht per core group (work splits along H rows) and binds
//     Ht as a per-node runtime arg with Wt/NC as compile-time.
//   - The H factory varies Wt per core group (work splits along W cols) and binds
//     Wt as a per-node runtime arg with Ht/NC as compile-time.
//   A single compute source can't satisfy both, hence H gets its own kernel.

#include <cstdint>

#include "experimental/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    // Per-node runtime: column count assigned to this worker core (varies per group).
    const uint32_t Wt = get_arg(args::Wt);

    // Compile-time: same for every worker core.
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t NC = get_arg(args::NC);

    experimental::DataflowBuffer dfb_input(dfb::input);
    experimental::DataflowBuffer dfb_scaler(dfb::scaler);
    experimental::DataflowBuffer dfb_output(dfb::output);

    compute_kernel_hw_startup(dfb_input.get_id(), dfb_scaler.get_id(), dfb_output.get_id());

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        dfb_input,
        dfb_scaler,
        dfb_output,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
        // with scaler=1.0 and we apply the user scalar via mul_unary_tile (SFPU) on each output.
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
            binop_with_scalar_tile_init();
            mul_unary_tile(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
}
