// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduce (W/H/HW) compute kernel, ported to Metal 2.0.
//
// Host bindings expected (per the W/H/HW factories' compute KernelSpecs):
//   compile_time_arg_bindings: { {"Ht", ...}, {"Wt", ...}, {"NC", ...},
//                                {"post_mul_scaler_bits", ...} (only if REDUCE_POST_MUL) }
//   dfb_bindings: { INPUT (CONSUMER, name="input"),
//                   SCALER (CONSUMER, name="scaler"),
//                   OUTPUT (PRODUCER, name="output") }

#include <cstdint>
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);

    compute_kernel_hw_startup(dfb::input, dfb::scaler, dfb::output);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        dfb::input,
        dfb::scaler,
        dfb::output,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
        // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
        // output DEST register.
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
