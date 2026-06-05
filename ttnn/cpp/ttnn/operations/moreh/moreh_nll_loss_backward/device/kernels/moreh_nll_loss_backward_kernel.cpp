// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "api/dataflow/circular_buffer.h"

// moreh nll_loss backward (faithful port — keep the per-tile loop, small chains inside):
//   input_grad = -(weight * output_grad)                [no divisor]
//   input_grad = -(weight * output_grad) * (1/divisor)  [reduction == mean]
//
// output_grad and 1/divisor are scalars (broadcast). The divisor path keeps the
// original's two per-tile stages with the 1-tile intermediate cb_tmp2; each stage is
// its own eltwise_chain(1) inside the C++ tile loop (one tile_regs window each, same
// as the original). cb_tmp2 is produced by stage 1 and consumed by stage 2 every
// iteration, so the single-tile buffer interleaves cleanly. Held scalars
// (output_grad, recip) are waited once externally and read CallerManaged in the
// chains (the bcast_hw idiom for operands held across a tile loop).
namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    using D = ckl::Dst;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;  // scalar (held)
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;  // streams 1 tile/iter
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;        // recip(divisor) (held scalar)
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;        // intermediate -(weight*og)
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    init_sfpu(cb_output_grad, cb_input_grad);

#if defined(DIVISOR)
    // recip(divisor) -> cb_tmp1 (held scalar, one tile).
    ckl::unary<ckl::Recip<D::D0>, cb_divisor, cb_tmp1, ckl::InputLifecycle::Bulk>(1);

    cb_wait_front(cb_tmp1, 1);         // held recip
    cb_wait_front(cb_output_grad, 1);  // held output_grad

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // stage 1: cb_tmp2 = -(weight * output_grad)
        ckl::eltwise_chain(
            1,
            ckl::BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::CallerManaged>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_tmp2>{});

        // stage 2: input_grad = cb_tmp2 * (1/divisor)
        compute_kernel_lib::mul<
            cb_tmp2,
            cb_tmp1,
            cb_input_grad,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::CallerManaged>(1);
    }

    cb_pop_front(cb_output_grad, 1);
    cb_pop_front(cb_tmp1, 1);
#else
    // input_grad = -(weight * output_grad).  Single stage per tile.
    cb_wait_front(cb_output_grad, 1);  // held output_grad

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            1,
            ckl::BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::CallerManaged>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_input_grad>{});
    }

    cb_pop_front(cb_output_grad, 1);
#endif
}
