// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"     // BinaryFpu, CopyTile, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"      // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"      // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// moreh nll_loss step2:  output = -input [* weight] [* (1/divisor)]
//   input, weight stream 1 tile/iter (full tiles); 1/divisor is a held SCALAR.
//
// recip(divisor) is precomputed once into cb_divisor_recip, then everything else is one chain per
// iteration with no intermediate-CB spill. The three per-tile ops are applied in a chain-friendly
// order (mathematically identical to the original): the held scalar recip folds into the input load
// as a scalar-bcast multiply, then negate, then the full-tile weight multiply (DEST-reuse):
//   -(input * recip) * weight  ==  -input * weight * recip
// weight is gated by an OptionalChainElement; recip forks the load element (scalar-bcast BinaryFpu
// vs plain CopyTile) since DestReuseBinary has no broadcast.
//
// The WEIGHT / DIVISOR factory #defines map to compile-time `constexpr bool`s, branched with
// `if constexpr` directly in kernel_main.
//
// NOTE (pre-existing reduction=mean failures, NOT fixable in this kernel): for a rank-1 input
// (e.g. shape [5,10] → target [5]) the divisor read from cb_divisor is garbage. The divisor is
// `moreh_sum(step1_result)` computed in moreh_nll_loss.cpp, and moreh_sum's full-reduce of a
// rank-1 tensor returns inf/garbage (its reduce kernel needs a 2D [Ht,Wt] tile grid). This kernel
// faithfully divides by whatever divisor it is handed, so it cannot correct it. Upstream
// workaround: reshape step1_result to [1, N] before the divisor moreh_sum in moreh_nll_loss.cpp.
namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;  // 1/divisor (held scalar)
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    using D = ckl::Dst;

#if defined(WEIGHT)
    constexpr bool has_weight = true;
#else
    constexpr bool has_weight = false;
#endif

#if defined(DIVISOR)
    constexpr bool has_divisor = true;
#else
    constexpr bool has_divisor = false;
#endif

    binary_op_init_common(cb_tmp_weight, cb_tmp_input, cb_output);

    if constexpr (has_divisor) {
        // recip(divisor) -> cb_divisor_recip (one tile, consumed Bulk by the loop chain).
        ckl::unary<
            ckl::Recip<D::D0>,
            cb_divisor,
            cb_divisor_recip,
            ckl::CopyTileReconfig::Input,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Bulk,
            ckl::OutputLifecycle::Streaming,
            ckl::PackTileReconfig::Output>(1);
    }

    // Full-tile weight multiply (DEST-reuse), gated on has_weight — collapses to a no-op tag when
    // the caller didn't pass a weight tensor.
    constexpr auto weight_mul = ckl::OptionalChainElement<
        has_weight,
        ckl::DestReuseBinary<
            cb_tmp_weight,
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA,
            D::D0,
            D::D0,
            ckl::DestReuseReconfig::Input,
            ckl::InputLifecycle::Streaming,
            ckl::OperandKind::Scalar>>{};

    constexpr auto negate = ckl::Negative<D::D0>{};
    constexpr auto pack_out =
        ckl::PackTile<cb_output, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{};

    if constexpr (has_divisor) {
        // D0 = input * recip (scalar-bcast); input streamed, recip held (Bulk scalar).
        ckl::eltwise_chain(
            per_core_tile_cnt,
            ckl::BinaryFpu<
                cb_tmp_input,
                cb_divisor_recip,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::Bulk,
                ckl::OperandKind::Scalar,
                D::D0,
                ckl::OperandKind::Scalar>{},
            negate,
            weight_mul,
            pack_out);
    } else {
        ckl::eltwise_chain(
            per_core_tile_cnt,
            ckl::CopyTile<
                cb_tmp_input,
                D::D0,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar,
                ckl::CopyTileReconfig::Input>{},
            negate,
            weight_mul,
            pack_out);
    }
}
