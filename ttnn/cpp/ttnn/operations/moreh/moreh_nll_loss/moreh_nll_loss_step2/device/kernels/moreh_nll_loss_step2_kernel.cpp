// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "api/dataflow/circular_buffer.h"

// moreh nll_loss step2:  output = -input [* weight] [* (1/divisor)]
//   input, weight stream 1 tile/iter (full tiles); 1/divisor is a held SCALAR.
// recip(divisor) is precomputed once into cb_divisor_recip (exactly as the original
// does), then everything else is per-tile. To keep it to one chain per iteration
// with no extra intermediate-CB spill, the three per-tile ops are applied in a
// chain-friendly order (mathematically identical):
//   input * recip      (bcast-scalar mul; only when DIVISOR)
//   negate
//   * weight           (full-tile DestReuse mul; only when WEIGHT)
// i.e. -(input*recip)*weight == -input*weight*recip. weight is a full tile so it
// is applied as a DEST-reuse multiply (no broadcast); recip is the only scalar.
namespace ckl = compute_kernel_lib;

template <bool HasWeight, bool HasDivisor>
inline void run(uint32_t n) {
    using D = ckl::Dst;
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;  // 1/divisor (held scalar)
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    if constexpr (HasDivisor && HasWeight) {
        ckl::eltwise_chain(
            n,
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
            ckl::Negative<D::D0>{},
            ckl::DestReuseBinary<
                cb_tmp_weight,
                ckl::BinaryFpuOp::Mul,
                ckl::DestReuseType::DEST_TO_SRCA,
                D::D0,
                D::D0,
                ckl::DestReuseReconfig::Input,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar>{},
            ckl::PackTile<cb_output, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
    } else if constexpr (HasDivisor && !HasWeight) {
        ckl::eltwise_chain(
            n,
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
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_output, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
    } else if constexpr (!HasDivisor && HasWeight) {
        ckl::eltwise_chain(
            n,
            ckl::CopyTile<
                cb_tmp_input,
                D::D0,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar,
                ckl::CopyTileReconfig::Input>{},
            ckl::Negative<D::D0>{},
            ckl::DestReuseBinary<
                cb_tmp_weight,
                ckl::BinaryFpuOp::Mul,
                ckl::DestReuseType::DEST_TO_SRCA,
                D::D0,
                D::D0,
                ckl::DestReuseReconfig::Input,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar>{},
            ckl::PackTile<cb_output, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
    } else {
        ckl::eltwise_chain(
            n,
            ckl::CopyTile<
                cb_tmp_input,
                D::D0,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar,
                ckl::CopyTileReconfig::Input>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_output, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
    }
}

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    binary_op_init_common(cb_tmp_weight, cb_tmp_input, cb_output);

#if defined(DIVISOR)
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;
    // recip(divisor) -> cb_divisor_recip (one tile, consumed Bulk by the loop chain).
    ckl::eltwise_chain(
        1,
        ckl::CopyTile<
            cb_divisor,
            ckl::Dst::D0,
            ckl::InputLifecycle::Bulk,
            ckl::OperandKind::Scalar,
            ckl::CopyTileReconfig::Input>{},
        ckl::Recip<ckl::Dst::D0>{},
        ckl::
            PackTile<cb_divisor_recip, ckl::Dst::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
#endif

#if defined(WEIGHT) && defined(DIVISOR)
    run<true, true>(per_core_tile_cnt);
#elif defined(WEIGHT)
    run<true, false>(per_core_tile_cnt);
#elif defined(DIVISOR)
    run<false, true>(per_core_tile_cnt);
#else
    run<false, false>(per_core_tile_cnt);
#endif
}
