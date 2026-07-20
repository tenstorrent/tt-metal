// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"        // BinaryFpu, CopyTile, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;
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
        ckl::unary<ckl::Recip<D::D0>, cb_divisor, cb_divisor_recip, ckl::input(ckl::InputLifecycle::Bulk)>(
            ckl::EltwiseShape::single());
    }

    constexpr auto weight_mul = ckl::OptionalChainElement<
        has_weight,
        ckl::DestReuseBinary<cb_tmp_weight, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>>{};

    constexpr auto negate = ckl::Negative<D::D0>{};
    constexpr auto pack_out = ckl::PackTile<cb_output>{};

    if constexpr (has_divisor) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(per_core_tile_cnt),
            ckl::BinaryFpu<
                cb_tmp_input,
                cb_divisor_recip,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::Bulk)>{},
            negate,
            weight_mul,
            pack_out);
    } else {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(per_core_tile_cnt), ckl::CopyTile<cb_tmp_input>{}, negate, weight_mul, pack_out);
    }
}
