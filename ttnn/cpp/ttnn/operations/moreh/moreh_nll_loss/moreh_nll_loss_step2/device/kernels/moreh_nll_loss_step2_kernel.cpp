// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"        // BinaryFpu, CopyTile, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"    // Optional
namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t dfb_tmp_weight_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_tmp_input_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_divisor_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_divisor_recip_id = tt::CBIndex::c_27;
    constexpr uint32_t dfb_output_id = tt::CBIndex::c_16;
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

    compute_kernel_hw_startup(dfb_tmp_weight_id, dfb_tmp_input_id, dfb_output_id);

    if constexpr (has_divisor) {
        ckl::unary<
            ckl::Recip<D::D0>,
            ckl::input(dfb_divisor_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_divisor_recip_id)>(ckl::IterationShape::one_tile());
    }

    // multiply weight
    constexpr auto weight_mul = ckl::Optional<
        has_weight,
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Mul, ckl::input(dfb_tmp_weight_id), ckl::DestReuseType::DEST_TO_SRCA>>{};

    constexpr auto negate = ckl::Negative<D::D0>{};
    constexpr auto pack_out = ckl::PackTile<ckl::output(dfb_output_id)>{};

    if constexpr (has_divisor) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_tmp_input_id),
                ckl::input(
                    dfb_divisor_recip_id,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd)>{},
            negate,
            weight_mul,
            pack_out);
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::CopyTile<ckl::input(dfb_tmp_input_id)>{},
            negate,
            weight_mul,
            pack_out);
    }
}
