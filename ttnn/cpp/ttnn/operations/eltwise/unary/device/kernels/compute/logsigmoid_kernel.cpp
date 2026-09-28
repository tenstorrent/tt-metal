#include "api/compute/eltwise_unary/logsigmoid_tt_poly_bf16.h"
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"         // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Logsigmoid

namespace ckl = compute_kernel_lib;

#if defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && !defined(TT_POLY_LLK_DISABLE) && \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
struct TTPolyGenerated : ckl::UnaryOp<TTPolyGenerated, ckl::Dst::D0> {
    // The evaluator may use fixed destination rows beyond its input tile.
    // Reserve the entire existing window so another lane cannot overlap them.
    static constexpr uint32_t lane_width = ckl::DEST_AUTO_LIMIT;
    static constexpr uint32_t max_dst() { return ckl::DEST_AUTO_LIMIT - 1; }
    static ALWI void init() {
        ckl::Negative<ckl::Dst::D1>::init();
        logsigmoid_tt_poly_bf16_tile_init();
    }
    static ALWI void exec_impl(uint32_t) { logsigmoid_tt_poly_bf16_tile(0); }
};
#endif
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

#if defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && !defined(TT_POLY_LLK_DISABLE) && \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
    if constexpr (!DST_ACCUM_MODE) {
#if defined(ARCH_WORMHOLE) && defined(TRISC_MATH)
        ckernel::math::clear_addr_mod_base();
#endif
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(num_tiles),
            ckl::CopyTile<
                ckl::input(
                    dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{},
            TTPolyGenerated{},
            ckl::PackTile<ckl::output(
                dfb_output_id,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});
    } else
#endif
    {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(num_tiles),
            ckl::CopyTile<
                ckl::input(
                    dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{},
            ckl::CopyTile<
                ckl::input(
                    dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>{},
            ckl::Negative<ckl::Dst::D1>{},
            ckl::Exp<ckl::Approx::Fast, ckl::Dst::D1>{},
            ckl::Logsigmoid<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_output_id,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});
    }
}
