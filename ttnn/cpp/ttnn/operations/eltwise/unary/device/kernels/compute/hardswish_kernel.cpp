#include "api/compute/eltwise_unary/hardswish_tt_poly_bf16.h"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional

namespace ckl = compute_kernel_lib;

constexpr bool kIsFloat32 = get_compile_time_arg_val(0) == 1;
constexpr bool kIsInt = get_compile_time_arg_val(1) == 1;
constexpr bool kIsFloat = !kIsFloat32 && !kIsInt;

#if defined(TT_POLY_BF16_UNARY_CONTEXT) && TT_POLY_BF16_UNARY_CONTEXT == 1 && !defined(TT_POLY_LLK_DISABLE) && \
    (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
struct TTPolyGenerated : ckl::UnaryOp<TTPolyGenerated, ckl::Dst::D0> {
    // The evaluator may use fixed destination rows beyond its input tile.
    // Reserve the entire existing window so another lane cannot overlap them.
    static constexpr uint32_t lane_width = ckl::DEST_AUTO_LIMIT;
    static constexpr uint32_t max_dst() { return ckl::DEST_AUTO_LIMIT - 1; }
    static ALWI void init() {
        ckl::Hardsigmoid<ckl::Dst::D0>::init();
        hardswish_tt_poly_bf16_tile_init();
    }
    static ALWI void exec_impl(uint32_t) { hardswish_tt_poly_bf16_tile(0); }
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
                    dfb_input_id,
                    ckl::WaitPolicy::PerTile,
                    kIsInt ? ckl::PopPolicy::PerTile : ckl::PopPolicy::None,
                    ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{},
            ckl::Hardsigmoid<ckl::Dst::D0>{},
            ckl::Optional<
                kIsFloat32,
                ckl::CopyTile<
                    ckl::input(
                        dfb_input_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::Dst::D1>>{},
            ckl::Optional<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
            ckl::Optional<
                kIsFloat,
                ckl::DestReuseBinary<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_input_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::DestReuseType::DEST_TO_SRCA>>{},
            ckl::PackTile<ckl::output(
                dfb_output_id,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});
    }
}
