// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"   // Where
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_ternary.hpp"   // Lerp, SnakeBeta
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

// where / lerp / snake_beta (TTT, no broadcast): copy in0/in1/in2 -> D0/D1/D2,
// op<DF>(0,1,2,0), pack D0. The op + dtype were a string macro; the factory now
// emits TERNARY_OP_SEL (0=where, 1=lerp, 4=snake_beta) + TERNARY_DF, and the
// kernel selects the concrete chain element via OptionalChainElement. Lifecycle:
// 3 Streaming inputs (wait1/pop1) + 1 Streaming output — matches the original.
#ifndef TERNARY_OP_SEL
#error "ternary_sfpu_no_bcast_ttt requires TERNARY_OP_SEL/TERNARY_DF from get_compute_defines"
#endif

namespace ckl = compute_kernel_lib;
constexpr int kSel = TERNARY_OP_SEL;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);  // caller-owned BIG init

    ckl::eltwise_chain(
        num_tiles,
        ckl::CopyTile<cb_pre_in1, ckl::Dst::D0, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_pre_in2, ckl::Dst::D1, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_pre_in3, ckl::Dst::D2, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::OptionalChainElement < kSel == 0,
        ckl::Where < TERNARY_DF,
        ckl::Dst::D0,
        ckl::Dst::D1,
        ckl::Dst::D2,
        ckl::Dst::D0 >> {},
        ckl::OptionalChainElement < kSel == 1,
        ckl::Lerp < TERNARY_DF,
        ckl::Dst::D0,
        ckl::Dst::D1,
        ckl::Dst::D2,
        ckl::Dst::D0 >> {},
        ckl::OptionalChainElement < kSel == 4,
        ckl::SnakeBeta < TERNARY_DF,
        ckl::Dst::D0,
        ckl::Dst::D1,
        ckl::Dst::D2,
        ckl::Dst::D0 >> {},
        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
