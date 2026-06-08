// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_ternary.hpp"   // Addcmul, Addcdiv
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

// addcmul / addcdiv (SFPU): out = in0 + value * (in1 OP in2).
//   copy in0/in1/in2 -> D0/D1/D2 ; addc{mul,div}_tile<DF>(0,1,2,0,value) ; pack D0.
// The op + dtype were a string macro (TERNARY_SFPU_OP_FUNC); now the program
// factory emits TERNARY_OP_SEL (2=addcmul, 3=addcdiv) + TERNARY_DF, and the
// kernel selects the concrete chain element via OptionalChainElement. Lifecycle:
// 3 Streaming inputs (wait1/pop1) + 1 Streaming output — matches the original.
#ifndef TERNARY_OP_SEL
#error "ternary_addc_ops_sfpu requires TERNARY_OP_SEL/TERNARY_DF from get_compute_defines"
#endif

namespace ckl = compute_kernel_lib;
constexpr int kSel = TERNARY_OP_SEL;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);  // caller-owned BIG init

    ckl::eltwise_chain(
        num_tiles,
        ckl::CopyTile<cb_in0, ckl::Dst::D0, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_in1, ckl::Dst::D1, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_in2, ckl::Dst::D2, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        // exactly one active (the inactive folds to a no-op tag, swallowing scalar_arg).
        ckl::OptionalChainElement < kSel == 2,
        ckl::Addcmul < TERNARY_DF,
        ckl::Dst::D0,
        ckl::Dst::D1,
        ckl::Dst::D2,
        ckl::Dst::D0 >> {scalar_arg},
        ckl::OptionalChainElement < kSel == 3,
        ckl::Addcdiv < TERNARY_DF,
        ckl::Dst::D0,
        ckl::Dst::D1,
        ckl::Dst::D2,
        ckl::Dst::D0 >> {scalar_arg},
        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
