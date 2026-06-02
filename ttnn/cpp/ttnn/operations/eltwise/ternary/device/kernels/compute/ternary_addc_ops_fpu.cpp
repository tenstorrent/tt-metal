// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"    // MulUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

// addcmul (FPU): out = in0 + value * (in1 * in2).
//   (in1 * in2) -> D0          BinaryFpu<Mul>, both inputs Streaming/Scalar, pops B,C early
//   [* value when value != 1]  MulUnary<D0> (binop_with_scalar)
//   D0 = in0 + D0              DestReuseBinary<Add, DEST_TO_SRCA>: DEST->srca, cb_in0->srcb,
//                              late wait on A — chain runs wait->exec->pop per element, so the
//                              emitted schedule is wait(B,C)->mul->pop(B,C)->[*v]->wait(A)->add->pop(A),
//                              byte-for-byte the original's early-pop ordering.
//
// `scalar_is_not_1` is a RUNTIME arg, so it drives compile-time dispatch via a bool-
// templated body (eltwise_optional.hpp runtime pattern), not compile-time OptionalChainElement<COND>.
// 1 tile/iter (num_tiles_per_cycle==1): inputs are Streaming + OperandKind::Scalar (pop advances
// the CB front each iter); Block+Streaming would be the absolute-index footgun. No new elements.
namespace ckl = compute_kernel_lib;

template <bool ScalarIsNot1>
inline void run_addcmul(uint32_t num_tiles, uint32_t scalar_arg) {
    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckl::eltwise_chain(
        num_tiles,
        ckl::BinaryFpu<
            cb_in1,
            cb_in2,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            ckl::BinaryDataFormatReconfig::None,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Streaming,
            ckl::OperandKind::Scalar,
            ckl::Dst::D0,
            ckl::OperandKind::Scalar>{},
        ckl::OptionalChainElement<ScalarIsNot1, ckl::MulUnary<ckl::Dst::D0>>{scalar_arg},
        ckl::DestReuseBinary<
            cb_in0,
            ckl::BinaryFpuOp::Add,
            ckl::DestReuseType::DEST_TO_SRCA,
            ckl::Dst::D0,
            ckl::Dst::D0,
            ckl::DestReuseReconfig::Input,
            ckl::InputLifecycle::Streaming,
            ckl::OperandKind::Scalar>{},
        ckl::PackTile<cb_out, ckl::Dst::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // output = input_a + value * input_b * input_c. Caller-owned BIG init.
    binary_op_init_common(cb_in1, cb_in2, cb_out);

    if (scalar_arg != 1u) {
        run_addcmul<true>(num_tiles, scalar_arg);
    } else {
        run_addcmul<false>(num_tiles, scalar_arg);
    }
}
