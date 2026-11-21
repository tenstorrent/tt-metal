// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DestReuseBinary DestOperandFormat: does the DEST -> src move keep DEST's precision?
// Stage 1 D0 = A + B with bf16 A/B on fp32 DEST, so the sum needs more mantissa bits than bf16 has.
// Stage 2 multiplies D0 by an fp32 CB C. DEST_TO_SRCA moves the sum into SrcA:
//   Inherit   -> SrcA keeps A's bf16 format, the sum is truncated to bf16.
//   FromInput -> SrcA takes C's format (Float32 -> Tf32), the sum survives the move.
// DEST_TO_SRCB is the mirror image (C on SrcA, the sum into SrcB).
//
// CT args: [n, reuse (0=SRCA,1=SRCB), dest_format (0=Inherit,1=FromInput), op (0=Add,2=Mul)].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t reuse = get_compile_time_arg_val(1);
    constexpr uint32_t dest_format = get_compile_time_arg_val(2);
    constexpr uint32_t op = get_compile_time_arg_val(3);

    using namespace compute_kernel_lib;
    constexpr DestReuseType R = (reuse == 0) ? DestReuseType::DEST_TO_SRCA : DestReuseType::DEST_TO_SRCB;
    constexpr BinaryFpuOp OP = (op == 0) ? BinaryFpuOp::Add : BinaryFpuOp::Mul;
    constexpr DestOperandFormat F = (dest_format == 0) ? DestOperandFormat::Inherit : DestOperandFormat::FromInput;

    // FromInput programs both sides with the input CB; Inherit only the input-fed side.
    using ToSrcA = DestReuseBinary<
        BinaryFpuOp::Mul,
        input(cb_c),
        DestReuseType::DEST_TO_SRCA,
        Dst::D0,
        DestOperandFormat::FromInput>;
    using ToSrcB = DestReuseBinary<
        BinaryFpuOp::Mul,
        input(cb_c),
        DestReuseType::DEST_TO_SRCB,
        Dst::D0,
        DestOperandFormat::FromInput>;
    static_assert(ToSrcA::reconfig_srca_dfb == cb_c && ToSrcA::reconfig_srcb_dfb == cb_c);
    static_assert(ToSrcB::reconfig_srca_dfb == cb_c && ToSrcB::reconfig_srcb_dfb == cb_c);

    DataflowBuffer cb_a_obj(cb_a);
    DataflowBuffer cb_b_obj(cb_b);
    DataflowBuffer cb_c_obj(cb_c);
    DataflowBuffer cb_out_obj(cb_out);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    eltwise_chain(
        IterationShape::tiles(n),
        BinaryFpu<BinaryFpuOp::Add, input(cb_a), input(cb_b)>{cb_a_obj, cb_b_obj},
        DestReuseBinary<OP, input(cb_c), R, Dst::D0, F>{cb_c_obj},
        PackTile<output(cb_out)>{cb_out_obj});
}
