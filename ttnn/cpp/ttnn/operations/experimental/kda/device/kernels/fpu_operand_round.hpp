// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fp32 DEST values rounded, nearest-even, to the precision the FPU's multiplier reads, before they are packed into an
// fp32 CB that an FPU op reads. Unpacking fp32 into SrcA / SrcB keeps 10 mantissa bits (TF32) by truncation, and the
// multiplier reads all 10 of SrcB's but only 9 of SrcA's (the implicit 1 and bits 9..6 in the even fidelity phases,
// bits 5..1 in the odd ones, bit 0 never). Both cuts drop toward zero: an fp32 operand loses about 0.72 * 2^-10 of its
// value on average in SrcA (0.72 * 2^-11 in SrcB), a bias that an eps-dominated norm passes on. Rounded to 9 bits
// first, an operand loses only zeros in either register, and an ELWADD or a copy (all 10 bits) loses nothing either.
// (Infinities stay infinite: the mask clears what the rounding added.)

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
inline void kda_round_operand() {
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        const sfpi::vUInt lsb = (bits >> 14) & 1;
        bits = (bits + 0x1FFFu + lsb) & 0xFFFFC000u;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void kda_round_operand_init() { MATH((SFPU_UNARY_INIT(unused))); }
ALWI void kda_round_operand_tile(uint32_t dst_index, VectorMode vector_mode = VectorMode::RC) {
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, kda_round_operand, dst_index, vector_mode)));
}

namespace kda {
// The rounding as an eltwise_chain element on DEST slot Slot.
template <compute_kernel_lib::Dst Slot = compute_kernel_lib::Dst::D0>
struct RoundOperand : compute_kernel_lib::UnaryOp<RoundOperand<Slot>, Slot> {
    static ALWI void init() { kda_round_operand_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        kda_round_operand_tile(compute_kernel_lib::to_u32(Slot) + slot_offset);
    }
};
}  // namespace kda

// DST tiles [first, first + count) rounded when Round (a no-op otherwise). The SFPU init resets the math
// counters and claims ADDR_MOD_7: a matmul issued after it re-runs its init first.
template <bool Round>
ALWI void kda_round_operand_tiles(uint32_t first, uint32_t count) {
    if constexpr (Round) {
        kda_round_operand_init();
        for (uint32_t i = 0; i < count; ++i) {
            kda_round_operand_tile(first + i);
        }
    }
}
