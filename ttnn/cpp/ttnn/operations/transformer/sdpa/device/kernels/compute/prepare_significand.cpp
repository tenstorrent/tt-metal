// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int BITS, int ITERATIONS>
inline void round_lofi_significand() {
    constexpr int shift = 24 - BITS;
    constexpr int bias = (1 << (shift - 1)) - 1;
    constexpr int mask = ~((1 << shift) - 1);
#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vInt raw = sfpi::as<sfpi::vInt>(x);
        sfpi::vInt odd = (raw >> shift) & 1;
        raw = (raw + bias + odd) & mask;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(raw);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t bits = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    // BF16 copy_init preserves Src zero flags for MOV, but FP32-DST datacopy
    // uses ELWADD. In asymmetric keep-SrcB-denorm mode its nominal zero
    // operand is tiny/nonzero, which perturbs RNE ties for very small inputs.
    // This preprocessor supports ordinary normal BF16 values, not subnormals.
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    negative_tile_init();
    for (uint32_t i = 0; i < count; i += batch) {
        cb_wait_front(0, batch);
        cb_reserve_back(16, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(0, j, j);
            MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, round_lofi_significand, (bits, 8), j, VectorMode::RC));
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            pack_tile(j, 16);
        }
        tile_regs_release();
        cb_push_back(16, batch);
    }
}
