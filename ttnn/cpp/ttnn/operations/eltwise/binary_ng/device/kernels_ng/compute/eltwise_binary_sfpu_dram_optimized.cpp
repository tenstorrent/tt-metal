// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/binary_comp.h"

#include "lltt.h"
#include "sfpi.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

#include "tools/profiler/kernel_profiler.hpp"

namespace experimental {

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU elementwise multiplication.
//
// The standard `mul_binary_tile` expands (per tile) to 4 faces x 8 sfpi
// iterations of compiled SFPU instructions. That sequence is identical from
// one tile to the next for a given precision mode; the only thing that needs
// to change per tile is the DST base. We therefore record one sfpi-iteration
// body once at init time into the MATH-thread replay buffer and replay it at
// runtime, matching the output of the FPU-based `BINARY_OP` (eltwise multiply)
// used in `eltwise_binary_dram_optimized.cpp`:
//   * multiply in FP32 inside the SFPU
//   * when dest is BF16, software RNE FP32 -> BF16 and clamp 0*x = x*0 = 0
//     so the result matches the FPU's BF16 multiply bit-exactly
//   * when dest is FP32, skip rounding/clamping
//
// Layout baked into the recorded body (offsets are in dest rows, relative to
// the current dst_reg base set by `_llk_math_eltwise_binary_sfpu_start_`):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1], next tile slot)
//   result overwrites offset 0
// The body ends with INCRWC(SFP_DESTREG_STRIDE) mirroring `sfpi::dst_reg++`.
//
// The per-tile wrapper preserves the 4-face / 2 x SETRWC face-advance
// structure of `_llk_math_eltwise_binary_sfpu_params_` but seeds the DST
// base with `idst0` so the fixed offsets land on DST[idst0] and
// DST[idst0 + 1]. Callers must therefore use the same pairing the kernel
// already relies on: `mul_binary_tile_replay(i*2, i*2 + 1, i*2)`.
// ---------------------------------------------------------------------------

// Number of instructions in one recorded sfpi-iteration body.
//   is_fp32_dest_acc_en = true  : SFPLOAD x2 + SFPMUL + SFPNOP + SFPSTORE + INCRWC = 6
//   is_fp32_dest_acc_en = false : + SFP_STOCH_RND + 2 * (SFPSETCC + SFPMOV + SFPENCC) = 13
constexpr std::uint32_t SFPU_BINARY_MUL_REPLAY_LEN = DST_ACCUM_MODE ? 6 : 13;

// A 32x32 tile occupies 64 rows in dest.
constexpr std::uint32_t SFPU_BINARY_MUL_DST_TILE_ROWS = 64;

// These helpers reference LLK symbols (e.g. `_llk_math_eltwise_binary_sfpu_start_`)
// which are only visible on the MATH thread. Gate the definitions accordingly so
// UNPACK/PACK TUs do not try to parse them.
#ifdef TRISC_MATH

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _program_sfpu_binary_mul_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_REPLAY_LEN);

    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPNOP;

    if constexpr (!is_fp32_dest_acc_en) {
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPENCC(0, 0, 0, 0);
    }

    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _llk_math_eltwise_binary_sfpu_binop_mul_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, SFPU_BINARY_MUL_REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

#endif  // TRISC_MATH

// Program the MATH-thread replay buffer with one sfpi-iteration body of the
// SFPU multiply. Must be called once, after `BINARY_SFPU_INIT`, so the SFPU
// is already configured. Uses `lltt::NoExec` so recording does not emit any
// actual SFPU work - only the recorded instructions are installed.
ALWI void sfpu_binary_init_mop() { MATH((_program_sfpu_binary_mul_replay_<APPROX, DST_ACCUM_MODE>())); }

// Drop-in replacement for `mul_binary_tile(idst0, idst1, odst)` that performs
// the multiply by replaying the pre-recorded body. Requires `idst1 == idst0 + 1`
// and `odst == idst0`, which matches the kernel's `(i*2, i*2 + 1, i*2)` layout.
ALWI void mul_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_binop_mul_replay_(idst0)));
}

}  // namespace experimental

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_batch = get_arg_val<uint32_t>(1);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
    experimental::sfpu_binary_init_mop();
#endif

    uint32_t remaining = num_tiles;
    while (remaining > 0) {
        uint32_t n_tiles;
        if (remaining >= num_tiles_per_batch) {
            n_tiles = num_tiles_per_batch;
        } else {
            n_tiles = remaining;
        }

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n_tiles);
        cb_wait_front(cb_post_lhs, n_tiles);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n_tiles);
        cb_wait_front(cb_post_rhs, n_tiles);

        cb_reserve_back(cb_out, n_tiles);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
        experimental::sfpu_binary_init_mop();
#endif
        tile_regs_acquire();
        {
            DeviceZoneScopedN("copy_tile_to_dst_init_short_with_dt");
            copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
            for (uint32_t i = 0; i < n_tiles; ++i) {
                copy_tile(cb_post_lhs, i, i * 2);
            }
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        {
            DeviceZoneScopedN("copy_tile and compute");
            for (uint32_t i = 0; i < n_tiles; ++i) {
                copy_tile(cb_post_rhs, i, i * 2 + 1);

#if HAS_ACTIVATIONS(POST)
                BINARY_SFPU_INIT
                experimental::sfpu_binary_init_mop();
#endif
                experimental::mul_binary_tile_replay(i * 2, i * 2 + 1, i * 2);
                // BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

                PROCESS_POST_ACTIVATIONS(i * 2);
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        {
            DeviceZoneScopedN("pack_tile");
            for (uint32_t i = 0; i < n_tiles; ++i) {
                pack_tile(i * 2, cb_out);
            }
            tile_regs_release();
        }

        cb_push_back(cb_out, n_tiles);
        cb_pop_front(cb_post_lhs, n_tiles);
        cb_pop_front(cb_post_rhs, n_tiles);
        remaining -= n_tiles;
    }
}
