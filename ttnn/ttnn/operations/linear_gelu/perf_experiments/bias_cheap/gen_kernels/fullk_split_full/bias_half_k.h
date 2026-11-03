// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bias_cheap experiment: the bias K step (ones tile x bias tile) as a half-K tile-op.
//
// Helper bypass: ckernel::matmul_block / llk_math_matmul runs the full 32x32x32 MOP (16 MVMULs per fidelity phase).
// For the bias step the in0 ones tile is nonzero only in k 0..15 (faces 0 / 2) and the bias tile only in rows 0..15
// (faces 0 / 1), so the second half of the recorded MVMUL sequence (B1A2, B1A3, B3A2, B3A3) adds exact zeros. This
// replays only the first 7 recorded MVMULs and issues the 8th (B2A1, second 8 rows) with ADDR_MOD_5 (the reset +
// fidelity-increment the 16th MVMUL carries), once per fidelity phase, then the MOP's end op. Same instructions, same
// fidelity phases, same operands: the products that remain are bit-identical; only the zero adds are skipped.
// The unpack side is the unchanged llk_unpack_AB_matmul call (full tiles), so no init / reconfig is needed.
//
// Assumes the unthrottled, non-transposed, full 32x32 MOP (recorded by matmul_block_init at replay_buf_offset);
// falls back to the library math path whenever the FW throttle has the throttled MOP programmed.
#pragma once

#include <cstdint>

#include "api/compute/matmul.h"

#ifdef TRISC_MATH
namespace bias_cheap {

template <MathFidelity math_fidelity>
inline void math_half_k(const std::uint32_t dst_index, const std::uint32_t ct_dim, const std::uint32_t rt_dim) {
    using namespace ckernel;
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t phases = high_fidelity ? to_underlying(math_fidelity) : 1;
    const bool reuse_a = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim;
    for (std::uint32_t t = 0; t < t_dim; t++) {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++) {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(
                dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));
            for (std::uint32_t phase = 0; phase < phases; phase++) {
                lltt::replay(math::replay_buf_offset, 7);
                if constexpr (high_fidelity) {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
                } else if (reuse_a) {
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0);
                } else {
                    TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
                }
            }
            if constexpr (high_fidelity) {
                if (reuse_a) {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                } else {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
            }
            if (rut == rut_dim - 1) {
                if (reuse_a) {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                } else {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
            }
        }
    }
}

}  // namespace bias_cheap
#endif

// matmul_block K-loop bound: the x / W K tiles only; the bias step is the post-compute below.
struct BiasInnerDim {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t block_k) const { return block_k - 1; }
};

// matmul_block post_compute: runs once per output subblock (in0 subblock outer, in1 subblock inner), after the
// K-loop and before tile_regs_commit. Unpacks the ones tile / bias tile pair exactly like the K-loop does for
// k = k_ext - 1, then multiplies it with the half-K sequence.
struct BiasPostCompute {
    uint32_t in0_cb;
    uint32_t in1_cb;
    uint32_t in1_base;
    uint32_t sb_h;
    uint32_t sb_w;
    uint32_t in1_per_core_w;
    uint32_t k_ext;
    uint32_t in0_sub = 0;
    uint32_t in1_sub = 0;

    ALWI void operator()(uint32_t /*out_subblock_num_tiles*/) {
        const uint32_t in0_index = in0_sub * sb_h * k_ext + (k_ext - 1);
        const uint32_t in1_index = in1_base + in1_sub * sb_w + (k_ext - 1) * in1_per_core_w;
        UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_index, in1_index, sb_w, sb_h, k_ext)));
#ifdef BIAS_FULL_K
        MATH((ckernel::matmul_block_math_dynamic_throttle(in0_cb, in1_cb, 0, false, sb_w, sb_h)));
#else
        MATH(([&] {
            const bool throttle_req = (*ckernel::throttle_ptr % 2) != 0;
            if (MM_THROTTLE != 0 || throttle_req || ckernel::throttled_mop_status != 0) {
                ckernel::matmul_block_math_dynamic_throttle(in0_cb, in1_cb, 0, false, sb_w, sb_h);
            } else {
                bias_cheap::math_half_k<MATH_FIDELITY>(0, sb_w, sb_h);
            }
        }()));
#endif
        if (++in1_sub * sb_w == in1_per_core_w) {
            in1_sub = 0;
            ++in0_sub;
        }
    }
};
