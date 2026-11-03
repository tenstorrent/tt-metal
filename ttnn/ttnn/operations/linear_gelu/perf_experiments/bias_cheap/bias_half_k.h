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
//
// BIAS_ONE_UNPACK (unpack the ones / bias pair once per subblock, hold it across the row tiles) is INCORRECT as written
// (PCC 0.67) and measured no extra math saving over the 4-tile unpack, so it was not debugged.
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

// sb_w == 1 path: srcA = bias tile and srcB = ones tile are the same for every row tile of the subblock, so they are
// unpacked once and held (CLR_NONE) across the rt_dim half-K tile-ops; only the last releases both. With the FW
// throttle active the replay buffer holds the throttled sequence, so the same 8 MVMULs are issued directly with NOP
// padding (the address mods are common to both MOPs).
template <MathFidelity math_fidelity>
inline void math_half_k_held(const std::uint32_t dst_index, const std::uint32_t rt_dim, const bool throttled) {
    using namespace ckernel;
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t phases = high_fidelity ? to_underlying(math_fidelity) : 1;
    for (std::uint32_t rut = 0; rut < rt_dim; rut++) {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + rut);
        for (std::uint32_t phase = 0; phase < phases; phase++) {
            if (!throttled) {
                lltt::replay(math::replay_buf_offset, 7);
            } else {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                TTI_NOP;
                TTI_NOP;
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                TTI_NOP;
                TTI_NOP;
            }
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
        }
        if (rut == rt_dim - 1) {
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        } else {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        }
    }
}

}  // namespace bias_cheap
#endif

#ifdef TRISC_UNPACK
namespace bias_cheap {

// One srcA (in1 = bias tile) + one srcB (in0 = ones tile) unpack under the !reuse_a matmul unpack MOP (ct < rt):
// _llk_unpack_AB_matmul_'s body with t_dim = 1 and a single-iteration srcB MOP. The library entry derives reuse_a
// from (ct, rt), so a (1, 1) call would pick the reuse_a branch the programmed MOP does not match.
inline void unpack_bias_pair(
    const std::uint32_t in0_cb,
    const std::uint32_t in1_cb,
    const std::uint32_t in0_index,
    const std::uint32_t in1_index) {
    using namespace ckernel;
    const std::uint32_t a_id = get_operand_id(in0_cb);
    const std::uint32_t b_id = get_operand_id(in1_cb);
    const std::uint32_t address_a =
        get_local_cb_interface(a_id).fifo_rd_ptr - 1 + get_local_cb_interface(a_id).fifo_page_size * in0_index;
    const std::uint32_t address_b =
        get_local_cb_interface(b_id).fifo_rd_ptr - 1 + get_local_cb_interface(b_id).fifo_page_size * in1_index;
    volatile std::uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TT_MOP(0, 0, unp_cfg_context == 0 ? 0 : 0xff);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
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
#ifdef BIAS_ONE_UNPACK
        if (sb_w == 1 && sb_h > 1) {
            UNPACK((bias_cheap::unpack_bias_pair(in0_cb, in1_cb, in0_index, in1_index)));
            MATH((bias_cheap::math_half_k_held<MATH_FIDELITY>(
                0, sb_h, MM_THROTTLE != 0 || (*ckernel::throttle_ptr % 2) != 0 || ckernel::throttled_mop_status != 0)));
        } else
#endif
        {
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
        }
        if (++in1_sub * sb_w == in1_per_core_w) {
            in1_sub = 0;
            ++in0_sub;
        }
    }
};
