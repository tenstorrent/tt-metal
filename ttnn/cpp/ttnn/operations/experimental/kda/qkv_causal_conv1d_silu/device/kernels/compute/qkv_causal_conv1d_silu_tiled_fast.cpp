// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute of the tiled qkv_causal_conv1d_silu path with the fused q/k L2 norm
// (program_config.fused_qk_l2_norm = True; the default uses qkv_causal_conv1d_silu_tiled.cpp, unchanged).
//
// Differences from the default N1 kernel:
//  1. The 4 taps accumulate in dest: ELWMUL on WH/BH adds into dest (the eltwise-binary MOP does not
//     zero dest in the non-reuse path; the packer zeroes it on release), so
//     dest = S_3*w0 + S_2*w1 + S_1*w2 + X*w3 with no partial packs and no dest-reuse adds. The
//     `partial` DFB stays bound but is unused. The accumulation order is the same as N1, but the
//     partial sums stay in dest (bf16 or fp32 per fp32_dest_acc_en) instead of being packed to bf16.
//  2. The SiLU on the pack thread is a hand-scheduled TTI sequence that runs two rows interleaved
//     (calculate_silu_fast2): exp(-x) with the exp_21f_bf16_tti body (clamped at 2^127), + 1, SFPARECIP, one
//     Newton step, * x. It is not bit-identical to silu_tile (op-level PCC vs fp32 torch 0.9999984 with fp32 dest).
//  3. q and k are L2-normalized per 128-channel head (B = 4 = one head per step) and token, exactly as
//     ChunkGdnFused's in-kernel QK_NORM does it (chunk_gdn_math.hpp prep_chunk / inv_rms):
//       q_n = q * rsqrt(sum_head q^2 + eps) * scale,   k_n = k * rsqrt(sum_head k^2 + eps),
//     with eps = QKV_CONV_QK_EPS (1e-6) and scale = QKV_CONV_Q_SCALE (1/sqrt(head_dim)), from the bf16-rounded
//     SiLU output. Column blocks < QKV_CONV_Q_BLOCKS are q, < QKV_CONV_QK_BLOCKS are k, the rest (v) is unchanged.
//     Per q/k step: SiLU -> ybuf (bf16); sum_t y_t^2 in dest (ELWMUL accumulates) -> sq (fp32);
//     sq @ ones -> row sums in dest -> rsqrt(+eps)[*scale] on the pack thread's SFPU (faces 0, 2) -> rn (fp32);
//     y_t * rn (column broadcast) -> out32. All SFPU work stays on the pack thread (the SiLU's LRegs).
//     The normalized q/k tiles are packed in fp32 into `out32` (the writer sends them to the fp32 q/k tensors);
//     v stays bf16 in `out`.
// So q/k/v are NOT bit-identical to the default kernel.

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef TRISC_PACK
namespace ckernel::sfpu {
// Two-row interleaved fast SiLU: rows d and d+1 of a face run the same one-Newton-step SiLU sequence
// on disjoint LRegs (row A: LREG0-3, row B: LREG4-7), instruction by instruction, so
// each instruction's latency is covered by the other row's instruction. To free LRegs: x is reloaded from
// dest for the final multiply, the exp bias 127 is an SFPADDI immediate, the Newton step uses
// s' = s - s*(den*s - 1) (no 2.0 constant), and the exp polynomial's c0 (1.001953, fp16a in exp_21f_bf16_tti) is
// 1 + an SFPADDI of 2^-9, so c0, c1 and c2 are exactly the exp_21f_bf16_tti constants.
// Constants: LREG12 = 1/ln2, LREG13 = c2, LREG14 = c1.
// With fp32 dest (DST_ACCUM_MODE) the bf16 rounding is left to the packer (same op-level PCC, -1 instruction/row).
template <int ITERATIONS>
inline void calculate_silu_fast2() {
    static_assert(ITERATIONS % 2 == 0);
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 1);
        TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG7, 1);
        TTI_SFPADDI(0x42fe, p_sfpu::LREG3, 0);
        TTI_SFPADDI(0x42fe, p_sfpu::LREG7, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x437e);
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x437e);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG7, p_sfpu::LREG5, 0);
        TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        TTI_SFPEXMAN(0, p_sfpu::LREG7, p_sfpu::LREG4, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPEXMAN(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG14, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG13, p_sfpu::LREG14, p_sfpu::LREG6, 0);
        TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 8);
        TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG7, 8);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG5, 0);
        TTI_SFPADDI(0x3b00, p_sfpu::LREG1, 0);  // + (c0 - 1) = 2^-9: c0 = 1.001953 as in exp_21f_bf16_tti
        TTI_SFPADDI(0x3b00, p_sfpu::LREG5, 0);
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, 1);
        TTI_SFPAND(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG4, 1);
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);  // e = exp(-x)
        TTI_SFPSETEXP(0, p_sfpu::LREG5, p_sfpu::LREG4, 2);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);  // den = e + 1
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);  // s
        TTI_SFPARECIP(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 2);  // u = den*s - 1
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, 2);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG1, 1);  // s = s - s*u
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG5, 1);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);  // x again
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // x * s
        TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        if constexpr (!DST_ACCUM_MODE) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                p_sfpu::LREG4,
                p_sfpu::LREG4,
                p_sfpu::LREG4,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        sfpi::dst_reg += 2;
    }
}
inline void silu_fast2_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = 4.791750143340323e-15f;
    sfpi::vConstFloatPrgm2 = 7.839635491371155e-08f;
}
template <int ITERATIONS>
inline void qk_fill_one() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat one = 1.0f;
        sfpi::dst_reg[0] = one;
        sfpi::dst_reg++;
    }
}
// dst = rsqrt(dst + eps) [* scale]: magic-constant seed + 3 Newton steps (fp32).
template <bool SCALE, int ITERATIONS>
inline void qk_inv_norm() {
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat s = sfpi::dst_reg[0];
        s = s + QKV_CONV_QK_EPS;
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vUInt(0x5f3759df) - (sfpi::as<sfpi::vUInt>(s) >> 1));
        sfpi::vFloat h = s * 0.5f;
        y = y * (1.5f - h * y * y);
        y = y * (1.5f - h * y * y);
        y = y * (1.5f - h * y * y);
        if constexpr (SCALE) {
            y = y * QKV_CONV_Q_SCALE;
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

template <uint32_t block_tiles, uint32_t Mt>
TT_KERNEL void compute(uint32_t step_start, uint32_t step_count) {
    constexpr uint32_t B = block_tiles;
    constexpr uint32_t tap_count = 4;
    // Shift slots of one step: S_1 at i, S_2 at B + i, S_3 at 2B + i. Weight slots: tap t at t*B + i.
    constexpr uint32_t s1_slot = 0;
    constexpr uint32_t s2_slot = B;
    constexpr uint32_t s3_slot = 2 * B;

    compute_kernel_hw_startup(dfb::shift, dfb::weights, dfb::out);
    DataflowBuffer x_in(dfb::x_in);
    DataflowBuffer shift(dfb::shift);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer out(dfb::out);
    constexpr uint32_t shift_step = 3 * B;
    silu_tile_init_pack();
    DataflowBuffer ybuf(dfb::ybuf);
    DataflowBuffer out32(dfb::out32);
    DataflowBuffer yq(dfb::yq);
    DataflowBuffer sq(dfb::sq);
    DataflowBuffer rn(dfb::rn);
    DataflowBuffer ones(dfb::ones);
    // The ones tile for the row-sum matmul, built once on the pack thread's SFPU.
    tile_regs_acquire();
    tile_regs_commit();
    ones.reserve_back(1);
    tile_regs_wait();
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, qk_fill_one, (8), 0, VectorMode::RC));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile(0, dfb::ones);
    tile_regs_release();
    ones.push_back(1);
    ones.wait_front(1);
    // Software-pipelined q/k epilogue: each pending q/k step advances one stage per call of qk_advance, so S1's
    // and S2's inputs were produced one iteration earlier and the math and pack threads rarely wait on each other:
    //   S0 (taps + SiLU -> ybuf), S1 (sum_t y_t^2 -> sq), then S2 (sq @ ones, rsqrt -> rn) + S3 (y_t * rn -> out32).
    // ybuf holds up to 3 steps (4B entries allocated), sq and rn 2 each. qk_pend is oldest first; qk_stage =
    // stages done (1 after S0, 2 after S1, 4 after S2 + S3).
    uint32_t qk_pend[4];
    uint32_t qk_stage[4];
    uint32_t qk_npend = 0;
    auto qk_stage0 = [&](uint32_t s) {
        // Math committed the taps of step s. Pack thread: SiLU in dest, then pack y to ybuf.
        pack_reconfig_data_format(dfb::ybuf);
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t i = 0; i < B; ++i) {
            PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu_fast2, (8), i, VectorMode::RC));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        ybuf.reserve_back(B);
        yq.reserve_back(B);
        for (uint32_t i = 0; i < B; ++i) {
            pack_tile(i, dfb::ybuf);
        }
        for (uint32_t i = 0; i < B; ++i) {
            pack_tile(i, dfb::yq);
        }
        tile_regs_release();
        ybuf.push_back(B);
        yq.push_back(B);
        qk_pend[qk_npend] = s;
        qk_stage[qk_npend] = 1;
        ++qk_npend;
    };
    // S1 of the pending step at ybuf position k: dest0 = sum_t y_t * y_t -> sq (fp32).
    // S1 reads the yq copy of y from its FIFO front: an unpack read window at a non-front offset can run past the
    // CB ring's end (reads are not wrapped), so S3's ybuf and S1's yq are separate FIFOs.
    auto qk_s1 = [&](uint32_t k) {
        (void)k;
        yq.wait_front(B);
        tile_regs_acquire();
        reconfig_data_format(dfb::yq, dfb::yq);
        mul_init(dfb::yq, dfb::yq);
        for (uint32_t i = 0; i < B; ++i) {
            mul_tiles(dfb::yq, dfb::yq, i, i, 0);
        }
        tile_regs_commit();
        yq.pop_front(B);
        sq.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::sq);
        pack_tile(0, dfb::sq);
        tile_regs_release();
        sq.push_back(1);
    };
    // S2: dest0 = sq @ ones (row sums in every column) -> rsqrt(+eps) [* scale] -> rn (fp32).
    auto qk_s2 = [&](uint32_t s) {
        sq.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format(dfb::ones, dfb::sq);
        matmul_init(dfb::sq, dfb::ones, 0);
        matmul_tiles(dfb::sq, dfb::ones, 0, 0, 0);
        tile_regs_commit();
        sq.pop_front(1);
        rn.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::rn);
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        if (s / Mt < QKV_CONV_Q_BLOCKS) {
            PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, qk_inv_norm, (true, 8), 0, VectorMode::C));
        } else {
            PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, qk_inv_norm, (false, 8), 0, VectorMode::C));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile(0, dfb::rn);
        tile_regs_release();
        rn.push_back(1);
    };
    // S3 of the oldest pending step: out_t = y_t * r (column 0 of rn broadcast) -> out32 (fp32).
    auto qk_s3 = [&]() {
        rn.wait_front(1);
        ybuf.wait_front(B);
        tile_regs_acquire();
        reconfig_data_format(dfb::ybuf, dfb::rn);
        mul_bcast_cols_init(dfb::ybuf, dfb::rn);
        for (uint32_t i = 0; i < B; ++i) {
            mul_tiles_bcast_cols(dfb::ybuf, dfb::rn, i, 0, i);
        }
        tile_regs_commit();
        ybuf.pop_front(B);
        rn.pop_front(1);
        out32.reserve_back(B);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out32);
        for (uint32_t i = 0; i < B; ++i) {
            pack_tile(i, dfb::out32);
        }
        tile_regs_release();
        out32.push_back(B);
    };
    // Lag-3 pipeline: every pending q/k step except fresh (the step whose S0 ran in this iteration) advances one
    // stage, newest first: S1 (mul) of step j-1, S2 (matmul) of j-2, S3 (bcast-cols mul) of j-3. Each stage's
    // input was produced by the pack thread in the previous iteration, so math never waits for the SiLU of step j.
    // Capacities: ybuf holds j-3..j (4 steps = 4B entries), sq and rn at most 2 each. S1 of the step at pending
    // position r reads ybuf position r (all older ybufs are still live); S2/S3 consume the sq/rn/ybuf fronts.
    // Outputs leave in step order (S3 in step order; v steps drain first), so the writer's in-order waits hold.
    auto qk_advance = [&](uint32_t fresh) {
        for (int32_t r = static_cast<int32_t>(qk_npend) - 1; r >= 0; --r) {
            if (qk_pend[r] == fresh) {
                continue;
            }
            const uint32_t st = qk_stage[r];
            if (st == 1) {
                qk_s1(static_cast<uint32_t>(r));
            } else if (st == 2) {
                qk_s2(qk_pend[r]);
            } else {
                qk_s3();
            }
            qk_stage[r] = st + 1;
        }
        if (qk_npend > 0 && qk_stage[0] == 4) {
            for (uint32_t r = 1; r < qk_npend; ++r) {
                qk_pend[r - 1] = qk_pend[r];
                qk_stage[r - 1] = qk_stage[r];
            }
            --qk_npend;
        }
    };
    PACK((ckernel::sfpu::silu_fast2_init()));

    const uint32_t step_end = step_start + step_count;
    uint32_t step = step_start;
    while (step < step_end) {
        // One unit: the steps of this core range that share a column block (one tap load).
        const uint32_t mt0 = step % Mt;
        const uint32_t unit_end = step + (Mt - mt0) < step_end ? step + (Mt - mt0) : step_end;
        weights.wait_front(tap_count * B);
        for (; step < unit_end; ++step) {
            const bool is_qk = step / Mt < QKV_CONV_QK_BLOCKS;
            if (!is_qk) {
                // A v step: its output must follow the pending q/k steps' outputs (the writer writes in step
                // order). Drain before this step's taps take a dest half (every thread keeps the same order).
                while (qk_npend > 0) {
                    qk_advance(step_end);
                }
            }
            // The 4 taps accumulate in dest (see the header).
            shift.wait_front(shift_step);
            x_in.wait_front(B);
            tile_regs_acquire();
            reconfig_data_format(dfb::x_in, dfb::weights);
            mul_bcast_rows_init(dfb::shift, dfb::weights);
            for (uint32_t tap = 0; tap < 3; ++tap) {
                const uint32_t shift_slot = tap == 0 ? s3_slot : (tap == 1 ? s2_slot : s1_slot);
                for (uint32_t i = 0; i < B; ++i) {
                    mul_tiles_bcast_rows(dfb::shift, dfb::weights, shift_slot + i, tap * B + i, i);
                }
            }
            mul_bcast_rows_init(dfb::x_in, dfb::weights);
            for (uint32_t i = 0; i < B; ++i) {
                mul_tiles_bcast_rows(dfb::x_in, dfb::weights, i, 3 * B + i, i);
            }
            tile_regs_commit();
            shift.pop_front(shift_step);
            x_in.pop_front(B);
            if (is_qk) {
                qk_stage0(step);
                qk_advance(step);
                continue;
            }
            pack_reconfig_data_format(dfb::out);
            out.reserve_back(B);
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t i = 0; i < B; ++i) {
                PACK(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu_fast2, (8), i, VectorMode::RC));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t i = 0; i < B; ++i) {
                pack_tile(i, dfb::out);
            }
            tile_regs_release();
            out.push_back(B);
        }
        weights.pop_front(tap_count * B);
    }
    while (qk_npend > 0) {
        qk_advance(step_end);
    }
    ones.pop_front(1);
}
