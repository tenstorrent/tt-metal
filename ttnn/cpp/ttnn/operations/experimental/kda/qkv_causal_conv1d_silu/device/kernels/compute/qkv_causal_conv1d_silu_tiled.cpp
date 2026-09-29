// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute of the tiled qkv_causal_conv1d_silu path, variant N1 (design.md sections 4.3 and 6.4).
//
// Each output tile gets the same FPU/SFPU sequence as the ROW_MAJOR path, so q/k/v are
// bit-identical: HiFi4 ELWMUL with a ROW-broadcast tap into a zeroed bf16 dest, pack the bf16
// partial, dest-reuse ELWADD of the previous partial (DEST_TO_SRCB), precise SiLU on the last tap.
// The tap order and the partial order are the same as there:
//   tap 0: S_3 * w0 -> partial      tap 1: S_2 * w1 + partial -> partial
//   tap 2: S_1 * w2 + partial -> partial      tap 3: silu(X * w3 + partial) -> out
// This kernel runs B tiles per dest acquire and runs each init once per tap, not once per tile.
// All DFBs hold bf16 tiles, so no data format reconfiguration is necessary.
//
// The SiLU runs on the pack thread (silu_tile_pack: the same calculate_silu code as silu_tile, so
// the result is the same), on the dest half that the pack thread owns. The math thread can run
// tap 0 of the next step on the other dest half meanwhile; taps 1-3 of the next step wait for this
// half. At B = 4 the step time goes from 5.46 us (SiLU on the math thread) to 5.23 us: the pack
// thread then bounds the step (SiLU 4.15 us + 4 x 4 tile packs 1.08 us).

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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
    silu_tile_init_pack();

    const uint32_t step_end = step_start + step_count;
    uint32_t step = step_start;
    while (step < step_end) {
        // One unit: the steps of this core range that share a column block (one tap load).
        const uint32_t mt0 = step % Mt;
        const uint32_t unit_end = step + (Mt - mt0) < step_end ? step + (Mt - mt0) : step_end;
        weights.wait_front(tap_count * B);
        for (; step < unit_end; ++step) {
            shift.wait_front(3 * B);

            // Tap 0: partial = S_3 * w0.
            tile_regs_acquire();
            mul_bcast_rows_init(dfb::shift, dfb::weights);
            for (uint32_t i = 0; i < B; ++i) {
                mul_tiles_bcast_rows(dfb::shift, dfb::weights, s3_slot + i, i, i);
            }
            tile_regs_commit();
            partial.reserve_back(B);
            tile_regs_wait();
            for (uint32_t i = 0; i < B; ++i) {
                pack_tile(i, dfb::partial);
            }
            tile_regs_release();
            partial.push_back(B);

            // Taps 1 and 2: partial = S_(3-tap) * w_tap + partial.
            for (uint32_t tap = 1; tap < 3; ++tap) {
                const uint32_t shift_slot = tap == 1 ? s2_slot : s1_slot;
                tile_regs_acquire();
                mul_bcast_rows_init(dfb::shift, dfb::weights);
                for (uint32_t i = 0; i < B; ++i) {
                    mul_tiles_bcast_rows(dfb::shift, dfb::weights, shift_slot + i, tap * B + i, i);
                }
                partial.wait_front(B);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
                for (uint32_t i = 0; i < B; ++i) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, i, i);
                }
                tile_regs_commit();
                partial.pop_front(B);
                partial.reserve_back(B);
                tile_regs_wait();
                for (uint32_t i = 0; i < B; ++i) {
                    pack_tile(i, dfb::partial);
                }
                tile_regs_release();
                partial.push_back(B);
            }
            shift.pop_front(3 * B);

            // Tap 3: out = silu(X * w3 + partial). The math thread stops after the add.
            x_in.wait_front(B);
            tile_regs_acquire();
            mul_bcast_rows_init(dfb::x_in, dfb::weights);
            for (uint32_t i = 0; i < B; ++i) {
                mul_tiles_bcast_rows(dfb::x_in, dfb::weights, i, 3 * B + i, i);
            }
            partial.wait_front(B);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
            for (uint32_t i = 0; i < B; ++i) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, i, i);
            }
            tile_regs_commit();
            partial.pop_front(B);
            x_in.pop_front(B);
            out.reserve_back(B);
            // Pack thread: wait for the math commit (as tile_regs_wait does), point the SFPU at the
            // packer's dest half, run the SiLU, wait for the SFPU, then pack.
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t i = 0; i < B; ++i) {
                silu_tile_pack(i);
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
}
