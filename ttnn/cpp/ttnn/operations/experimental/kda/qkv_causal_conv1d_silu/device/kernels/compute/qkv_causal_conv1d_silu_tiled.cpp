// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute of the tiled qkv_causal_conv1d_silu path, variant N1 (design.md sections 4.3 and 6.4).
//
// Each output tile gets the same FPU/SFPU sequence as the ROW_MAJOR path, so q/k/v are
// bit-identical: HiFi4 ELWMUL with a ROW-broadcast tap into a zeroed bf16 dest, ELWADD of the bf16
// partial (SrcA) and the bf16 product (SrcB), precise SiLU on the last tap.
// The tap order and the partial order are the same as there:
//   tap 0: S_3 * w0 -> partial      tap 1: S_2 * w1 + partial -> partial
//   tap 2: S_1 * w2 + partial -> partial      tap 3: silu(X * w3 + partial) -> out
// Each init runs once per tap, not once per tile. All DFBs hold bf16 tiles, so no data format
// reconfiguration is necessary. The SiLU runs on the pack thread (silu_tile_pack: the same
// calculate_silu code as silu_tile, so the result is the same), on the dest half that the pack thread
// owns; the math thread works on the other dest half meanwhile.
//
// The factory selects one of two flows with QKV_CONV_PARTIALS_IN_DEST:
//
// 1 (bf16 half-sync dest, q/k/v all in L1): the partial stays in dest. For a group of G <= 4 tiles,
//   dest tile i holds the partial of tile i and dest tile G + i the product of the current tap, so
//   the group fills one 16-bit dest half. The add moves the partial into SrcA and the product into
//   SrcB (MOVD2A / MOVD2B; the unpacker only raises the two source data valids) and ELWADDs them into
//   dest tile i. Flow 0 adds the same two bf16 values in the same source registers (its partial is
//   the bf16 pack of the same dest value, unpacked into SrcA; its product is moved from dest to
//   SrcB), so the sums are the same bits. The product tiles are zeroed before taps 2 and 3; for tap 1
//   the dest half is zero from the packer's release, as for every multiply of flow 0. This removes
//   the 3 partial packs and unpacks per tile and their round trip through L1.
//
// 0 (all other cases): the partial goes through the partial DFB: B tiles per dest acquire, pack
//   the bf16 partial, dest-reuse ELWADD of the partial (DEST_TO_SRCB). With DRAM outputs flow 1 lets
//   the output writes congest the NoC (the op gets slower than flow 0), and an fp32 dest would keep
//   fp32 partials that flow 0 rounds to bf16.

#if QKV_CONV_PARTIALS_IN_DEST

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace qkv_conv_tiled {
// bf16 dest: tiles per dest acquire. G partials + G products fill one 16-bit dest half (8 tiles).
constexpr uint32_t max_group_tiles = 4;
constexpr uint32_t dest_rows_per_tile = 64;
}  // namespace qkv_conv_tiled

#ifdef TRISC_MATH
// dest[acc_tile] = dest[acc_tile] + dest[acc_tile + G] (product_row_offset = 64 G): per face, MOVD2A
// the partial face into SrcA, MOVD2B the product face into SrcB (the same moves as
// move_d2a_fixed_face / move_d2b_fixed_face, the second one G tiles further), and run the dest-reuse
// ELWADD MOP (one face per run; it writes dest at the D counter and advances D by one face).
template <uint32_t product_row_offset>
inline void qkv_conv_dest_add_tile(uint32_t acc_tile) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(acc_tile);
#pragma GCC unroll 0
    for (uint32_t face = 0; face < 4; ++face) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 4);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 8);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 12);
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, product_row_offset + 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, product_row_offset + 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, product_row_offset + 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, product_row_offset + 12);
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}

// Zeroes dest tiles [first, first + count) of the math dest half (16-row ZEROACC per face). The D
// counter is 0 here (clear_dst_reg_addr), so row offset + block index stays below the bank size
// (see eltwise_binary_run_with_dest_reuse, tt-metal#53693).
inline void qkv_conv_zero_dest_tiles(uint32_t first, uint32_t count) {
#pragma GCC unroll 0
    for (uint32_t t = first; t < first + count; ++t) {
#pragma GCC unroll 0
        for (uint32_t face = 0; face < 4; ++face) {
            TT_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_1, get_dest_index_in_faces(t, face));
        }
    }
}
#endif

#ifdef TRISC_UNPACK
// The dummy source data valids of one dest + dest add tile: one SrcA and one SrcB per face (the same
// dummy unpack as a dest-reuse binary op, see llk_unpack_a_detail::dest_reuse_dummy_unpack).
inline void qkv_conv_dummy_srcab_tile() {
#pragma GCC unroll 0
    for (uint32_t face = 0; face < 4; ++face) {
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }
}
#endif

template <uint32_t block_tiles, uint32_t Mt>
inline void qkv_conv_compute(uint32_t step_start, uint32_t step_count) {
    using namespace qkv_conv_tiled;
    constexpr uint32_t B = block_tiles;
    constexpr uint32_t tap_count = 4;
    // Shift slots of one step: S_1 at i, S_2 at B + i, S_3 at 2B + i. Weight slots: tap t at t*B + i.
    constexpr uint32_t s1_slot = 0;
    constexpr uint32_t s2_slot = B;
    constexpr uint32_t s3_slot = 2 * B;
    // bf16 dest: group size and groups per step.
    constexpr uint32_t G = B < max_group_tiles ? B : max_group_tiles;
    static_assert(B % G == 0, "block_tiles must be a multiple of the group size");
    constexpr uint32_t groups = B / G;

    static_assert(!DST_ACCUM_MODE, "partials in dest need a bf16 dest (see the header comment)");
    compute_kernel_hw_startup(dfb::shift, dfb::weights, dfb::out);
    DataflowBuffer x_in(dfb::x_in);
    DataflowBuffer shift(dfb::shift);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer out(dfb::out);
    silu_tile_init_pack();

    // Pack thread: wait for the math commit (as tile_regs_wait does), point the SFPU at the packer's
    // dest half, run the SiLU on dest tiles 0..n-1, wait for the SFPU, then pack them to out.
    auto silu_pack_out = [&](uint32_t n) {
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t i = 0; i < n; ++i) {
            silu_tile_pack(i);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();
    };

    const uint32_t step_end = step_start + step_count;
    uint32_t step = step_start;
    while (step < step_end) {
        // One unit: the steps of this core range that share a column block (one tap load).
        const uint32_t mt0 = step % Mt;
        const uint32_t unit_end = step + (Mt - mt0) < step_end ? step + (Mt - mt0) : step_end;
        weights.wait_front(tap_count * B);
        for (; step < unit_end; ++step) {
            shift.wait_front(3 * B);
            x_in.wait_front(B);
            for (uint32_t g = 0; g < groups; ++g) {
                const uint32_t t0 = g * G;
                tile_regs_acquire();

                // Tap 0: partial (dest i) = S_3 * w0.
                mul_bcast_rows_init(dfb::shift, dfb::weights);
                for (uint32_t i = 0; i < G; ++i) {
                    mul_tiles_bcast_rows(dfb::shift, dfb::weights, s3_slot + t0 + i, t0 + i, i);
                }

                // Taps 1-3: product (dest G + i) = S_(3-tap) * w_tap; partial = partial + product.
                for (uint32_t tap = 1; tap < tap_count; ++tap) {
                    if (tap > 1) {
                        MATH((qkv_conv_zero_dest_tiles(G, G)));
                    }
                    if (tap < 3) {
                        const uint32_t shift_slot = (tap == 1 ? s2_slot : s1_slot) + t0;
                        mul_bcast_rows_init(dfb::shift, dfb::weights);
                        for (uint32_t i = 0; i < G; ++i) {
                            mul_tiles_bcast_rows(dfb::shift, dfb::weights, shift_slot + i, tap * B + t0 + i, G + i);
                        }
                    } else {
                        mul_bcast_rows_init(dfb::x_in, dfb::weights);
                        for (uint32_t i = 0; i < G; ++i) {
                            mul_tiles_bcast_rows(dfb::x_in, dfb::weights, t0 + i, tap * B + t0 + i, G + i);
                        }
                    }
                    MATH((llk_math_eltwise_binary_init<
                          EltwiseBinaryType::ELWADD,
                          BroadcastType::NONE,
                          MATH_FIDELITY,
                          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::x_in, dfb::x_in, 0)));
                    for (uint32_t i = 0; i < G; ++i) {
                        MATH((qkv_conv_dest_add_tile<G * dest_rows_per_tile>(i)));
                        UNPACK((qkv_conv_dummy_srcab_tile()));
                    }
                }
                tile_regs_commit();
                out.reserve_back(G);
                silu_pack_out(G);
                out.push_back(G);
            }
            shift.pop_front(3 * B);
            x_in.pop_front(B);
        }
        weights.pop_front(tap_count * B);
    }
}

#else  // QKV_CONV_PARTIALS_IN_DEST

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <uint32_t block_tiles, uint32_t Mt>
inline void qkv_conv_compute(uint32_t step_start, uint32_t step_count) {
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

#endif  // QKV_CONV_PARTIALS_IN_DEST

// The one kernel entry point (the JIT allows exactly one) runs the flow selected above.
template <uint32_t block_tiles, uint32_t Mt>
TT_KERNEL void compute(uint32_t step_start, uint32_t step_count) {
    qkv_conv_compute<block_tiles, Mt>(step_start, step_count);
}
