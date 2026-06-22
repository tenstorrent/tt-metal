// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused SwiGLU compute kernel — PACKER_L1_ACC variant.
//
// Pattern (modelled on bmm_large_block_zm_fused_bias_activation.cpp without
// FUSE_BIAS):
//   * Each matmul phase has num_blocks K-blocks. dst is acquired/released
//     once per (sb_m, sb_n) subblock pair within each K-block.
//   * For K-blocks [0..num_blocks-2]:
//        - The K-loop accumulates the block's per-output-tile dot products
//          into a fresh dst (acquire_dst clears via dst_section_done).
//        - The packer is configured with L1_ACC=0 on block 0 (overwrite) and
//          L1_ACC=1 on block 1+ (add to existing L1). Each subblock packs
//          into mm_partials_cb, then the kernel pops what it just pushed so
//          the CB ring stays bounded. Because the CB is sized exactly to
//          one full block, the next K-block's packer writes land back in
//          the SAME L1 slots, accumulating physically into the same region.
//   * For the LAST K-block:
//        - L1_ACC stays enabled. The block's dst is added into the partials.
//        - Subblocks push into mm_partials WITHOUT popping, so after the
//          K-loop the partials CB has out_block_num_tiles tiles available
//          holding the final accumulated sum.
//   * After the K-loop, a second pass copies each subblock from partials_cb
//     into dst, optionally applies SILU on the pack, and packs into the
//     final CB (cb_gate_intermed / cb_up_intermed / cb_out).
//
// Cross-K-block accumulation is handled by the packer (PACKER_L1_ACC), not by
// reloading dst between K-blocks — a dst-reload approach produced Inf outputs.
//
// File map:
//   matmul_phase           (~L56)  — single-matmul phase that accumulates
//                                     via PACKER_L1_ACC across K-blocks. Used
//                                     for the down matmul (and gate alone if
//                                     up isn't fused).
//   matmul_phase_fused_gu  (~L194) — gate+up fused matmul phase: one K-block
//                                     read of x feeds two output CBs (gate,
//                                     up) via two matmul subblocks per pass.
//   multiply_phase         (~L346) — elementwise silu(gate) * up, producing
//                                     the activated CB.
//   kernel_main            (~L384) — chunk loop: read counts/idx from scratch
//                                     CBs, decide effective_chunks via the
//                                     UNPACK→{MATH,PACK} mailbox handshake,
//                                     then per-chunk dispatch the fused-GU
//                                     phase, multiply phase, and down phase.
//
// Thread-private symbols `mailbox_write`/`mailbox_read` live in the
// `ckernel` namespace (one mailbox slot per (sender, receiver) thread
// pair). See `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/
// reader_bmm_tile_layout_in0_receiver.cpp` for the canonical
// production usage; we use it here to broadcast the device-side count
// value computed inside an UNPACK-thread block to MATH and PACK so all
// three threads agree on `effective_chunks` without re-reading L1.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"

namespace {

template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    bool apply_silu_on_final>
FORCE_INLINE void matmul_phase(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t partials_cb_id, uint32_t final_cb_id) {
    // Reconfig packer for partials format (previous phase's final_cb format
    // would otherwise leak). pack_reconfig_data_format (the reconfig variant)
    // does NOT reset L1_ACC — we do that explicitly below.
    PACK((pack_reconfig_data_format(partials_cb_id)));
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // block 0 must overwrite, not accumulate
#endif

    // Cross-K-block accumulation via PACKER_L1_ACC, using the proven production
    // matmul packing discipline (see bmm_large_block_zm_fused_bias_activation.cpp):
    // each subblock is reserved → packed in-order (pack_tile_block) → pushed, and
    // the full block is drained (wait_front + pop_front) BETWEEN K-blocks.
    //
    // The drain is load-bearing for correctness, not just ring hygiene. wait_front
    // blocks until the packer has committed the tiles it just wrote, so block N+1's
    // L1_ACC read-modify-write cannot race ahead of block N's write landing in the
    // same L1 slot; the matching pop_front wraps the CB write pointer back so the
    // next block accumulates physically into block 0's slots. A previous
    // whole-block-reserve / out-of-order-pack variant skipped this drain — it
    // nondeterministically lost a subblock's first-tile (column 0) contribution
    // when out_subblock_w > 1 (PCC ~0.83-0.98 run-to-run on the small-token
    // routed-expert test). It "worked" only at out_subblock_w == 1, where the
    // extra tile_regs barrier per pack incidentally serialised the writes — at the
    // cost of down-matmul efficiency. The fused gate/up phase escapes the race
    // because its two interleaved matmuls already separate consecutive same-slot
    // packs. Keeping the perf-optimal wide subblock here is now safe.
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        cb_wait_front(in1_cb_id, in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        for (uint32_t sb_m = 0; sb_m < in0_num_subblocks; ++sb_m) {
            int in1_index_subblock_offset = 0;
            for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                tile_regs_acquire();

                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_subblock_offset;
                uint32_t in1_index = in1_index_subblock_offset;
                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                    matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_index,
                        in1_index,
                        dst_index,
                        /*transpose=*/0,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                    in0_index += 1;
                    in1_index += in1_per_core_w;
                }

                tile_regs_commit();
                cb_reserve_back(partials_cb_id, out_subblock_num_tiles);
                tile_regs_wait();
                pack_tile_block(0, partials_cb_id, out_subblock_num_tiles);
                tile_regs_release();
                cb_push_back(partials_cb_id, out_subblock_num_tiles);

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

#ifdef PACKER_L1_ACC
        // After block 0 finishes, flip L1_ACC on so blocks 1..N-1 accumulate.
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif

        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);

        // Drain all but the last K-block (see header comment): forces the
        // packer's writes visible before the next block's L1_ACC RMW and wraps
        // the write pointer back to block 0's slots. The last block's output is
        // left pushed for the second-pass copy below.
        if (block + 1 < num_blocks) {
            for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                cb_wait_front(partials_cb_id, out_subblock_num_tiles);
                cb_pop_front(partials_cb_id, out_subblock_num_tiles);
            }
        }
    }

    // After the K-loop: partials_cb_id has out_block_num_tiles tiles holding
    // the final accumulated sum. Move them through dst into final_cb_id,
    // applying silu on the way if requested.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // future packs (to final_cb) must overwrite
#endif
    // Packer was configured for partials_cb format during matmul. The final
    // pack lands in final_cb (different format) — reconfigure both packer
    // data format and SrcA before the copy/pack loop.
    PACK((pack_reconfig_data_format(final_cb_id)));
    // matmul puts in1 → SrcA, in0 → SrcB. Reconfigure SrcA from in1 to
    // partials so copy_tile reads partials.
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, partials_cb_id);

    for (uint32_t sb = 0; sb < (out_block_num_tiles / out_subblock_num_tiles); ++sb) {
        tile_regs_acquire();
        cb_wait_front(partials_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            copy_tile(partials_cb_id, i, i);
        }
        cb_pop_front(partials_cb_id, out_subblock_num_tiles);

        tile_regs_commit();

        if constexpr (apply_silu_on_final) {
            apply_activation_from_pack<KernelActivation::SILU>(out_subblock_num_tiles);
        } else {
            tile_regs_wait();
        }

        cb_reserve_back(final_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, final_cb_id);
        }
        cb_push_back(final_cb_id, out_subblock_num_tiles);

        tile_regs_release();
    }
}

// Fused gate+up matmul phase. Per K-block, we:
//   1. Wait on x, gate, up (cb_wait_front all three).
//   2. For each (sb_m, sb_n) subblock: do TWO matmul_block sequences using
//      the SAME shared x K-block — first matmul x*gate → partials_gu, then
//      matmul x*up → partials_up. Each pack goes to its respective partials
//      CB; L1_ACC progression (overwrite for block 0, accumulate after) is
//      the SAME for both partials (PACKER_L1_ACC is a global packer state).
//   3. Pop x, gate, up once per K-block (the same x K-block feeds both
//      matmuls, so x is read from DRAM once per K-block instead of twice).
// After the K-loop, copy partials_gu → gate_intermed (with silu fused on
// the pack), then partials_up → up_intermed (no activation). Both partials
// CBs are Float16_b so switching between them needs no format reconfig.
template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles>
FORCE_INLINE void matmul_phase_fused_gu(
    uint32_t x_cb_id,
    uint32_t gate_cb_id,
    uint32_t up_cb_id,
    uint32_t partials_gu_cb_id,
    uint32_t partials_up_cb_id,
    uint32_t gate_intermed_cb_id,
    uint32_t up_intermed_cb_id) {
    PACK((pack_reconfig_data_format(partials_gu_cb_id)));
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

    // Reserve both partials CBs once for the full per-core block. pack_tile
    // with output_tile_index writes to absolute slots; WrPtr doesn't advance
    // until cb_push_back below. Across K-blocks 1..N-1, L1_ACC packs land
    // back in the SAME L1 slots — accumulating physically — which is what
    // we want. No per-K-block pop+repush needed.
    cb_reserve_back(partials_gu_cb_id, out_block_num_tiles);
    cb_reserve_back(partials_up_cb_id, out_block_num_tiles);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(x_cb_id, in0_block_num_tiles);
        cb_wait_front(gate_cb_id, in1_block_num_tiles);
        cb_wait_front(up_cb_id, in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        uint32_t partials_slot_idx = 0;
        for (uint32_t sb_m = 0; sb_m < in0_num_subblocks; ++sb_m) {
            int in1_index_subblock_offset = 0;
            for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                // --- Gate matmul: x * gate -> partials_gu ---
                tile_regs_acquire();
                {
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                        matmul_block(
                            x_cb_id,
                            gate_cb_id,
                            in0_index,
                            in1_index,
                            /*dst_index=*/0,
                            /*transpose=*/0,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        in0_index += 1;
                        in1_index += in1_per_core_w;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                    pack_tile<true>(i, partials_gu_cb_id, partials_slot_idx + i);
                }
                tile_regs_release();

                // --- Up matmul: x * up -> partials_up (same x, different in1) ---
                tile_regs_acquire();
                {
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                        matmul_block(
                            x_cb_id,
                            up_cb_id,
                            in0_index,
                            in1_index,
                            /*dst_index=*/0,
                            /*transpose=*/0,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        in0_index += 1;
                        in1_index += in1_per_core_w;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                    pack_tile<true>(i, partials_up_cb_id, partials_slot_idx + i);
                }
                tile_regs_release();
                partials_slot_idx += out_subblock_num_tiles;

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

#ifdef PACKER_L1_ACC
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif

        cb_pop_front(x_cb_id, in0_block_num_tiles);
        cb_pop_front(gate_cb_id, in1_block_num_tiles);
        cb_pop_front(up_cb_id, in1_block_num_tiles);
    }
    // Make the accumulated partials visible to the second-pass copy loops.
    cb_push_back(partials_gu_cb_id, out_block_num_tiles);
    cb_push_back(partials_up_cb_id, out_block_num_tiles);

    // After K-loop: partials_gu holds gate-matmul accumulator,
    // partials_up holds up-matmul accumulator. Copy each to its intermed
    // CB, fusing silu on the gate copy's final pack.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

    // Gate partials → gate_intermed (silu applied via MATH-thread SFPU on dst,
    // NOT packer-fused). Per subblock:
    //   * copy partials_gu → dst (UNPACK reads bf16, MATH stores in dst regs).
    //   * silu_tile on each dst tile — runs on the MATH thread's SFPU,
    //     overlapping with the next subblock's UNPACK rather than gating the
    //     pack pipeline as apply_activation_from_pack would.
    //   * pack dst → gate_intermed without per-tile SFPU.
    PACK((pack_reconfig_data_format(gate_intermed_cb_id)));
    // SrcA was last configured for the up matmul's in1 (up_cb_id). Switch
    // to partials_gu so copy_tile reads the accumulator.
    copy_tile_to_dst_init_short_with_dt(up_cb_id, partials_gu_cb_id);
    for (uint32_t sb = 0; sb < (out_block_num_tiles / out_subblock_num_tiles); ++sb) {
        tile_regs_acquire();
        cb_wait_front(partials_gu_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            copy_tile(partials_gu_cb_id, i, i);
        }
        cb_pop_front(partials_gu_cb_id, out_subblock_num_tiles);
        // MATH-thread SFPU pass: apply silu to each dst tile before pack.
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            silu_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(gate_intermed_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, gate_intermed_cb_id);
        }
        cb_push_back(gate_intermed_cb_id, out_subblock_num_tiles);
        tile_regs_release();
    }

    // Up partials are NOT copied to a separate cb_up_intermed: the multiply
    // phase reads cb_partials_up directly (bf16) and pairs each tile with
    // cb_gate_intermed (bf8 after silu+pack). Skipping the copy saves 48KB of
    // L1 and lets cb_in0_down_full stay double-buffered.
    (void)up_intermed_cb_id;
}

template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void multiply_phase(uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id) {
    cb_wait_front(gate_cb_id, out_block_num_tiles);
    cb_wait_front(up_cb_id, out_block_num_tiles);

    // Reconfigure packer for activated format and unpacker for both
    // gate_cb (SrcA) and up_cb (SrcB). After phase 2's second pass the
    // SrcA was configured for partials_gu but SrcB still points at the
    // old cb_in0_x (bf8) from matmul — mul_tiles_init's full_init only
    // reprograms the unpack MOP, not the data formats. Without the
    // explicit reconfig SrcB reads bf16 up_intermed bytes as bf8 and the
    // multiply collapses to denormal magnitudes.
    PACK((pack_reconfig_data_format(activated_cb_id)));
    reconfig_data_format(gate_cb_id, up_cb_id);
    mul_tiles_init(gate_cb_id, up_cb_id);

    constexpr uint32_t num_subblocks = out_block_num_tiles / out_subblock_num_tiles;
    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(activated_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        cb_push_back(activated_cb_id, out_subblock_num_tiles);
        tile_regs_release();
        base += out_subblock_num_tiles;
    }
    cb_pop_front(gate_cb_id, out_block_num_tiles);
    cb_pop_front(up_cb_id, out_block_num_tiles);
}

}  // namespace

void kernel_main() {
    // Phase 1 (gate)
    constexpr uint32_t g_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t g_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);
    // Phase 2 (up)
    constexpr uint32_t u_in0_block_w = get_compile_time_arg_val(8);
    constexpr uint32_t u_in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t u_in0_block_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t u_in0_subblock_num_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t u_in1_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t u_in1_block_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t u_in1_per_core_w = get_compile_time_arg_val(14);
    constexpr uint32_t u_num_blocks = get_compile_time_arg_val(15);
    // Phase 4 (down)
    constexpr uint32_t d_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t d_in0_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in0_subblock_num_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t d_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_in1_per_core_w = get_compile_time_arg_val(22);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);
    // Subblock dims
    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(29);
    // Multi-chunk: when M_tiles_full > chunk_M_tiles we run all 4 phases
    // num_chunks times. Reader/writer feed/drain chunk-N+1 while compute is
    // still on chunk N via the existing CBs.
    constexpr uint32_t num_chunks = get_compile_time_arg_val(30);
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(31);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(32);

    // CBs
    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_up_intermed = get_named_compile_time_arg_val("cb_up_intermed");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_partials_gu = get_named_compile_time_arg_val("cb_mm_partials_gu");
    constexpr uint32_t cb_partials_up = get_named_compile_time_arg_val("cb_mm_partials_up");
    constexpr uint32_t cb_partials_d = get_named_compile_time_arg_val("cb_mm_partials_d");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_counts_scratch = get_named_compile_time_arg_val("cb_counts_scratch");
    constexpr uint32_t cb_idx_scratch = get_named_compile_time_arg_val("cb_idx_scratch");

    // Wait for the reader (BRISC) to push the per-expert counts/idx into
    // shared L1. UNPACK reads the L1 via LocalCBInterface and broadcasts
    // count_value to MATH and PACK via the inter-thread mailbox (MATH cannot
    // access get_local_cb_interface symbols at link time). Production matmul
    // uses the same UNPACK→mailbox→MATH/PACK pattern (see
    // circular_buffer.h::read_tile_value).
    cb_wait_front(cb_counts_scratch, 1);
    cb_wait_front(cb_idx_scratch, 1);
    uint32_t count_value = 0;
    UNPACK(({
        const uint32_t counts_l1_addr = get_local_cb_interface(cb_counts_scratch).fifo_rd_ptr << 4;
        const uint32_t idx_l1_addr = get_local_cb_interface(cb_idx_scratch).fifo_rd_ptr << 4;
        const volatile tt_l1_ptr uint32_t* counts_ptr =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counts_l1_addr);
        const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1_addr);
        const uint32_t global_expert_id = idx_ptr[local_expert_id];
        count_value = counts_ptr[global_expert_id];
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count_value);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count_value);
    }));
    MATH(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    // count is in TOKEN rows; convert to tile rows (ceil) and then to chunks.
    const uint32_t count_tiles = (count_value + 31) / 32;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;

    // SiLU is now applied as a MATH-thread SFPU pass on dst (silu_tile)
    // between copy_tile and pack_tile — not packer-fused via
    // apply_activation_from_pack. Empirically the packer-fused variant
    // serialises the pack pipeline against the SFPU, slowing down the
    // gate-intermed write. silu_tile_init() configures the MATH-side SFPU
    // for silu; the pack then runs plain (no per-tile SFPU on the pack
    // thread). Same total compute, better pipelining.
    silu_tile_init();

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0_x, cb_in1_gate, cb_partials_gu);

    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        // matmul_block_init only re-programs addressing, not SrcA/SrcB formats. On
        // chunk >= 1 the unpacker is left on multiply_phase's operands, so reset it
        // to the gate/up inputs here (in1 -> SrcA, in0 -> SrcB).
        reconfig_data_format(cb_in1_gate, cb_in0_x);
        matmul_block_init(
            cb_in0_x,
            cb_in1_gate,
            /*transpose=*/0,
            gu_out_subblock_w,
            gu_out_subblock_h,
            g_in0_block_w);

        // Phases 1 & 2 fused: gate matmul + up matmul share the same per-K-block
        // x push from the reader, so x DRAM mcast bytes are halved (one x read
        // per K-block feeds both matmuls). Both matmuls accumulate into their
        // own partials CB; after the K-loop, partials_gu -> gate_intermed (with
        // silu) and partials_up -> up_intermed are produced by the same fused
        // function.
        matmul_phase_fused_gu<
            g_in0_block_w,
            g_in0_num_subblocks,
            g_in0_block_num_tiles,
            g_in0_subblock_num_tiles,
            g_in1_num_subblocks,
            g_in1_block_num_tiles,
            g_in1_per_core_w,
            g_num_blocks,
            gu_out_subblock_h,
            gu_out_subblock_w,
            gu_out_subblock_num_tiles,
            gu_out_block_num_tiles>(
            cb_in0_x, cb_in1_gate, cb_in1_up, cb_partials_gu, cb_partials_up, cb_gate_intermed, cb_up_intermed);

        // Phase 3: elementwise multiply (cb_gate_intermed is silu(partials_gu)
        // in bf8; cb_partials_up is the up matmul accumulator in bf16). The
        // multiply does the format conversion via reconfig_data_format inside
        // multiply_phase — both unpacker srcs get reset to the input CB
        // formats.
        multiply_phase<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
            cb_gate_intermed, cb_partials_up, cb_activated);
        (void)cb_up_intermed;

        // Phase 4: down matmul, output to cb_out.
        // multiply_phase left the unpacker on (cb_gate_intermed, cb_partials_up);
        // matmul_block_init does not re-program data formats, so reset the down
        // operands here (in1 -> SrcA, in0 -> SrcB) before the matmul.
        reconfig_data_format(cb_in1_down, cb_in0_down_full);
        matmul_block_init(
            cb_in0_down_full,
            cb_in1_down,
            /*transpose=*/0,
            d_out_subblock_w,
            d_out_subblock_h,
            d_in0_block_w);
        matmul_phase<
            d_in0_block_w,
            d_in0_num_subblocks,
            d_in0_block_num_tiles,
            d_in0_subblock_num_tiles,
            d_in1_num_subblocks,
            d_in1_block_num_tiles,
            d_in1_per_core_w,
            d_num_blocks,
            d_out_subblock_h,
            d_out_subblock_w,
            d_out_subblock_num_tiles,
            d_out_block_num_tiles,
            /*apply_silu_on_final=*/false>(cb_in0_down_full, cb_in1_down, cb_partials_d, cb_out);
    }  // end chunk loop
}
