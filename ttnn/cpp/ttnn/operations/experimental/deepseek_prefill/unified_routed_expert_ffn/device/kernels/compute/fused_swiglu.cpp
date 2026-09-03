// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused SwiGLU compute kernel — PACKER_L1_ACC variant.
//
// Pattern (modelled on bmm_large_block_zm_fused_bias_activation.cpp without
// FUSE_BIAS), with one deviation shared by every matmul phase here: the partials
// CB is reserved ONCE for the whole per-core output block instead of per subblock.
//   * Each matmul phase has num_blocks K-blocks. dst is acquired/released
//     once per (sb_m, sb_n) subblock pair within each K-block, and the K-loop
//     accumulates that subblock's per-output-tile dot products into it.
//   * The packer is configured with L1_ACC=0 on block 0 (overwrite) and L1_ACC=1
//     on block 1+ (add to existing L1). Each subblock packs to an ABSOLUTE
//     partials slot (pack_tile<true>); WrPtr does not advance until the single
//     push_back after the K-loop, so every K-block's packs land in the SAME L1
//     addresses and accumulate physically in place.
//   * Reserving once means no cb_push_back/wait_front round trip separates the
//     K-blocks, so each block ends with an explicit packer drain (see
//     drain_packer_before_reaccumulate). Without it, block N+1's L1_ACC
//     read-modify-write can overtake block N's pack of the same slot and
//     silently drop that contribution.
//   * After the K-loop the partials CB is pushed once, holding the final
//     accumulated sum, and a second pass copies each subblock back into dst,
//     optionally applies SILU, and packs into the final CB (cb_gate_intermed /
//     cb_out).
//
// Cross-K-block accumulation is handled by the packer (PACKER_L1_ACC), not by
// reloading dst between K-blocks — a dst-reload approach produced Inf outputs.
//
// File map:
//   matmul_phase           (~L100)  — single-matmul phase that accumulates
//                                     via PACKER_L1_ACC across K-blocks. Used
//                                     for the down matmul (and gate alone if
//                                     up isn't fused).
//   matmul_phase_fused_gu  (~L320) — gate+up fused matmul phase: one K-block
//                                     read of x feeds two output CBs (gate,
//                                     up) via two matmul subblocks per pass.
//   multiply_phase         (~L660) — elementwise silu(gate) * up, producing
//                                     the activated CB.
//   kernel_main            (~L710) — chunk loop: read counts/idx from scratch
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
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp"

#ifdef FUSE_BIAS
// Row-broadcast bias add (gpt-oss). Bias is a (1, N) tensor tiled to one
// tile-row; add_tiles_bcast_rows adds its row 0 to every row of the output
// tile. Same primitive the canonical matmul FUSE_BIAS path uses.
#include "api/compute/bcast.h"
#endif

// SwiGLU-OAI, SiTU-GLU and clamped SiLU-GLU all evaluate their activation as one binary SFPU
// op over the raw gate/up accumulators, so they share the phase-3 path below and differ only
// in the op called. The program factory sets exactly one variant define; each variant
// caches as a distinct program, so a stray second define would mean wrong numerics with no
// host-side signal.
#if (defined(SWIGLU_OAI) + defined(SITU_GLU) + defined(CLAMPED_SILU_GLU)) > 1
#error "SWIGLU_OAI, SITU_GLU and CLAMPED_SILU_GLU are mutually exclusive activation variants"
#endif
#if defined(SWIGLU_OAI) || defined(SITU_GLU) || defined(CLAMPED_SILU_GLU)
#define FUSED_BINARY_ACT 1
#endif

#ifdef SWIGLU_OAI
// Computes (clamp(up,±L)+1) * clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L)).
// Default SwiGLUConfigGPTOSS (alpha=1.702, clamp_limit=7.0) matches M3's config.json.
// swiglu_sfpu.h lives under the gpt-oss moe_gpt op; this repo-root-relative include
// resolves on the kernel include path (same convention as bmm_fused_activation.hpp
// above). It could later move to a shared kernel-include dir, but the path is valid
// as-is (verified on Blackhole via test_swigluoai_routed_expert.py).
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#endif

#ifdef SITU_GLU
// Computes (beta_gate*tanh(gate/beta_gate)*sigmoid(gate)) * (beta_up*tanh(up/beta_up)).
// Bakes SituGluConfigKimi (beta_gate=4.0, beta_up=25.0), Kimi K3's config.
#include "api/compute/situ_glu.h"
#endif

#ifdef CLAMPED_SILU_GLU
// Computes silu(min(gate,L)) * clamp(up,±L). Bakes ClampedSiluGluConfigDsV4 (limit=10.0).
#include "api/compute/clamped_silu_glu.h"
#endif

namespace {

// Packer-completion barrier for the K-block boundary of a PACKER_L1_ACC phase.
//
// Every K-block re-packs the same absolute partials slots, and the accumulate is
// a packer-side L1 read-modify-write. Pack issue must therefore stop until the
// packer has drained, or the next block's read can sample a slot the previous
// block's write has not reached yet — a lost contribution, nondeterministic and
// worst on small per-core blocks where the same slot recurs a few packs later.
//
// Packer-idle is the same condition cb_push_back waits on before publishing
// tiles_received (llk_push_tiles -> STALLWAIT(STALL_THCON, PACK)), i.e. exactly
// the guarantee a per-K-block CB drain would buy, minus the drain.
FORCE_INLINE void drain_packer_before_reaccumulate() {
#ifdef PACKER_L1_ACC
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK));
#endif
}

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
    bool apply_silu_on_final,
    uint32_t d_per_core_N = 0,
    // Real (unpadded) K-tiles of the reduction dim. Defaults to the full padded
    // extent (reduce everything); the down phase passes the true count so no
    // padded K position is ever reduced.
    uint32_t real_k_tiles = num_blocks * in0_block_w>
FORCE_INLINE void matmul_phase(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t partials_cb_id,
    uint32_t final_cb_id,
    uint32_t m_subblocks,
    uint32_t n_subblocks,
    uint32_t down_bias_cb_id = 0) {
    // sb_m indexes tile-rows directly, so m_subblocks (a tile-row count) bounds it
    // only at unit subblock height.
    static_assert(out_subblock_h == 1, "m_subblocks bounds sb_m only when out_subblock_h == 1");
    static_assert(
        in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles == out_block_num_tiles,
        "subblock grid must tile the output block exactly");
    static_assert(real_k_tiles <= num_blocks * in0_block_w, "real_k_tiles must not exceed the padded K extent");
    // Adaptive per_core_M: this core's down output is m_subblocks tile-rows this
    // chunk. The partials accumulator shrinks with it; the down WEIGHT block stays
    // full-width (M-independent). Rows past m_subblocks are never written by the
    // reader's (bounded) activated mcast, so they are simply not computed — no MAC,
    // no pack, no partials slot.
    //
    // n_subblocks bounds the OTHER free dim the same way: subblocks past it are
    // wholly phantom output columns (col >= N_down_tiles_full) whose weights the
    // reader never fetched, so MACing them would only multiply stale L1. Unlike the
    // M bound this does NOT shrink EFF_OUT — the partials ring and the final_cb
    // push stay full width, since the writer drains a fixed subblock count.
    const uint32_t EFF_M = m_subblocks;
    const uint32_t EFF_OUT = m_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    // Reconfig packer for partials format (previous phase's final_cb format
    // would otherwise leak). pack_reconfig_data_format (the reconfig variant)
    // does NOT reset L1_ACC — we do that explicitly below.
    pack_reconfig_data_format(partials_cb_id);
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // block 0 must overwrite, not accumulate
#endif

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer partials_cb(partials_cb_id);
    CircularBuffer final_cb(final_cb_id);

    // Reserve the partials block ONCE. pack_tile with an absolute output_tile_index
    // writes fixed slots and WrPtr does not advance until the push_back after the
    // K-loop, so every K-block re-packs the SAME L1 addresses and PACKER_L1_ACC
    // accumulates physically in place — nothing ties the pushed count to the ring
    // size. The K-loop pays for the missing CB round trip with an explicit packer
    // drain per block.
    //
    // Reserve the FULL block, never the runtime EFF_OUT. partials_cb is single-
    // buffered at exactly one max block, so pushing a SHORT block leaves the ring
    // write pointer off a block boundary and the next full-size block straddles the
    // end of the CB — and the packs above use an absolute output_tile_index off
    // get_write_ptr(), so they run past the end instead of wrapping. EFF_OUT does
    // NOT shrink only on an expert's final chunk: this op runs every local expert in
    // one kernel, so per_core_M (hence EFF_OUT) drops on each expert's tail chunk and
    // rises again on the next expert's first chunk. Only [0, EFF_OUT) is ever packed;
    // the unpacked tail is drained by the pointer-only pop after the copy loop.
    partials_cb.reserve_back(out_block_num_tiles);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // The reader pushes the FULL compile-time-max activated block (filling only
        // its first per_core_M tile-rows), so this wait/pop stays compile-time sized.
        in0_cb.wait_front(in0_block_num_tiles);
        in1_cb.wait_front(in1_block_num_tiles);

        // Reduce only the REAL K-tiles this block covers. Neither operand is
        // zero-filled past real_k_tiles — the down weights hold stale L1 and the
        // activated columns hold the gate/up matmul's garbage output on the hidden
        // padding columns — and a reduction position contaminates EVERY output
        // column, so a padded position must never enter the MAC. kt_dim passed to
        // matmul_block stays in0_block_w: it is the in0 row stride (the block is
        // still physically in0_block_w wide), not the step count.
        const uint32_t k_done = block * in0_block_w;
        const uint32_t k_left = k_done < real_k_tiles ? real_k_tiles - k_done : 0;
        const uint32_t k_steps = k_left < in0_block_w ? k_left : in0_block_w;

        // A block entirely past real_k_tiles contributes nothing: skip its MAC AND
        // its pack (with L1_ACC on, packing an untouched dst would ADD garbage to
        // the accumulator). Its CBs are still drained below — the reader and writer
        // push every padded block unconditionally.
        if (k_steps > 0) {
            int in0_index_subblock_offset = 0;
            uint32_t partials_slot_idx = 0;
            for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
                int in1_index_subblock_offset = 0;
                for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                    // Phantom-column subblocks: no MAC, no pack. The slot counters still
                    // advance so the surviving subblocks keep their absolute partials slots.
                    if (sb_n < n_subblocks) {
                        tile_regs_acquire();
                        {
                            uint32_t in0_index = in0_index_subblock_offset;
                            uint32_t in1_index = in1_index_subblock_offset;
                            for (uint32_t inner_dim = 0; inner_dim < k_steps; ++inner_dim) {
                                matmul_block(
                                    in0_cb_id,
                                    in1_cb_id,
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
                            pack_tile<true>(i, partials_cb_id, partials_slot_idx + i);
                        }
                        tile_regs_release();
                    }
                    partials_slot_idx += out_subblock_num_tiles;

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }
        }

#ifdef PACKER_L1_ACC
        // After block 0 finishes, flip L1_ACC on so blocks 1..N-1 accumulate.
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif
        // This block's packs must land before the next one accumulates on top of
        // them. Unlike the fused gate/up phase there is no second matmul between
        // consecutive same-slot packs here, and a skipped block (k_steps == 0)
        // removes even the subblock loop that would otherwise sit between them.
        drain_packer_before_reaccumulate();

        in0_cb.pop_front(in0_block_num_tiles);
        in1_cb.pop_front(in1_block_num_tiles);
    }
    // Make the accumulated partials visible to the second-pass copy below.
    partials_cb.push_back(out_block_num_tiles);

    // After the K-loop: partials_cb_id has EFF_OUT tiles holding the final
    // accumulated sum. Move them through dst into final_cb_id, applying silu on
    // the way if requested.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // future packs (to final_cb) must overwrite
#endif
    // Packer was configured for partials_cb format during matmul. The final
    // pack lands in final_cb (different format) — reconfigure both packer
    // data format and SrcA before the copy/pack loop.
    pack_reconfig_data_format(final_cb_id);
#ifdef FUSE_BIAS
    // Down bias (gpt-oss): add the (1, emb) bias (broadcast across rows) to the
    // down-matmul output as it is drained from partials -> final. down_bias_cb
    // holds this core's per_core_N_d columns (read once by the reader). d out
    // subblock height is 1, so the flat tile index gives col = flat % d_per_core_N.
    (void)in1_cb_id;
    CircularBuffer down_bias_cb(down_bias_cb_id);
    down_bias_cb.wait_front(d_per_core_N);
    // Down matmul left SrcA on down weights (bf4), SrcB on activated (bf8).
    // Reconfig SrcA=partials (Float16_b), SrcB=down_bias before the bcast init.
    reconfig_data_format(partials_cb_id, down_bias_cb_id);
    add_bcast_rows_init(partials_cb_id, down_bias_cb_id);
#else
    (void)down_bias_cb_id;
    // matmul puts in1 → SrcA, in0 → SrcB. Reconfigure SrcA from in1 to
    // partials so copy_tile reads partials.
    reconfig_data_format_srca(in1_cb_id, partials_cb_id);
    copy_init(partials_cb_id);
#endif

    const uint32_t eff_subblocks = EFF_OUT / out_subblock_num_tiles;
    for (uint32_t sb = 0; sb < eff_subblocks; ++sb) {
        tile_regs_acquire();
        partials_cb.wait_front(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
#ifdef FUSE_BIAS
            const uint32_t flat = sb * out_subblock_num_tiles + i;
            add_tiles_bcast_rows(partials_cb_id, down_bias_cb_id, i, flat % d_per_core_N, i);
#else
            copy_tile(partials_cb_id, i, i);
#endif
        }
        partials_cb.pop_front(out_subblock_num_tiles);

        tile_regs_commit();

        if constexpr (apply_silu_on_final) {
            apply_activation_from_pack<KernelActivation::SILU>(out_subblock_num_tiles);
        } else {
            tile_regs_wait();
        }

        final_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, final_cb_id);
        }
        final_cb.push_back(out_subblock_num_tiles);

        tile_regs_release();
    }

    // Pointer-only drain of the partials tail: slots [EFF_OUT, out_block_num_tiles)
    // were never packed this chunk. Popping them keeps the single-buffered partials
    // ring block-aligned across chunks (see the reserve above).
    if (EFF_OUT < out_block_num_tiles) {
        const uint32_t partials_pad = out_block_num_tiles - EFF_OUT;
        partials_cb.wait_front(partials_pad);
        partials_cb.pop_front(partials_pad);
    }

    // The writer drains final_cb at the compile-time-MAX subblock count (it cannot
    // shrink its drain without losing ring balance) and drops the rows past the
    // runtime per_core_M via its own row guards. Hand it the leftover slots WITHOUT
    // packing: their L1 holds stale bytes that are never written out, so the
    // reserve/push pair alone is enough to keep producer and consumer aligned.
    constexpr uint32_t FULL_SUBBLOCKS = out_block_num_tiles / out_subblock_num_tiles;
    for (uint32_t sb = eff_subblocks; sb < FULL_SUBBLOCKS; ++sb) {
        final_cb.reserve_back(out_subblock_num_tiles);
        final_cb.push_back(out_subblock_num_tiles);
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
    uint32_t out_block_num_tiles,
    // x_cb_id / x_rm_cb_id are compile-time so the tilize helper (which takes the
    // input/output CB as template args, like conv_bmm_tilize.cpp) can consume them.
    uint32_t x_cb_id,
    uint32_t x_rm_cb_id,
    bool tilize_x = false>
FORCE_INLINE void matmul_phase_fused_gu(
    uint32_t gate_cb_id,
    uint32_t up_cb_id,
    uint32_t partials_gu_cb_id,
    uint32_t partials_up_cb_id,
    uint32_t gate_intermed_cb_id,
    uint32_t up_intermed_cb_id,
    uint32_t m_subblocks,
    uint32_t n_subblocks) {
    // No real_k_tiles bound here, unlike the down phase: the host asserts
    // K_gate_tiles % in0_block_w == 0, so the gate/up K-loop covers the reduction
    // dim exactly and there is no padded K position to skip. This phase's padding
    // is all in N (the hidden dim), where a matmul is column-independent — the
    // unwritten weight columns past N_gate_tiles_full corrupt only their OWN output
    // columns, which map 1:1 onto the down K positions the down phase excludes.
    //
    // Adaptive per_core_M: this core's gate/up WORK is bounded by the runtime
    // per_core_M (m_subblocks tile-rows) — MACs, tilize strips, copies and packs
    // all scale with it. The gate/up WEIGHT blocks stay full-width (M-independent).
    //
    // The CB BLOCK SIZE, however, is the compile-time MAX and never varies: every
    // reserve/push/wait/pop below moves a full max block, and the runtime remainder
    // is padded with O(1) pointer-only bumps. See the ring-alignment note in
    // adaptive_chunk.hpp — a block size that changes between experts overshoots
    // fifo_limit, which never wraps, and the CB pointer runs away into L1.
    const uint32_t EFF_M = m_subblocks;
    const uint32_t x_block_tiles = m_subblocks * in0_block_w;  // runtime x work (per_core_M rows)
    const uint32_t EFF_OUT = m_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    // Constant CB block sizes (per_core_M_max-derived, from the host CB sizing).
    constexpr uint32_t X_BLOCK_TILES_MAX = in0_block_num_tiles;
    constexpr uint32_t EFF_OUT_MAX = out_block_num_tiles;
    pack_reconfig_data_format(partials_gu_cb_id);
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

    CircularBuffer x_cb(x_cb_id);
    CircularBuffer gate_cb(gate_cb_id);
    CircularBuffer up_cb(up_cb_id);
    CircularBuffer partials_gu_cb(partials_gu_cb_id);
    CircularBuffer partials_up_cb(partials_up_cb_id);
    CircularBuffer gate_intermed_cb(gate_intermed_cb_id);

    // Reserve both partials CBs once for the FULL max block. pack_tile with
    // output_tile_index writes to absolute slots; WrPtr doesn't advance until
    // cb_push_back below. Across K-blocks 1..N-1, L1_ACC packs land back in the
    // SAME L1 slots — accumulating physically — which is what we want. No
    // per-K-block pop+repush needed. Only slots [0, EFF_OUT) are ever written;
    // the rest of the block is stale and is dropped downstream (the down matmul
    // MAC-skips rows >= per_core_M and the writer never emits them).
    partials_gu_cb.reserve_back(EFF_OUT_MAX);
    partials_up_cb.reserve_back(EFF_OUT_MAX);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if constexpr (tilize_x) {
            //  Row-major x: tilize this K-block's cb_x_rm strips (bf16) -> x_cb
            //  (cb_in0_x, bf8_b) before the matmul consumes it. L1_ACC is turned
            //  off so the tilize packs OVERWRITE x_cb rather than accumulate; the
            //  shared tilize helper (same one conv_bmm_tilize.cpp uses) then
            //  reconfigures unpack SrcA + pack format, drives the per-strip
            //  wait/reserve/tilize/push/pop over the runtime x_block_tiles /
            //  in0_block_w tile-rows (adaptive per_core_M), and restores init on
            //  exit. The helper left SrcA pointing at the bf16 row-major input, so
            //  restore it to the gate/up weight format before resuming the matmul
            //  (SrcB still holds x_cb_id — the BH tilize path never touches it);
            //  then restore the partials packer + L1_ACC state for this block.
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
            const uint32_t n_strips = x_block_tiles / in0_block_w;
            compute_kernel_lib::tilize<
                in0_block_w,
                x_rm_cb_id,
                x_cb_id,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(n_strips);
            // The helper consumed/produced only the runtime rows, but the reader
            // pushed a full max cb_x_rm block and the matmul below pops a full max
            // cb_in0_x block. Settle both with pointer-only bumps (no tilize work
            // on the stale rows) so every CB moves a constant-size block.
            {
                const uint32_t x_pad = X_BLOCK_TILES_MAX - x_block_tiles;
                if (x_pad > 0) {
                    CircularBuffer x_rm_cb(x_rm_cb_id);
                    x_rm_cb.wait_front(x_pad);
                    x_rm_cb.pop_front(x_pad);
                    x_cb.reserve_back(x_pad);
                    x_cb.push_back(x_pad);
                }
            }
            reconfig_data_format_srca(gate_cb_id);
            matmul_block_init(x_cb_id, gate_cb_id, 0, out_subblock_w, out_subblock_h, in0_block_w);
            pack_reconfig_data_format(x_cb_id, partials_gu_cb_id);
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
#endif
        }
        x_cb.wait_front(X_BLOCK_TILES_MAX);
        gate_cb.wait_front(in1_block_num_tiles);
        up_cb.wait_front(in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        uint32_t partials_slot_idx = 0;
        for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
            int in1_index_subblock_offset = 0;
            for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                // Phantom-column subblocks (col >= N_gate_tiles_full): the reader/writer
                // never fetched those weight tiles, so both matmuls would just multiply
                // stale L1. Skip the MAC and the pack; the slot counters still advance so
                // the surviving subblocks keep their absolute partials slots, and the
                // pushed EFF_OUT stays full width for the activated mcast.
                if (sb_n < n_subblocks) {
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
                }
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
        // Same reserve-once accumulator, same barrier. The interleaved up matmul
        // puts a handful of packs between consecutive same-slot packs, but that is
        // a side effect of the loop shape, not ordering: at one subblock and
        // out_subblock_w == 1 it is a single pack.
        drain_packer_before_reaccumulate();

        x_cb.pop_front(X_BLOCK_TILES_MAX);
        gate_cb.pop_front(in1_block_num_tiles);
        up_cb.pop_front(in1_block_num_tiles);
    }
    // Make the accumulated partials visible to the second-pass copy loops.
    partials_gu_cb.push_back(EFF_OUT_MAX);
    partials_up_cb.push_back(EFF_OUT_MAX);

    // After K-loop: partials_gu holds gate-matmul accumulator,
    // partials_up holds up-matmul accumulator. Copy each to its intermed
    // CB, fusing silu on the gate copy's final pack.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

#ifdef FUSED_BINARY_ACT
    // Leave both partials pushed (bf16) so binary_activation_phase() can read raw gate and
    // up together. Activating off the bf8 gate_intermed would lose precision first.
    (void)gate_intermed_cb_id;
    (void)up_intermed_cb_id;
#else
    // Gate partials → gate_intermed (silu applied via MATH-thread SFPU on dst,
    // NOT packer-fused). Per subblock:
    //   * copy partials_gu → dst (UNPACK reads bf16, MATH stores in dst regs).
    //   * silu_tile on each dst tile — runs on the MATH thread's SFPU,
    //     overlapping with the next subblock's UNPACK rather than gating the
    //     pack pipeline as apply_activation_from_pack would.
    //   * pack dst → gate_intermed without per-tile SFPU.
    pack_reconfig_data_format(gate_intermed_cb_id);
    // SrcA was last configured for the up matmul's in1 (up_cb_id). Switch
    // to partials_gu so copy_tile reads the accumulator.
    reconfig_data_format_srca(up_cb_id, partials_gu_cb_id);
    copy_init(partials_gu_cb_id);
    for (uint32_t sb = 0; sb < (EFF_OUT / out_subblock_num_tiles); ++sb) {
        tile_regs_acquire();
        partials_gu_cb.wait_front(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            copy_tile(partials_gu_cb_id, i, i);
        }
        partials_gu_cb.pop_front(out_subblock_num_tiles);
        // MATH-thread SFPU pass: apply silu to each dst tile before pack.
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            silu_tile(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        gate_intermed_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, gate_intermed_cb_id);
        }
        gate_intermed_cb.push_back(out_subblock_num_tiles);
        tile_regs_release();
    }
    // The copy loop above did real work only for the runtime rows. Settle the rest
    // of the constant-size block with pointer-only bumps: drop the stale tail of
    // partials_gu and pad gate_intermed so multiply_phase sees a full max block.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            partials_gu_cb.wait_front(pad);
            partials_gu_cb.pop_front(pad);
            gate_intermed_cb.reserve_back(pad);
            gate_intermed_cb.push_back(pad);
        }
    }

    // Up partials are NOT copied to a separate cb_up_intermed: the multiply
    // phase reads cb_partials_up directly (bf16) and pairs each tile with
    // cb_gate_intermed (bf8 after silu+pack). Skipping the copy saves 48KB of
    // L1 and lets cb_in0_down_full stay double-buffered.
    (void)up_intermed_cb_id;
#endif
}

#ifdef FUSED_BINARY_ACT
// Dst-accumulator mode (fp32 dest accum on/off) from the host ComputeConfig,
// passed via -DFP32_DEST_ACC_EN. Defaults to bf16 dst (0) if not passed.
#ifndef FP32_DEST_ACC_EN
#define FP32_DEST_ACC_EN 0
#endif

// Both ops share the (gate, up, out) dst-index signature and bake their constants into a
// config struct, so only these lines differ. Each variant is named explicitly rather than
// left to an #else, so a new one fails the build instead of silently compiling as whichever
// variant owned the fallback.
#if defined(SWIGLU_OAI)
#define BINARY_ACT_INIT() MATH((ckernel::llk_math_eltwise_binary_sfpu_swiglu_init()))
#define BINARY_ACT_TILE(fp32, g, u, o) MATH((ckernel::llk_math_eltwise_binary_sfpu_swiglu<fp32>(g, u, o)))
#elif defined(SITU_GLU)
// situ_glu_tile takes its fp32-dest mode from DST_ACCUM_MODE and wraps itself in MATH().
#define BINARY_ACT_INIT() situ_glu_tile_init()
#define BINARY_ACT_TILE(fp32, g, u, o) situ_glu_tile(g, u, o)
#elif defined(CLAMPED_SILU_GLU)
// clamped_silu_glu_tile takes its fp32-dest mode from DST_ACCUM_MODE and wraps itself in MATH().
#define BINARY_ACT_INIT() clamped_silu_glu_tile_init()
#define BINARY_ACT_TILE(fp32, g, u, o) clamped_silu_glu_tile(g, u, o)
#else
#error "FUSED_BINARY_ACT is set but no activation variant matched"
#endif

// Replaces gate-silu + multiply_phase: reads the raw bf16 gate/up accumulators from their
// partials CBs and writes the activated result to activated_cb. Runs on MATH (between
// tile_regs_acquire/commit), matching this kernel's silu_tile structure.
//
// The binary op pins BOTH gate and up in dst, so each output tile costs 2 dst slots. With
// 8 dst tiles available we stream in chunks of <=4 output tiles. The activated CB is
// drained count-based, so push granularity need not match out_subblock_num_tiles.
template <uint32_t out_block_num_tiles, uint32_t per_core_N_gu = 0>
FORCE_INLINE void binary_activation_phase(
    uint32_t prev_srcA_cb_id,
    uint32_t gate_partials_cb_id,
    uint32_t up_partials_cb_id,
    uint32_t activated_cb_id,
    uint32_t eff_out_tiles,
    uint32_t gate_bias_cb_id = 0,
    uint32_t up_bias_cb_id = 0) {
    // Adaptive per_core_M: this core produces eff_out_tiles = per_core_M * pcN
    // activated tiles this chunk (0 if it owns no row -> nothing to do).
    const uint32_t EFF_OUT = eff_out_tiles;
    // CB blocks are the constant compile-time max (see matmul_phase_fused_gu);
    // only the activation work below is bounded by the runtime EFF_OUT.
    constexpr uint32_t EFF_OUT_MAX = out_block_num_tiles;
    // Dst budget derived from the host ComputeConfig (via -DFP32_DEST_ACC_EN) so
    // it and the SFPU op's fp32-dest template below stay in sync with the
    // program factory's DST_CAPACITY / fp32_dest_acc_en (no silent drift). The
    // 16-tile dst reg file halves under fp32 dest accum. Each output tile pins
    // gate+up simultaneously -> 2 dst slots -> kActChunk output tiles / acquire.
    constexpr bool kFp32DestAccEn = (FP32_DEST_ACC_EN != 0);
    // SiTU-GLU's op reads DST_ACCUM_MODE instead of the define; both come from the same
    // host fp32_dest_acc_en, so assert they agree rather than trusting it.
    static_assert(kFp32DestAccEn == DST_ACCUM_MODE, "FP32_DEST_ACC_EN disagrees with DST_ACCUM_MODE");
    constexpr uint32_t kDstCapacity = kFp32DestAccEn ? 4u : 8u;
    constexpr uint32_t kActChunk = kDstCapacity / 2;

    CircularBuffer gate_partials_cb(gate_partials_cb_id);
    CircularBuffer up_partials_cb(up_partials_cb_id);
    CircularBuffer activated_cb(activated_cb_id);

    gate_partials_cb.wait_front(EFF_OUT_MAX);
    up_partials_cb.wait_front(EFF_OUT_MAX);

    pack_reconfig_data_format(activated_cb_id);
    // SrcA was last configured for the up matmul's in1 weights (prev_srcA_cb_id,
    // e.g. bf4). Reconfig to the Float16_b partials so copy_tile reads the
    // accumulator with the right format. Both partials CBs are Float16_b, so this
    // single init covers reads from gate AND up partials. (Passing a Float16_b CB
    // as the "old" operand would no-op the reconfig and leave SrcA on bf4.)
#ifdef FUSE_BIAS
    // Add gate/up bias (broadcast across rows) before the activation. The
    // bias CBs were read once by the reader (per_core_N_gu tiles each) and are
    // NOT popped here (reused across chunks). add_bcast_rows sets SrcA=partials,
    // SrcB=bias — same format for gate and up, so one init covers both.
    (void)prev_srcA_cb_id;
    CircularBuffer gate_bias_cb(gate_bias_cb_id);
    CircularBuffer up_bias_cb(up_bias_cb_id);
    gate_bias_cb.wait_front(per_core_N_gu);
    up_bias_cb.wait_front(per_core_N_gu);
    // The gate/up matmul left SrcA on the up weights (bf4) and SrcB on x (bf8).
    // The bcast add reads SrcA=partials (Float16_b), SrcB=bias — reconfig both
    // before the init (add_bcast uses both operands, unlike the copy path).
    reconfig_data_format(gate_partials_cb_id, gate_bias_cb_id);
    add_bcast_rows_init(gate_partials_cb_id, gate_bias_cb_id);
#else
    (void)gate_bias_cb_id;
    (void)up_bias_cb_id;
    reconfig_data_format_srca(prev_srcA_cb_id, gate_partials_cb_id);
    copy_init(gate_partials_cb_id);
#endif

    for (uint32_t base = 0; base < EFF_OUT; base += kActChunk) {
        const uint32_t remaining = EFF_OUT - base;
        const uint32_t c = remaining < kActChunk ? remaining : kActChunk;
        tile_regs_acquire();
        // gate -> dst[0..c), up -> dst[c..2c) (bias-added when FUSE_BIAS)
        for (uint32_t j = 0; j < c; ++j) {
#ifdef FUSE_BIAS
            const uint32_t ncol = (base + j) % per_core_N_gu;
            add_tiles_bcast_rows(gate_partials_cb_id, gate_bias_cb_id, base + j, ncol, j);
            add_tiles_bcast_rows(up_partials_cb_id, up_bias_cb_id, base + j, ncol, c + j);
#else
            copy_tile(gate_partials_cb_id, base + j, j);
            copy_tile(up_partials_cb_id, base + j, c + j);
#endif
        }
        // Fused gate/up activation; result written in place to dst[j]
        // (out == gate slot, mirroring moe_gpt's swiglu(0,1,0)).
        for (uint32_t j = 0; j < c; ++j) {
            BINARY_ACT_TILE(kFp32DestAccEn, j, c + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        activated_cb.reserve_back(c);
        for (uint32_t j = 0; j < c; ++j) {
            pack_tile(j, activated_cb_id);
        }
        activated_cb.push_back(c);
        tile_regs_release();
    }
    // Pad cb_activated up to the constant block (pointer-only); the reader drains
    // the full max block and mcasts only the runtime rows.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            activated_cb.reserve_back(pad);
            activated_cb.push_back(pad);
        }
    }
    gate_partials_cb.pop_front(EFF_OUT_MAX);
    up_partials_cb.pop_front(EFF_OUT_MAX);
}
#endif

// eff_out_tiles = number of valid output tiles this core produces this chunk
// (= per_core_M * per_core_N_gu, runtime-adaptive). Precomputed by the caller so
// multiply_phase needs no per_core_M/N template info.
template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void multiply_phase(
    uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id, uint32_t eff_out_tiles) {
    CircularBuffer gate_cb(gate_cb_id);
    CircularBuffer up_cb(up_cb_id);
    CircularBuffer activated_cb(activated_cb_id);

    const uint32_t EFF_OUT = eff_out_tiles;
    // CB blocks are the constant compile-time max (see matmul_phase_fused_gu);
    // only the multiply work below is bounded by the runtime EFF_OUT.
    constexpr uint32_t EFF_OUT_MAX = out_block_num_tiles;

    gate_cb.wait_front(EFF_OUT_MAX);
    up_cb.wait_front(EFF_OUT_MAX);

    // Reconfigure packer for activated format and unpacker for both
    // gate_cb (SrcA) and up_cb (SrcB). After phase 2's second pass the
    // SrcA was configured for partials_gu but SrcB still points at the
    // old cb_in0_x (bf8) from matmul — mul_init's full_init only
    // reprograms the unpack MOP, not the data formats. Without the
    // explicit reconfig SrcB reads bf16 up_intermed bytes as bf8 and the
    // multiply collapses to denormal magnitudes.
    pack_reconfig_data_format(activated_cb_id);
    reconfig_data_format(gate_cb_id, up_cb_id);
    mul_init(gate_cb_id, up_cb_id);

    const uint32_t num_subblocks = EFF_OUT / out_subblock_num_tiles;
    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        activated_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        activated_cb.push_back(out_subblock_num_tiles);
        tile_regs_release();
        base += out_subblock_num_tiles;
    }
    // Pad cb_activated up to the constant block (pointer-only); the reader drains
    // the full max block and mcasts only the runtime rows.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            activated_cb.reserve_back(pad);
            activated_cb.push_back(pad);
        }
    }
    gate_cb.pop_front(EFF_OUT_MAX);
    up_cb.pop_front(EFF_OUT_MAX);
}

}  // namespace

void kernel_main() {
    // Per-core valid N-subblock counts. per_core_N is the GRID-ceil'd width, so the
    // highest-gx cores own phantom output columns whose weights were never fetched;
    // these bound the MAC to the subblocks holding real columns. Runtime (not
    // compile-time) because they vary per core and one kernel serves the whole grid.
    const uint32_t gu_valid_n_subblocks = get_arg_val<uint32_t>(0);
    const uint32_t d_valid_n_subblocks = get_arg_val<uint32_t>(1);

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
    // Multi-chunk: the number of chunks is chosen at RUNTIME from each expert's
    // device token count (see the picker below). num_chunks_max is the compile-time
    // upper bound (host = ceil(M_tiles_full / min_chunk)) used only to clamp the
    // runtime chunk count defensively. chunk_M_max is the CB-sized maximum
    // chunk (per_core_M_max * GRID_Y); the picker never returns more than this.
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(30);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(31);
    // chunk_M_max is the CB-sized MAXIMUM chunk (per_core_M_max * GRID_Y). The
    // runtime picker (adaptive_chunk::num_chunks) sizes the actual chunk to the
    // device token count and never exceeds this.
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(32);
    // x_is_row_major: tilize cb_x_rm -> cb_in0_x before the gate/up matmul.
    // 0 => x already TILE in cb_in0_x.
    constexpr uint32_t x_is_row_major = get_compile_time_arg_val(33);
    // Real (unpadded) down-K tiles. The down K-loop runs over the GRID-padded
    // extent (K_down_tiles_padded), and the down phase skips every K position at
    // or past this count — whole padded blocks and the partial tail alike. That
    // is what lets the reader and writer leave the padded down weights and the
    // padded hidden (gate/up N-OOB) columns unwritten: nothing ever reduces them.
    constexpr uint32_t d_K_down_tiles = get_compile_time_arg_val(34);

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
    // This expert's region size in tile-rows; caps the device-provided count.
    constexpr uint32_t m_tiles_full = get_named_compile_time_arg_val("m_tiles_full");
    // Row-major bf16 x staging (x_is_row_major only); tilize input CB. Unused
    // when x is TILE.
    constexpr uint32_t cb_x_rm = get_named_compile_time_arg_val("cb_x_rm");
#ifdef FUSE_BIAS
    constexpr uint32_t cb_gate_bias = get_named_compile_time_arg_val("cb_gate_bias");
    constexpr uint32_t cb_up_bias = get_named_compile_time_arg_val("cb_up_bias");
    constexpr uint32_t cb_down_bias = get_named_compile_time_arg_val("cb_down_bias");
#else
    constexpr uint32_t cb_gate_bias = 0;
    constexpr uint32_t cb_up_bias = 0;
    constexpr uint32_t cb_down_bias = 0;
#endif

    // First Compute API call in the kernel, as compute_kernel_hw_startup.h requires: it does MMIO
    // config of MATH/PACK/UNPACK and needs those units idle. Nothing below it depends on the count
    // handshake, and the SFPU init that follows must not run before it (SiTU-GLU's tanh_init loads
    // the vConstFloatPrgm registers the activation reads back on every tile).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0_x, cb_in1_gate, cb_partials_gu);

    CircularBuffer counts_scratch_cb(cb_counts_scratch);
    CircularBuffer idx_scratch_cb(cb_idx_scratch);

    // Wait for the reader (BRISC) to push the counts/idx into shared L1. They
    // are pushed ONCE and stay resident, so UNPACK can re-index them per expert.
    counts_scratch_cb.wait_front(1);
    idx_scratch_cb.wait_front(1);

    // SiLU is applied as a MATH-thread SFPU pass on dst (silu_tile) between
    // copy_tile and pack_tile — not packer-fused via apply_activation_from_pack.
    // Empirically the packer-fused variant serialises the pack pipeline against
    // the SFPU, slowing down the gate-intermed write. silu_tile_init() configures
    // the MATH-side SFPU for silu; the pack then runs plain (no per-tile SFPU on
    // the pack thread). Same total compute, better pipelining.
    // Init once — shared across all experts.
#ifdef FUSED_BINARY_ACT
    // The binary activations init their own SFPU tables (recip for SwiGLU-OAI and clamped
    // SiLU-GLU, tanh for SiTU-GLU) instead of silu's. The recip table sets
    // vConstFloatPrgm0 = 2.0f, which nothing between here and the tile calls reprograms.
    BINARY_ACT_INIT();
#else
    silu_tile_init();
#endif


    // ======================= per-local-expert loop =======================
    // Run the full gate/up/down FFN for every local expert in this program.
    for (uint32_t local_expert_id = 0; local_expert_id < experts_per_chip; ++local_expert_id) {
        // This expert's token count via the UNPACK→{MATH,PACK} mailbox (MATH/PACK
        // cannot read the counts/idx L1 via the CB interface).
        // count -> effective_chunks bounds this expert's chunk loop; count=0 => the
        // loop body is skipped entirely.
        uint32_t count_value = 0;
        UNPACK(({
            const uint32_t counts_l1_addr = get_local_cb_interface(cb_counts_scratch).fifo_rd_ptr << 4;
            const uint32_t idx_l1_addr = get_local_cb_interface(cb_idx_scratch).fifo_rd_ptr << 4;
            const volatile tt_l1_ptr uint32_t* counts_ptr =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(counts_l1_addr);
            const volatile tt_l1_ptr uint32_t* idx_ptr =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1_addr);
            const uint32_t global_expert_id = idx_ptr[local_expert_id];
            count_value = counts_ptr[global_expert_id];
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count_value);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count_value);
        }));
        MATH(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        // count is in TOKEN rows; convert to tile rows (ceil), then let the runtime
        // picker size THIS expert's chunks. The picker derives chunk_M_tiles (hence
        // per_core_M and the chunk count) from this expert's own count, so no
        // expected-token argument is needed and each expert's work scales to its
        // actual load. All three kernels (reader/compute/writer) run the identical
        // picker on the same count, so they agree on the row mapping; rows past the
        // count in the tail chunk are zero-filled by the reader and dropped by the
        // writer.
        // The count is device-produced and unvalidated, so clamp it to this
        // program's capacity before deriving the chunk layout — arithmetic, so the
        // bound holds in Release where ASSERT is a no-op. Reader and writer apply
        // the identical clamp, keeping the three row mappings in lockstep (see
        // adaptive_chunk::clamp_count_tiles).
        const uint32_t count_tiles_raw = (count_value + 31) / 32;
        const uint32_t count_tiles =
            adaptive_chunk::clamp_count_tiles(count_tiles_raw, chunk_M_max, num_chunks_max, m_tiles_full);
        ASSERT(count_tiles == count_tiles_raw);
        const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M_max);

        for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
            // Per-chunk per_core_M (per_core_M_max for full chunks, a smaller divisor
            // for the tail). The gate/up + multiply phases do per_core_M rows of real
            // work; the down matmul keeps its full compile-time ring and MAC-skips
            // rows >= per_core_M (see matmul_phase). re_eff_out_gu = per_core_M *
            // per_core_N_gu (g_in1_num_subblocks * gu_out_subblock_num_tiles ==
            // per_core_N_gu since gu_out_subblock_h == 1).
            const uint32_t re_m_valid = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M_max);
            const uint32_t re_eff_out_gu = re_m_valid * g_in1_num_subblocks * gu_out_subblock_num_tiles;
            //
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
                gu_out_block_num_tiles,
                /*x_cb_id=*/cb_in0_x,
                /*x_rm_cb_id=*/cb_x_rm,
                /*tilize_x=*/(x_is_row_major != 0)>(
                cb_in1_gate,
                cb_in1_up,
                cb_partials_gu,
                cb_partials_up,
                cb_gate_intermed,
                cb_up_intermed,
                /*m_subblocks=*/re_m_valid,
                /*n_subblocks=*/gu_valid_n_subblocks);

#ifdef FUSED_BINARY_ACT
            // Phase 3: fused activation on the raw bf16 accumulators -> cb_activated.
            // cb_in1_up is the unpacker's last SrcA operand, passed so the partials reconfig
            // (weights df -> Float16_b) actually fires.
            binary_activation_phase<gu_out_block_num_tiles, g_in1_per_core_w>(
                cb_in1_up, cb_partials_gu, cb_partials_up, cb_activated, re_eff_out_gu, cb_gate_bias, cb_up_bias);
            (void)cb_gate_intermed;
            (void)cb_up_intermed;
#else
            // Phase 3: elementwise multiply (cb_gate_intermed is silu(partials_gu)
            // in bf8; cb_partials_up is the up matmul accumulator in bf16). The
            // multiply does the format conversion via reconfig_data_format inside
            // multiply_phase — both unpacker srcs get reset to the input CB
            // formats.
            multiply_phase<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
                cb_gate_intermed, cb_partials_up, cb_activated, re_eff_out_gu);
            (void)cb_up_intermed;
#endif

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
                /*apply_silu_on_final=*/false,
                /*d_per_core_N=*/d_in1_per_core_w,
                /*real_k_tiles=*/d_K_down_tiles>(
                cb_in0_down_full,
                cb_in1_down,
                cb_partials_d,
                cb_out,
                /*m_subblocks=*/re_m_valid,
                /*n_subblocks=*/d_valid_n_subblocks,
                cb_down_bias);
        }  // end chunk loop

#ifdef FUSE_BIAS
        // Pop this expert's biases so the reader can refill for the next expert.
        CircularBuffer(cb_gate_bias).pop_front(g_in1_per_core_w);
        CircularBuffer(cb_up_bias).pop_front(g_in1_per_core_w);
        CircularBuffer(cb_down_bias).pop_front(d_in1_per_core_w);
#endif
    }  // end per-local-expert loop
}
