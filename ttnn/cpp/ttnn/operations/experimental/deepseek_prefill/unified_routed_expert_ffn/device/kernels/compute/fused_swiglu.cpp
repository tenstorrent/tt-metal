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
//     accumulated sum, and a second pass copies each subblock back into dst
//     and packs it into the final CB (cb_out). The fused gate/up phase instead
//     pushes subblock by subblock during its last K-block and activates each one
//     on the PACK thread as it lands (see matmul_phase_fused_gu).
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
// Profiler-only hardware counters / FW throttle samples (MATH thread; empty without PROFILE_KERNEL).
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/compute/re_hwprof.hpp"

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

// SwiGLU-OAI computes (clamp(up,±L)+1) * clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L))
// (SwiGLUConfigGPTOSS: alpha=1.702, clamp_limit=7.0, matching M3's config.json).
// SiTU-GLU computes (beta_gate*tanh(gate/beta_gate)*sigmoid(gate)) * (beta_up*tanh(up/beta_up))
// (SituGluConfigKimi: beta_gate=4.0, beta_up=25.0, Kimi K3's config).
// Clamped SiLU-GLU computes silu(min(gate,L)) * clamp(up,±L) (ClampedSiluGluConfigDsV4: limit=10.0).
// binary_act_sfpu.h holds op-local copies of the three stock SFPU ops (moe_gpt swiglu_sfpu.h,
// ckernel_sfpu_situ_glu.h, ckernel_sfpu_clamped_silu_glu.h) with the same math in fewer SFPU
// instructions per row; see that header. This kernel runs the activation on PACK.
#if defined(FUSED_BINARY_ACT) && defined(TRISC_PACK)
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/compute/binary_act_sfpu.h"
#endif

// Dst-accumulator mode (fp32 dest accum on/off) from the host ComputeConfig,
// passed via -DFP32_DEST_ACC_EN. Defaults to bf16 dst (0) if not passed.
#ifndef FP32_DEST_ACC_EN
#define FP32_DEST_ACC_EN 0
#endif

// The gate/up activation runs its SFPU op on the PACK thread (the apply_activation_from_pack /
// moe_gpt pattern), so it overlaps the FPU work of the last gate/up K-block on MATH (see
// matmul_phase_fused_gu). All three binary variants share the (gate, up, out) dst-index signature
// and bake their constants into a config struct; binary_act_sfpu.h dispatches on the variant define.
#if defined(FUSED_BINARY_ACT)
#define PACK_ACT_INIT() \
    PACK((ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::re_act::binary_act_init)))
#define PACK_ACT_TILE(g, u, o)                       \
    PACK((SFPU_BINARY_CALL(                          \
        DST_SYNC_MODE,                               \
        DST_ACCUM_MODE,                              \
        re_act::binary_act,                          \
        ((FP32_DEST_ACC_EN != 0), 8 /*ITERATIONS*/), \
        g,                                           \
        u,                                           \
        o,                                           \
        ::ckernel::VectorMode::RC)))
#else
#define PACK_ACT_INIT() silu_tile_init_pack()
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
    // Width of the LAST N-subblock. Equals out_subblock_w for an exact subblock grid; smaller
    // when the grid is ragged, which the program factory allows only when no divisor of
    // per_core_N clears the width where this matmul stops being math-bound.
    uint32_t out_subblock_w_tail = out_subblock_w,
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
    static_assert(out_subblock_w_tail >= 1 && out_subblock_w_tail <= out_subblock_w, "bad tail width");
    static_assert(
        in0_num_subblocks * ((in1_num_subblocks - 1) * out_subblock_w + out_subblock_w_tail) * out_subblock_h ==
            out_block_num_tiles,
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
    // Tiles per output row once the ragged tail is counted -- i.e. the true per_core_N.
    constexpr uint32_t N_PER_ROW = (in1_num_subblocks - 1) * out_subblock_w + out_subblock_w_tail;
    const uint32_t EFF_OUT = m_subblocks * N_PER_ROW * out_subblock_h;
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
        RE_HW_SAMPLE_THROTTLE();
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
                    // Ragged grid: the last N-subblock is narrower, so it MACs and packs fewer
                    // tiles. Identical to out_subblock_w on an exact grid.
                    const uint32_t this_w = (sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w;
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
                                    this_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index += 1;
                                in1_index += in1_per_core_w;
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < this_w * out_subblock_h; ++i) {
                            pack_tile<true>(i, partials_cb_id, partials_slot_idx + i);
                        }
                        tile_regs_release();
                    }
                    partials_slot_idx += this_w * out_subblock_h;

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

    // Walk (sb_m, sb_n) rather than a flat subblock count: on a ragged grid the last
    // N-subblock is narrower, so the drain must pop and push exactly what the MAC packed.
    // Tiles still leave in row-major (m, n) order and still total m_subblocks * per_core_N,
    // so the writer's fixed drain contract is unchanged -- only the subblock widths differ.
    uint32_t flat_base = 0;
    for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
        for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
            const uint32_t this_w = (sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w;
            const uint32_t this_tiles = this_w * out_subblock_h;
            tile_regs_acquire();
            partials_cb.wait_front(this_tiles);
            for (uint32_t i = 0; i < this_tiles; ++i) {
#ifdef FUSE_BIAS
                // Column of this tile within the core's N slice, from the true (m, n) position.
                const uint32_t flat = flat_base + i;
                add_tiles_bcast_rows(partials_cb_id, down_bias_cb_id, i, flat % d_per_core_N, i);
#else
                copy_tile(partials_cb_id, i, i);
#endif
            }
            partials_cb.pop_front(this_tiles);

            tile_regs_commit();

            if constexpr (apply_silu_on_final) {
                apply_activation_from_pack<KernelActivation::SILU>(this_tiles);
            } else {
                tile_regs_wait();
            }

            final_cb.reserve_back(this_tiles);
            for (uint32_t i = 0; i < this_tiles; ++i) {
                pack_tile(i, final_cb_id);
            }
            final_cb.push_back(this_tiles);

            tile_regs_release();
            flat_base += this_tiles;
        }
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
    // Pad the SAME ragged width sequence the writer waits on, row by row: it drains
    // in0_num_subblocks M-rows of in1_num_subblocks subblocks each, and a uniform-width pad
    // would desynchronise its wait_front on a ragged grid.
    for (uint32_t sb_m = EFF_M; sb_m < in0_num_subblocks; ++sb_m) {
        for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
            const uint32_t pad_tiles =
                ((sb_n + 1 == in1_num_subblocks) ? out_subblock_w_tail : out_subblock_w) * out_subblock_h;
            final_cb.reserve_back(pad_tiles);
            final_cb.push_back(pad_tiles);
        }
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
// The LAST K-block is different: it is walked subblock by subblock, and each subblock's
// activation (binary op -> cb_activated, or silu -> cb_gate_intermed) runs on the PACK thread's
// SFPU while MATH computes the next subblock's last-block matmul. See the pipeline comment at
// the K-loop.
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
    // x_cb_id / x_rm_cb_id / x_tl_cb_id are compile-time so the tilize helper (which takes the
    // input/output CB as template args, like conv_bmm_tilize.cpp) can consume them.
    uint32_t x_cb_id,
    uint32_t x_rm_cb_id,
    uint32_t x_tl_cb_id,
    bool tilize_x = false>
FORCE_INLINE void matmul_phase_fused_gu(
    uint32_t gate_cb_id,
    uint32_t up_cb_id,
    uint32_t partials_gu_cb_id,
    uint32_t partials_up_cb_id,
    uint32_t gate_intermed_cb_id,
    uint32_t up_intermed_cb_id,
    // Binary activations write cb_activated directly (bias-added first on gpt-oss); SiLU writes
    // gate_intermed_cb_id. See the activation pipeline at the K-loop.
    uint32_t activated_cb_id,
    uint32_t gate_bias_cb_id,
    uint32_t up_bias_cb_id,
    uint32_t m_subblocks,
    uint32_t n_subblocks,
    // Row-shared tilize (tilize_x only): this chunk's first gate/up block number, the row's owner
    // slot count and this core's slot. Block x_blk_base + b is tilized by the core in slot
    // (x_blk_base + b) % x_slots alone (adaptive_chunk::x_owner_col: never the column's weight
    // sender); the reader multicasts the tiles to the rest of the row.
    uint32_t x_blk_base = 0,
    uint32_t x_slots = 1,
    uint32_t x_rank = 0) {
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

    // Row-major x, row-shared tilize. Only the block's owner tilizes it: cb_x_rm (bf16, read by
    // its own reader) -> cb_x_tl (bf8), which its reader multicasts into cb_in0_x (x_cb) on every
    // core of the row. The owner tilizes ONE block ahead -- block b+1 before its matmul of block b,
    // block 0 before the loop -- so the multicast lands while the row still computes block b.
    // A core that does not own a block runs straight through to its matmul with no tilize and no
    // unpack/pack reconfig.
    //
    // L1_ACC is turned off so the tilize packs OVERWRITE cb_x_tl rather than accumulate; the
    // shared tilize helper (same one conv_bmm_tilize.cpp uses) then reconfigures unpack SrcA +
    // pack format, drives the per-strip wait/reserve/tilize/push/pop over the runtime
    // x_block_tiles / in0_block_w tile-rows (adaptive per_core_M), and restores init on exit. The
    // helper left SrcA pointing at the bf16 row-major input, so restore it to the gate/up weight
    // format before resuming the matmul (SrcB still holds x_cb_id — the BH tilize path never
    // touches it); then restore the partials packer + L1_ACC state (`l1_acc` = that of the block
    // whose matmul runs next).
    [[maybe_unused]] auto tilize_owned_x_block = [&](uint32_t l1_acc) {
        // Instantiated only on the row-major path: on TILE x the tilize CBs are not allocated.
        if constexpr (tilize_x) {
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
            const uint32_t n_strips = x_block_tiles / in0_block_w;
            compute_kernel_lib::tilize<
                in0_block_w,
                x_rm_cb_id,
                x_tl_cb_id,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(n_strips);
            // The helper consumed/produced only the runtime rows, but the reader pushed a full max
            // cb_x_rm block and pops a full max cb_x_tl block. Settle both with pointer-only bumps (no
            // tilize work on the stale rows) so every CB moves a constant-size block.
            const uint32_t x_pad = X_BLOCK_TILES_MAX - x_block_tiles;
            if (x_pad > 0) {
                CircularBuffer x_rm_cb(x_rm_cb_id);
                CircularBuffer x_tl_cb(x_tl_cb_id);
                x_rm_cb.wait_front(x_pad);
                x_rm_cb.pop_front(x_pad);
                x_tl_cb.reserve_back(x_pad);
                x_tl_cb.push_back(x_pad);
            }
            reconfig_data_format_srca(gate_cb_id);
            matmul_block_init(x_cb_id, gate_cb_id, 0, out_subblock_w, out_subblock_h, in0_block_w);
            pack_reconfig_data_format(x_tl_cb_id, partials_gu_cb_id);
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(l1_acc)));
#else
            (void)l1_acc;
#endif
        } else {
            (void)l1_acc;
        }
    };
    // Owner slot of the next block to tilize (block 0 here, block + 1 in the loop).
    [[maybe_unused]] uint32_t x_slot = 0;
    if constexpr (tilize_x) {
        x_slot = x_blk_base % x_slots;
        if (x_slot == x_rank) {
            tilize_owned_x_block(0);
        }
    }

    // ---- Activation pipelined into the last K-block ------------------------------------------
    // Subblock s of the output is final once the LAST K-block has packed it, so its activation
    // (bias add + binary SFPU op -> cb_activated, or silu -> cb_gate_intermed) can run while the
    // FPU computes the last-block matmuls of the next subblock. The SFPU op runs on the PACK thread
    // (apply_activation_from_pack pattern), MATH only moves the partials into dst, so the FPU and
    // SFPU work overlap instead of running back to back after the K-loop.
    //
    // The last block is walked in LW-wide subblocks (LW = out_subblock_w, halved when gate + up of a
    // full subblock would not fit one dst half), so each activation chunk has a matmul to hide
    // behind. A MATH commit waits for the SFPU to go idle (dest_section_flip), so each SFPU chunk
    // overlaps exactly one dst acquire of matmul work: gate and up share that acquire when 2 * LW
    // fits a dst half.
    //
    // Hand-off: after its last-block packs, subblock s's LW slots are pushed on both partials CBs,
    // so pack_tile<true> indices are relative to the pushed pointer (always 0..LW-1 here) and the
    // slot addresses are the same ones blocks 0..N-2 accumulated into. The activation pops what it
    // read. Pads (pointer-only) keep every CB moving the constant max block, as before.
#ifdef FUSED_BINARY_ACT
    const uint32_t act_out_cb_id = activated_cb_id;
    (void)gate_intermed_cb_id;
#ifndef FUSE_BIAS
    (void)gate_bias_cb_id;
    (void)up_bias_cb_id;
#endif
    constexpr uint32_t kActSlots = 2;  // gate and up pinned in dst together per activated tile
#else
    // SiLU: silu(gate) -> cb_gate_intermed (bf8); multiply_phase pairs it with cb_partials_up, which
    // therefore stays resident here (pushed, never popped).
    const uint32_t act_out_cb_id = gate_intermed_cb_id;
    (void)activated_cb_id;
    (void)gate_bias_cb_id;
    (void)up_bias_cb_id;
    constexpr uint32_t kActSlots = 1;
#endif
    // Up partials are NOT copied to a separate cb_up_intermed (see multiply_phase).
    (void)up_intermed_cb_id;
    CircularBuffer act_out_cb(act_out_cb_id);

    static_assert(out_subblock_h == 1, "the last-block walk indexes tile-rows by sb_m");
    constexpr uint32_t kDstHalf = (FP32_DEST_ACC_EN != 0) ? 4u : 8u;
    constexpr uint32_t kActChunk = kDstHalf / kActSlots;
    constexpr uint32_t LW =
        (2 * out_subblock_w <= kDstHalf || (out_subblock_w % 2) != 0) ? out_subblock_w : out_subblock_w / 2;
    constexpr bool kFusedGU = (2 * LW <= kDstHalf);
    constexpr uint32_t LN = (in1_num_subblocks * out_subblock_w) / LW;  // LW-subblocks per tile-row
    static_assert(LN * LW == in1_num_subblocks * out_subblock_w, "LW must tile per_core_N");
    static_assert(((LW + 1) / 2) * kActSlots <= kDstHalf, "activation half-chunk must fit a dst half");
    // Phantom columns end on an out_subblock_w boundary, so an LW-subblock is real iff it starts below it.
    const uint32_t real_cols = n_subblocks * out_subblock_w;
    const uint32_t l1_acc_last = (num_blocks > 1) ? 1u : 0u;  // L1_ACC state of the last block's packs

    // Mode switches between the last-block matmul and the activation's partials -> dst pass. Every
    // thread runs the same control flow, so `in_act` agrees across UNPACK / MATH / PACK.
    bool in_act = false;
    auto enter_act = [&]() {
        if (in_act) {
            return;
        }
#if defined(FUSED_BINARY_ACT) && defined(FUSE_BIAS)
        // Gate/up bias, broadcast across rows (gpt-oss). Read once per expert by the reader and
        // popped by the caller at the expert's end.
        CircularBuffer(gate_bias_cb_id).wait_front(in1_per_core_w);
        CircularBuffer(up_bias_cb_id).wait_front(in1_per_core_w);
        reconfig_data_format(partials_gu_cb_id, gate_bias_cb_id);
        add_bcast_rows_init(partials_gu_cb_id, gate_bias_cb_id);
#else
        reconfig_data_format_srca(partials_gu_cb_id);
        copy_init(partials_gu_cb_id);
#endif
#ifdef PACKER_L1_ACC
        PACK((llk_pack_reconfig_l1_acc(0)));  // activation packs overwrite
#endif
        pack_reconfig_data_format(act_out_cb_id);
        in_act = true;
    };
    auto leave_act = [&]() {
        if (!in_act) {
            return;
        }
        reconfig_data_format(gate_cb_id, x_cb_id);  // in1 -> SrcA, in0 -> SrcB
        matmul_block_init(x_cb_id, gate_cb_id, /*transpose=*/0, LW, out_subblock_h, in0_block_w);
        pack_reconfig_data_format(partials_gu_cb_id);
#ifdef PACKER_L1_ACC
        PACK((llk_pack_reconfig_l1_acc(l1_acc_last)));
#endif
        in_act = false;
    };
    // Activate tiles [lo, hi) of the oldest handed-off subblock (rd pointer of the partials sits on
    // its first tile; flat_base = its first flat (m, n) index, for the bias column).
    auto act_tiles = [&](uint32_t lo, uint32_t hi, [[maybe_unused]] uint32_t flat_base) {
        partials_gu_cb.wait_front(LW);
#ifdef FUSED_BINARY_ACT
        partials_up_cb.wait_front(LW);
#endif
        for (uint32_t base = lo; base < hi; base += kActChunk) {
            const uint32_t c = (hi - base) < kActChunk ? (hi - base) : kActChunk;
            tile_regs_acquire();
            for (uint32_t j = 0; j < c; ++j) {
#ifdef FUSED_BINARY_ACT
#ifdef FUSE_BIAS
                const uint32_t ncol = (flat_base + base + j) % in1_per_core_w;
                add_tiles_bcast_rows(partials_gu_cb_id, gate_bias_cb_id, base + j, ncol, j);
                add_tiles_bcast_rows(partials_up_cb_id, up_bias_cb_id, base + j, ncol, c + j);
#else
                copy_tile(partials_gu_cb_id, base + j, j);
                copy_tile(partials_up_cb_id, base + j, c + j);
#endif
#else
                copy_tile(partials_gu_cb_id, base + j, j);
#endif
            }
            tile_regs_commit();
            // PACK: wait for MATH's commit, point the SFPU at the packer's dst half, activate in
            // place (out == gate slot), and let the packer start only once the SFPU is done.
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t j = 0; j < c; ++j) {
#ifdef FUSED_BINARY_ACT
                PACK_ACT_TILE(j, c + j, j);
#else
                silu_tile_pack(j);
#endif
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            act_out_cb.reserve_back(c);
            for (uint32_t j = 0; j < c; ++j) {
                pack_tile(j, act_out_cb_id);
            }
            act_out_cb.push_back(c);
            tile_regs_release();
        }
    };
    // Retire the oldest handed-off subblock's partials slots.
    auto pop_pending = [&]() {
        partials_gu_cb.wait_front(LW);
        partials_gu_cb.pop_front(LW);
#ifdef FUSED_BINARY_ACT
        partials_up_cb.wait_front(LW);
        partials_up_cb.pop_front(LW);
#endif
    };
    // A phantom subblock (no real columns) was never MAC'd: retire its slots and hand the activated
    // stream pointer-only slots, exactly as the old pad did.
    auto skip_pending = [&]() {
        pop_pending();
        act_out_cb.reserve_back(LW);
        act_out_cb.push_back(LW);
    };
    // Last-block matmul of one LW-subblock into dst[dst_base, dst_base + LW).
    auto mm_last = [&](uint32_t in1_cb, uint32_t in0_base, uint32_t in1_base, uint32_t dst_base) {
        uint32_t in0_index = in0_base;
        uint32_t in1_index = in1_base;
        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
            matmul_block(
                x_cb_id, in1_cb, in0_index, in1_index, dst_base, /*transpose=*/0, LW, out_subblock_h, in0_block_w);
            in0_index += 1;
            in1_index += in1_per_core_w;
        }
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        RE_HW_SAMPLE_THROTTLE();
        if constexpr (tilize_x) {
            if (++x_slot == x_slots) {
                x_slot = 0;
            }
            if (block + 1 < num_blocks && x_slot == x_rank) {
                tilize_owned_x_block(block == 0 ? 0 : 1);
            }
        }
        x_cb.wait_front(X_BLOCK_TILES_MAX);
        gate_cb.wait_front(in1_block_num_tiles);
        up_cb.wait_front(in1_block_num_tiles);

        if (block + 1 == num_blocks) {
            break;  // the last block runs pipelined with the activation, below
        }

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

    // ---- Last K-block (its x / gate / up are waited on above) ----
    // Packs keep the L1_ACC state the K-loop left (on for num_blocks > 1, off for a single block).
    if constexpr (LW != out_subblock_w) {
        matmul_block_init(x_cb_id, gate_cb_id, /*transpose=*/0, LW, out_subblock_h, in0_block_w);
    }
    bool pend = false;
    bool pend_real = false;
    uint32_t pend_flat = 0;
    uint32_t flat = 0;
    for (uint32_t sb_m = 0; sb_m < EFF_M; ++sb_m) {
        const uint32_t in0_base = sb_m * in0_subblock_num_tiles;
        for (uint32_t j = 0; j < LN; ++j) {
            const bool real = j * LW < real_cols;
            const uint32_t in1_base = j * LW;
            if constexpr (kFusedGU) {
                // Previous subblock's activation, then this subblock's gate + up in ONE acquire, which
                // MATH runs while PACK's SFPU works on the activation.
                if (pend) {
                    if (pend_real) {
                        enter_act();
                        act_tiles(0, LW, pend_flat);
                        pop_pending();
                    } else {
                        skip_pending();
                    }
                }
                if (real) {
                    leave_act();
                    tile_regs_acquire();
                    mm_last(gate_cb_id, in0_base, in1_base, /*dst_base=*/0);
                    mm_last(up_cb_id, in0_base, in1_base, /*dst_base=*/LW);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < LW; ++i) {
                        pack_tile<true>(i, partials_gu_cb_id, i);
                    }
                    for (uint32_t i = 0; i < LW; ++i) {
                        pack_tile<true>(LW + i, partials_up_cb_id, i);
                    }
                    tile_regs_release();
                }
            } else {
                // Gate and up need a dst half each: split the previous subblock's activation in two,
                // one half in front of each matmul.
                constexpr uint32_t kHalf = (LW + 1) / 2;
                if (pend && pend_real) {
                    enter_act();
                    act_tiles(0, kHalf, pend_flat);
                }
                if (real) {
                    leave_act();
                    tile_regs_acquire();
                    mm_last(gate_cb_id, in0_base, in1_base, /*dst_base=*/0);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < LW; ++i) {
                        pack_tile<true>(i, partials_gu_cb_id, i);
                    }
                    tile_regs_release();
                }
                if (pend) {
                    if (pend_real) {
                        enter_act();
                        act_tiles(kHalf, LW, pend_flat);
                        pop_pending();
                    } else {
                        skip_pending();
                    }
                }
                if (real) {
                    leave_act();
                    tile_regs_acquire();
                    mm_last(up_cb_id, in0_base, in1_base, /*dst_base=*/0);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < LW; ++i) {
                        pack_tile<true>(i, partials_up_cb_id, i);
                    }
                    tile_regs_release();
                }
            }
            // Hand this subblock's (now final) slots to the activation. The push waits for the packer,
            // so the unpacker reads the accumulated values.
            partials_gu_cb.push_back(LW);
            partials_up_cb.push_back(LW);
            pend = true;
            pend_real = real;
            pend_flat = flat;
            flat += LW;
        }
    }
    // The last block's operands are done: free them before the trailing activation so the reader
    // and writer can refill.
    x_cb.pop_front(X_BLOCK_TILES_MAX);
    gate_cb.pop_front(in1_block_num_tiles);
    up_cb.pop_front(in1_block_num_tiles);

    if (pend) {
        if (pend_real) {
            enter_act();
            act_tiles(0, LW, pend_flat);
            pop_pending();
        } else {
            skip_pending();
        }
    }

    // Settle the rest of the constant-size blocks with pointer-only bumps: the partials slots past
    // the runtime rows were never packed, and the activation stream gets the matching pad.
    {
        const uint32_t pad = EFF_OUT_MAX - EFF_OUT;
        if (pad > 0) {
            partials_gu_cb.push_back(pad);
            partials_up_cb.push_back(pad);
            partials_gu_cb.wait_front(pad);
            partials_gu_cb.pop_front(pad);
#ifdef FUSED_BINARY_ACT
            partials_up_cb.wait_front(pad);
            partials_up_cb.pop_front(pad);
#endif
            act_out_cb.reserve_back(pad);
            act_out_cb.push_back(pad);
        }
    }
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // later phases' packs overwrite
#endif
}

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

#ifdef TRISC_UNPACK
// UNPACK only: local expert e's tile-row count from the counts / idx scratch CBs, band-filtered
// and clamped exactly as the reader and writer do (adaptive_chunk::clamp_count_tiles).
template <
    uint32_t min_tokens,
    uint32_t max_tokens,
    uint32_t clamp_chunk,
    uint32_t num_chunks_max,
    uint32_t m_tiles_full>
FORCE_INLINE uint32_t expert_count_tiles(uint32_t cb_counts, uint32_t cb_idx, uint32_t e) {
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_counts).fifo_rd_ptr << 4);
    const volatile tt_l1_ptr uint32_t* idx_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_idx).fifo_rd_ptr << 4);
    const uint32_t count = adaptive_chunk::count_in_band(counts_ptr[idx_ptr[e]], min_tokens, max_tokens);
    const uint32_t count_tiles_raw = (count + 31) / 32;
    const uint32_t count_tiles =
        adaptive_chunk::clamp_count_tiles(count_tiles_raw, clamp_chunk, num_chunks_max, m_tiles_full);
    ASSERT(count_tiles == count_tiles_raw);
    return count_tiles;
}
#endif

}  // namespace

void kernel_main() {
    // Per-core valid N-subblock counts. per_core_N is the GRID-ceil'd width, so the
    // highest-gx cores own phantom output columns whose weights were never fetched;
    // these bound the MAC to the subblocks holding real columns. Runtime (not
    // compile-time) because they vary per core and one kernel serves the whole grid.
    const uint32_t gu_valid_n_subblocks = get_arg_val<uint32_t>(0);
    const uint32_t d_valid_n_subblocks = get_arg_val<uint32_t>(1);
    // Grid M-row: selects this core's expert group (adaptive_chunk::plan_launch).
    const uint32_t my_gy = get_arg_val<uint32_t>(2);
    // Grid column: picks the gate/up x blocks this core tilizes for its row (row-major x only).
    const uint32_t my_gx = get_arg_val<uint32_t>(3);

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
    // Width of the LAST down N-subblock. Equals d_out_subblock_w unless the down subblock grid
    // is ragged -- which the program factory allows only when no divisor of per_core_N_d clears
    // the 3-tile width where the matmul goes unpack-bound.
    constexpr uint32_t d_out_subblock_w_tail = get_compile_time_arg_val(29);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(30);
    // Multi-chunk: the number of chunks is chosen at RUNTIME from each expert's
    // device token count (see the picker below). num_chunks_max is the compile-time
    // upper bound (host = ceil(M_tiles_full / min_chunk)) used only to clamp the
    // runtime chunk count defensively. chunk_M_max is the CB-sized maximum
    // chunk (per_core_M_max * GRID_Y); the picker never returns more than this.
    constexpr uint32_t num_chunks_max = get_compile_time_arg_val(31);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(32);
    // chunk_M_max is the CB-sized MAXIMUM chunk (per_core_M_max * GRID_Y). The
    // runtime picker (adaptive_chunk::num_chunks) sizes the actual chunk to the
    // device token count and never exceeds this.
    constexpr uint32_t chunk_M_max = get_compile_time_arg_val(33);
    // x_is_row_major: tilize cb_x_rm -> cb_in0_x before the gate/up matmul.
    // 0 => x already TILE in cb_in0_x.
    constexpr uint32_t x_is_row_major = get_compile_time_arg_val(34);
    // Real (unpadded) down-K tiles. The down K-loop runs over the GRID-padded
    // extent (K_down_tiles_padded), and the down phase skips every K position at
    // or past this count — whole padded blocks and the partial tail alike. That
    // is what lets the reader and writer leave the padded down weights and the
    // padded hidden (gate/up N-OOB) columns unwritten: nothing ever reduces them.
    constexpr uint32_t d_K_down_tiles = get_compile_time_arg_val(35);
    constexpr uint32_t min_active_tokens = get_compile_time_arg_val(36);
    constexpr uint32_t max_active_tokens = get_compile_time_arg_val(37);

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
    // Row-shared tilize output (x_is_row_major only): bf8 tiles of the x blocks this core owns,
    // multicast by the reader to the whole row. grid_x = cores per M-row.
    constexpr uint32_t cb_x_tl = get_named_compile_time_arg_val("cb_x_tl");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
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
    RE_HW_INIT();

    CircularBuffer counts_scratch_cb(cb_counts_scratch);
    CircularBuffer idx_scratch_cb(cb_idx_scratch);

    // Wait for the reader (NCRISC) to push the counts/idx into shared L1. They
    // are pushed ONCE and stay resident, so UNPACK can re-index them per expert.
    counts_scratch_cb.wait_front(1);
    idx_scratch_cb.wait_front(1);

    // The gate/up activation (SiLU, or the binary SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU op)
    // runs its SFPU work on the PACK thread, pipelined into the last gate/up K-block (see
    // matmul_phase_fused_gu), so the SFPU is initialised on PACK. Init once, shared across all
    // experts: the binary ops' tables (recip for SwiGLU-OAI and clamped SiLU-GLU, tanh for
    // SiTU-GLU) live in the vConstFloatPrgm registers, which nothing between here and the tile
    // calls reprograms. MATH runs no SFPU op in this kernel.
    PACK_ACT_INIT();

    // ======================= per-local-expert loop =======================
    // Run the full gate/up/down FFN for every local expert this core's row group owns.
    //
    // Expert-parallel row groups (adaptive_chunk.hpp): UNPACK reads the counts, builds the same
    // launch plan (per-phase layout and expert -> group map) the reader and writer run, walks its phases, and sends
    // MATH/PACK (whose CB interface cannot reach the counts L1) the tile-row count and group size of each expert this
    // core's group works on, in the order the reader feeds them, then a 0 sentinel. Experts
    // another group owns, or with no rows, never reach MATH/PACK, so the mailbox carries one
    // entry pair per worked expert.
    constexpr uint32_t per_core_M_max = chunk_M_max / adaptive_chunk::kGridY;
    constexpr uint32_t clamp_chunk = per_core_M_max * adaptive_chunk::kMinGroupRows;
#ifdef TRISC_UNPACK
    auto tiles_of = [&](uint32_t e) {
        return expert_count_tiles<min_active_tokens, max_active_tokens, clamp_chunk, num_chunks_max, m_tiles_full>(
            cb_counts_scratch, cb_idx_scratch, e);
    };
#endif
    // Read by UNPACK's scan only.
    [[maybe_unused]] adaptive_chunk::LaunchPlan plan;
    [[maybe_unused]] uint8_t expert_group[2][experts_per_chip];
    [[maybe_unused]] uint32_t phase = 0;
    [[maybe_unused]] uint32_t next_expert = 0;
    UNPACK(({ plan = adaptive_chunk::plan_launch(tiles_of, per_core_M_max, expert_group); }));
    // Gate/up K-blocks done so far, counted exactly as the reader counts them (x_blk there):
    // every core of a row runs the same experts and chunks, so they agree on each block's owner.
    uint32_t x_blk = 0;

    while (true) {
        // Next expert this group works on: its tile-row count, already band-filtered and clamped
        // (0 = no more experts). UNPACK scans, MATH/PACK receive.
        uint32_t count_tiles = 0;
        // Group height in the low half, this core's row within the group (the x tilize owner
        // rotation skips its weight-sender columns, see adaptive_chunk::x_owner_col) in the high
        // half: one mailbox word per expert.
        uint32_t group_word = 0;
        UNPACK(({
            while (count_tiles == 0) {
                if (next_expert == experts_per_chip || plan.layout[phase] == adaptive_chunk::kNoLayout) {
                    if (phase == 1) {
                        break;
                    }
                    phase = 1;
                    next_expert = 0;
                    continue;
                }
                uint32_t group_base = 0;
                uint32_t rows = 0;
                const uint32_t my_group = adaptive_chunk::layout_group(
                    adaptive_chunk::layout_at(adaptive_chunk::kGridY, plan.layout[phase]), my_gy, group_base, rows);
                if (expert_group[phase][next_expert] == my_group) {
                    count_tiles = tiles_of(next_expert);
                    group_word = rows | ((my_gy - group_base) << 16);
                }
                ++next_expert;
            }
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count_tiles);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count_tiles);
            if (count_tiles != 0) {
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, group_word);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, group_word);
            }
        }));
        MATH(count_tiles = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(count_tiles = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        if (count_tiles == 0) {
            break;
        }
        MATH(group_word = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(group_word = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        const uint32_t group_rows = group_word & 0xFFFF;
        const uint32_t my_mt = group_word >> 16;
        // Row-shared tilize owners of this row (adaptive_chunk::x_owner_col): slot count, my slot.
        const uint32_t x_key = adaptive_chunk::x_owner_key(my_gy - my_mt, group_rows, my_mt);
        const uint32_t x_slots = adaptive_chunk::x_owner_slots(grid_x, x_key & 0xFFFF, x_key >> 16);
        const uint32_t x_rank = adaptive_chunk::x_owner_rank(my_gx, x_key & 0xFFFF, x_key >> 16, x_slots);
        const uint32_t chunk_M = per_core_M_max * group_rows;
        const uint32_t effective_chunks = adaptive_chunk::num_chunks(count_tiles, chunk_M);

        for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
            // Per-chunk per_core_M (per_core_M_max for full chunks, a smaller divisor
            // for the tail). The gate/up + multiply phases do per_core_M rows of real
            // work; the down matmul keeps its full compile-time ring and MAC-skips
            // rows >= per_core_M (see matmul_phase). re_eff_out_gu = per_core_M *
            // per_core_N_gu (g_in1_num_subblocks * gu_out_subblock_num_tiles ==
            // per_core_N_gu since gu_out_subblock_h == 1).
            const uint32_t re_m_valid = adaptive_chunk::per_core_M_for_chunk(chunk, count_tiles, chunk_M, group_rows);
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
            //
            // Phase zones (MATH thread only, so one marker pair per phase per core): RE-GU, RE-ACT
            // and RE-DN tile the per-expert timeline, waits included, to split the per-expert
            // fixed cost between gate/up fill, activation and the down phase.
            {
#ifdef TRISC_MATH
                DeviceZoneScopedN("RE-GU");
#endif
                RE_HW_BEGIN(0);
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
                    /*x_tl_cb_id=*/cb_x_tl,
                    /*tilize_x=*/(x_is_row_major != 0)>(
                    cb_in1_gate,
                    cb_in1_up,
                    cb_partials_gu,
                    cb_partials_up,
                    cb_gate_intermed,
                    cb_up_intermed,
                    cb_activated,
                    cb_gate_bias,
                    cb_up_bias,
                    /*m_subblocks=*/re_m_valid,
                    /*n_subblocks=*/gu_valid_n_subblocks,
                    /*x_blk_base=*/x_blk,
                    x_slots,
                    x_rank);
                // Tile-mults this window issued: per_core_M rows x real N columns x K, gate and up.
                RE_HW_END(0, re_m_valid * gu_valid_n_subblocks * gu_out_subblock_w * g_in0_block_w * g_num_blocks * 2);
            }
            x_blk += g_num_blocks;

            {
#ifdef TRISC_MATH
                DeviceZoneScopedN("RE-ACT");
#endif
#ifdef FUSED_BINARY_ACT
                // Phase 3: the binary activation already ran inside the gate/up phase (pipelined into
                // its last K-block) and pushed cb_activated.
                (void)re_eff_out_gu;
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
            }

            // Phase 4: down matmul, output to cb_out.
            // multiply_phase left the unpacker on (cb_gate_intermed, cb_partials_up);
            // matmul_block_init does not re-program data formats, so reset the down
            // operands here (in1 -> SrcA, in0 -> SrcB) before the matmul.
            {
#ifdef TRISC_MATH
                DeviceZoneScopedN("RE-DN");
#endif
                RE_HW_BEGIN(1);
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
                    /*out_subblock_w_tail=*/d_out_subblock_w_tail,
                    /*d_per_core_N=*/d_in1_per_core_w,
                    /*real_k_tiles=*/d_K_down_tiles>(
                    cb_in0_down_full,
                    cb_in1_down,
                    cb_partials_d,
                    cb_out,
                    /*m_subblocks=*/re_m_valid,
                    /*n_subblocks=*/d_valid_n_subblocks,
                    cb_down_bias);
                RE_HW_END(
                    1,
                    re_m_valid * d_K_down_tiles *
                        ((d_valid_n_subblocks == d_in1_num_subblocks)
                             ? (d_in1_num_subblocks - 1) * d_out_subblock_w + d_out_subblock_w_tail
                             : d_valid_n_subblocks * d_out_subblock_w));
            }
        }  // end chunk loop

#ifdef FUSE_BIAS
        // Pop this expert's biases so the reader can refill for the next expert (the reader
        // pushes them only for the experts this group works on).
        CircularBuffer(cb_gate_bias).pop_front(g_in1_per_core_w);
        CircularBuffer(cb_up_bias).pop_front(g_in1_per_core_w);
        CircularBuffer(cb_down_bias).pop_front(d_in1_per_core_w);
#endif
    }  // end per-local-expert loop
    RE_HW_EMIT();
}
