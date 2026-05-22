// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Fused DeepSeek-V3 routed-expert FFN (SwiGLU) compute kernel.
//
// Logically this kernel computes, for the M-chunk this core is responsible for:
//
//      gate     = x @ Wg                                  (matmul #1: gate proj)
//      gate     = silu(gate)                              (fused in pack of #1)
//      up       = x @ Wu                                  (matmul #2: up proj)
//      activated= gate * up                               (elementwise multiply)
//      y        = activated @ Wd                          (matmul #3: down proj)
//
// All four classical "ops" stay resident in one Risc-V compute pass — only the
// final `y` block is handed to the writer. This collapses what used to be four
// rows in tt-perf-report into a single fused row.
//
// CB choreography (per chunk on this core):
//
//   in0_x            : reader streams one K-block of x at a time. Reused
//                      across phase 1 (gate) and phase 2 (up) — the reader
//                      pushes the SAME tiles twice (it'll be cheap because they
//                      can stay in DRAM read-ahead, or come from an L1 scratch
//                      CB; this kernel doesn't care, it just wait_front's).
//   in1_gate         : reader streams one K-block of Wg per K-block.
//   in1_up           : reader streams one K-block of Wu per K-block.
//   in1_down         : reader streams one K-block of Wd per K-block.
//   cb_gate_intermed : block-sharded L1 buffer holding the gate-silu result
//                      for the whole core's per_core_M x per_core_N tile block
//                      (gate_per_core_M * gate_per_core_N tiles).
//   cb_up_intermed   : same shape; holds the up matmul result.
//   cb_activated    : per-core block-sharded L1 buffer for the elementwise
//                      product gate*up. Re-used as in0 (A) for the down matmul.
//   cb_mm_partials_gu: in-flight partials buffer for gate AND up matmul spill
//                      and reload (subblock-sized). The two matmuls don't run
//                      concurrently so the same partials CB is fine.
//   cb_mm_partials_d : in-flight partials buffer for the down matmul.
//   cb_out           : single-subblock CB the writer drains to the DRAM output.
//
// All shape constants are passed as positional compile-time args; CB ids are
// passed as named compile-time args. The kernel handles only ONE M-chunk —
// the host sequences chunks externally for v1.
// ============================================================================

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "internal/mod_div_lib.h"

#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"  // silu_tile_init_pack / silu_tile_pack

// ---------------------------------------------------------------------------
// reload_from_cb_to_dst — port of bmm_large_block_zm_fused_bias_activation's
// helper. Pulls a previously-packed subblock back from the partials CB into
// DST so the next K-block iteration of matmul can accumulate on top of it.
// ---------------------------------------------------------------------------
FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    CircularBuffer mm_partials_cb(mm_partials_cb_id);
    // The unpacker srcA was set up for in1_cb_id; reconfig it to read from
    // mm_partials_cb (the spilled output of the previous K-block iteration).
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    mm_partials_cb.wait_front(out_subblock_num_tiles);
    copy_block_matmul_partials(mm_partials_cb_id, 0, 0, out_subblock_num_tiles);
    mm_partials_cb.pop_front(out_subblock_num_tiles);
    // Back to matmul mode for srcA == in0_cb_id.
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, /*transpose=*/0, out_subblock_w, out_subblock_h, in0_block_w);
}

// ---------------------------------------------------------------------------
// matmul_phase — One full per-core M x N output block built as
//     subblocks(in0_num_subblocks * in1_num_subblocks)
// across num_blocks K-iterations. Writes the final per-block result to
// `final_out_cb_id`; uses `partials_cb_id` for spill-and-reload between K
// iterations.
//
// Behavior knobs:
//   - apply_silu_in_pack: after the last K block, before packing each
//     subblock, run silu over the dst tiles. Used for phase 1 (gate proj).
//
// This matches the structure of bmm_large_block_zm_fused_bias_activation's
// inner block loop, but specialized to no batching, no row/column outer
// blocks, no bias, no untilize, and a fixed activation (silu) gated by a
// template flag.
// ---------------------------------------------------------------------------
template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    bool apply_silu_in_pack>
FORCE_INLINE void matmul_phase(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t partials_cb_id, uint32_t final_out_cb_id) {
    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer partials_cb(partials_cb_id);
    CircularBuffer final_out_cb(final_out_cb_id);

    constexpr bool spill = num_blocks > 1;
    bool enable_reload = false;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const bool last_out = (block == num_blocks - 1);

        // Reader synchronization: in0 (x or activated) and in1 (Wg/Wu/Wd
        // K-block tile-row).
        in0_cb.wait_front(in0_block_num_tiles);
        in1_cb.wait_front(in1_block_num_tiles);

        // Reserve room in the final output CB once, on the first block, but
        // only if we'll need to spill (last_out spills directly anyway).
        if (block == 0 && !last_out) {
            final_out_cb.reserve_back(out_block_num_tiles);
        }

        uint32_t in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
            uint32_t in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
                tile_regs_acquire();

                // Pull the previous K-block's partial sum back into DST so
                // matmul_block accumulates on top.
                if (enable_reload) {
                    reload_from_cb_to_dst(
                        in0_cb_id,
                        in1_cb_id,
                        partials_cb_id,
                        out_subblock_num_tiles,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                }

                // Accumulate this K-block's contribution into DST. matmul_block
                // strides over in0_block_w along the inner dim internally; we
                // walk over (in0_num_subblocks x in1_num_subblocks) here.
                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_subblock_offset;
                uint32_t in1_index = in1_index_subblock_offset;
                for (uint32_t inner = 0; inner < in0_block_w; ++inner) {
                    matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_index,
                        in1_index,
                        dst_index,
                        /*transpose=*/false,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                    in0_index += 1;             // stride right within in0 block
                    in1_index += in1_block_w;   // stride down within in1 block
                }

                if (last_out) {
                    // Final pack of this subblock — goes to the FINAL CB (the
                    // intermediate L1 buffer or the activated CB), optionally
                    // with silu fused in the pack step.
                    tile_regs_commit();
                    final_out_cb.reserve_back(out_subblock_num_tiles);

                    if constexpr (apply_silu_in_pack) {
                        // Apply silu(dst[i]) for each tile in the subblock,
                        // riding the PACK thread so we don't have to copy
                        // through MATH. This mirrors the pattern in
                        // apply_activation_from_pack() in bmm_fused_activation.
                        PACK(TTI_SEMWAIT(
                            p_stall::STALL_TDMA | p_stall::STALL_CFG,
                            semaphore::t6_sem(semaphore::MATH_PACK),
                            p_stall::STALL_ON_ZERO));
                        PACK(TT_SETC16(
                            DEST_TARGET_REG_CFG_MATH_Offset_ADDR32,
                            ckernel::packer::get_packer_dest_offset()));
                        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                            silu_tile_pack(i);
                        }
                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    } else {
                        tile_regs_wait();
                    }

                    // Pack subblock to final destination.
                    pack_tile_block(/*start_dst_index=*/0, final_out_cb_id, out_subblock_num_tiles);
                    tile_regs_release();
                    final_out_cb.push_back(out_subblock_num_tiles);
                } else {
                    // Intermediate K-block — pack to partials CB for reload
                    // next iteration.
                    tile_regs_commit();
                    partials_cb.reserve_back(out_subblock_num_tiles);
                    tile_regs_wait();
                    pack_tile_block(/*start_dst_index=*/0, partials_cb_id, out_subblock_num_tiles);
                    tile_regs_release();
                    partials_cb.push_back(out_subblock_num_tiles);
                }

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        // Spill bookkeeping: between blocks (other than the second-to-last
        // ➝ last transition) we throw away the previous block's partials
        // because we just re-spilled. Same scheme as
        // bmm_large_block_zm_fused_bias_activation.cpp.
        if constexpr (spill) {
            if (block < num_blocks - 2) {
                // Drain the partials CB in subblock-sized chunks (CB API
                // requires constant-size wait_front increments).
                for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                    partials_cb.wait_front(out_subblock_num_tiles);
                    partials_cb.pop_front(out_subblock_num_tiles);
                }
            }
            if (block == num_blocks - 2) {
                enable_reload = true;
            }
        }

        in0_cb.pop_front(in0_block_num_tiles);
        in1_cb.pop_front(in1_block_num_tiles);
    }
}

// ---------------------------------------------------------------------------
// elementwise_multiply_phase — gate_silu (in cb_gate_intermed) * up (in
// cb_up_intermed) → cb_activated. Both inputs are full per-core blocks of
// size per_core_M * per_core_N tiles (= 8 * 6 = 48 in v1). We process them in
// subblock-sized chunks so the dst regs aren't overrun (max 8 tiles at lo-fi
// half-sync, 16 at full-sync) and so the pack/unpack interleave with the
// matmul state machine.
// ---------------------------------------------------------------------------
template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void elementwise_multiply_phase(
    uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id) {
    CircularBuffer gate_cb(gate_cb_id);
    CircularBuffer up_cb(up_cb_id);
    CircularBuffer activated_cb(activated_cb_id);

    // Both intermediates are fully populated (matmul_phase reserved+pushed
    // out_subblock_num_tiles at a time, total = out_block_num_tiles).
    gate_cb.wait_front(out_block_num_tiles);
    up_cb.wait_front(out_block_num_tiles);

    // Switch SFPU/unpacker state from matmul-mode to elementwise-binary-mode.
    // mul_tiles_init handles the unpacker reconfig for srcA <- gate, srcB <-
    // up, and the math-engine reconfig.
    mul_tiles_init(gate_cb_id, up_cb_id);

    static_assert(
        out_block_num_tiles % out_subblock_num_tiles == 0,
        "out_block_num_tiles must be a multiple of out_subblock_num_tiles");
    constexpr uint32_t num_subblocks = out_block_num_tiles / out_subblock_num_tiles;

    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        tile_regs_commit();

        activated_cb.reserve_back(out_subblock_num_tiles);
        tile_regs_wait();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        tile_regs_release();
        activated_cb.push_back(out_subblock_num_tiles);

        base += out_subblock_num_tiles;
    }

    gate_cb.pop_front(out_block_num_tiles);
    up_cb.pop_front(out_block_num_tiles);
}

// ============================================================================
// kernel_main — top-level entry for one M-chunk on one core.
// ============================================================================

void kernel_main() {
    // -------------------------------------------------------------------
    // Compile-time args — see the .hpp/program_factory for the full list.
    //
    // Positional args are grouped by phase:
    //   [ 0..7 ]  gate matmul shape
    //   [ 8..15]  up   matmul shape  (typically identical to gate)
    //   [16..23]  down matmul shape
    //   [24..25]  fused-multiply shape (block & subblock tile counts)
    // -------------------------------------------------------------------

    // ---- Phase 1 (gate proj): x @ Wg → cb_gate_intermed, with silu in pack
    constexpr uint32_t g_in0_block_w           = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks     = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles   = get_compile_time_arg_val(2);
    constexpr uint32_t g_in0_subblock_num_tiles= get_compile_time_arg_val(3);
    constexpr uint32_t g_in1_num_subblocks     = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles   = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_block_w           = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks            = get_compile_time_arg_val(7);

    // ---- Phase 2 (up proj): x @ Wu → cb_up_intermed (same shape as gate, but
    // we expose it separately so the host doesn't have to assume identity).
    constexpr uint32_t u_in0_block_w           = get_compile_time_arg_val(8);
    constexpr uint32_t u_in0_num_subblocks     = get_compile_time_arg_val(9);
    constexpr uint32_t u_in0_block_num_tiles   = get_compile_time_arg_val(10);
    constexpr uint32_t u_in0_subblock_num_tiles= get_compile_time_arg_val(11);
    constexpr uint32_t u_in1_num_subblocks     = get_compile_time_arg_val(12);
    constexpr uint32_t u_in1_block_num_tiles   = get_compile_time_arg_val(13);
    constexpr uint32_t u_in1_block_w           = get_compile_time_arg_val(14);
    constexpr uint32_t u_num_blocks            = get_compile_time_arg_val(15);

    // ---- Phase 4 (down proj): activated @ Wd → cb_out
    constexpr uint32_t d_in0_block_w           = get_compile_time_arg_val(16);
    constexpr uint32_t d_in0_num_subblocks     = get_compile_time_arg_val(17);
    constexpr uint32_t d_in0_block_num_tiles   = get_compile_time_arg_val(18);
    constexpr uint32_t d_in0_subblock_num_tiles= get_compile_time_arg_val(19);
    constexpr uint32_t d_in1_num_subblocks     = get_compile_time_arg_val(20);
    constexpr uint32_t d_in1_block_num_tiles   = get_compile_time_arg_val(21);
    constexpr uint32_t d_in1_block_w           = get_compile_time_arg_val(22);
    constexpr uint32_t d_num_blocks            = get_compile_time_arg_val(23);

    // out_subblock_h / out_subblock_w / per-block tile counts. The same
    // subblock shape is used for gate, up, and the multiply (they live on the
    // same M x N output block per core); down has its own subblock shape.
    constexpr uint32_t gu_out_subblock_h       = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w       = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles  = get_compile_time_arg_val(26);  // per_core_M * per_core_N
    constexpr uint32_t d_out_subblock_h        = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w        = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles   = get_compile_time_arg_val(29);  // per_core_M * per_core_N_down

    // ---- Named CB ids -------------------------------------------------------
    constexpr uint32_t cb_in0_x          = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate       = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up         = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down       = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed  = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_up_intermed    = get_named_compile_time_arg_val("cb_up_intermed");
    constexpr uint32_t cb_activated      = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_mm_partials_gu = get_named_compile_time_arg_val("cb_mm_partials_gu");
    constexpr uint32_t cb_mm_partials_d  = get_named_compile_time_arg_val("cb_mm_partials_d");
    constexpr uint32_t cb_out            = get_named_compile_time_arg_val("cb_out");

    // -------------------------------------------------------------------
    // Initial setup. The first matmul (gate) needs full mm_block_init —
    // subsequent phases only need short reconfigs because the HW config
    // (dataformats, fp32 dest accum, etc.) doesn't change.
    // -------------------------------------------------------------------

    // SFPU needs to be primed for silu before phase 1 — silu_tile_init_pack
    // sets up the PACK-side SFPU state used in matmul_phase's pack fusion.
    silu_tile_init_pack();

    // Full matmul init: configures unpacker HW, math HW, packer HW, and the
    // matmul control words for the gate phase. Subsequent phases only need
    // mm_block_init_short* to flip src CBs / subblock dims without redoing
    // the HW configuration.
    mm_block_init(
        cb_in0_x,
        cb_in1_gate,
        cb_gate_intermed,
        /*transpose=*/0,
        gu_out_subblock_w,
        gu_out_subblock_h,
        g_in0_block_w);

    // =====================================================================
    // PHASE 1 — gate matmul: x @ Wg → cb_gate_intermed
    // Silu is fused in the pack step of the LAST K-block iteration of every
    // subblock; output lands in cb_gate_intermed as gate_silu values.
    // =====================================================================
    matmul_phase<
        g_in0_block_w,
        g_in0_num_subblocks,
        g_in0_block_num_tiles,
        g_in0_subblock_num_tiles,
        g_in1_num_subblocks,
        g_in1_block_num_tiles,
        g_in1_block_w,
        g_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        gu_out_block_num_tiles,
        /*apply_silu_in_pack=*/true>(cb_in0_x, cb_in1_gate, cb_mm_partials_gu, cb_gate_intermed);

    // =====================================================================
    // PHASE 2 — up matmul: x @ Wu → cb_up_intermed
    // No activation fused; just a vanilla per-core block matmul. We reuse
    // cb_mm_partials_gu for spill (the gate phase has fully drained it by
    // here: matmul_phase pops everything before returning).
    // Reconfig srcA from cb_gate_intermed (where the gate's silu-pack last
    // touched it) back to cb_in0_x.
    // =====================================================================
    mm_block_init_short_with_dt(
        cb_in0_x,
        cb_in1_up,
        /*old_in1_cb_id=*/cb_in1_gate,
        /*transpose=*/0,
        gu_out_subblock_w,
        gu_out_subblock_h,
        u_in0_block_w);

    matmul_phase<
        u_in0_block_w,
        u_in0_num_subblocks,
        u_in0_block_num_tiles,
        u_in0_subblock_num_tiles,
        u_in1_num_subblocks,
        u_in1_block_num_tiles,
        u_in1_block_w,
        u_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        gu_out_block_num_tiles,
        /*apply_silu_in_pack=*/false>(cb_in0_x, cb_in1_up, cb_mm_partials_gu, cb_up_intermed);

    // =====================================================================
    // PHASE 3 — elementwise multiply: cb_gate_intermed * cb_up_intermed →
    // cb_activated. Both buffers are fully populated at this point.
    // mul_tiles_init() handles the srcA/srcB reconfig from matmul mode.
    // =====================================================================
    elementwise_multiply_phase<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
        cb_gate_intermed, cb_up_intermed, cb_activated);

    // =====================================================================
    // PHASE 4 — down matmul: activated @ Wd → cb_out
    // Re-init for matmul. We can't use mm_block_init_short_with_dt here
    // because we're coming OUT of eltwise-binary mode, not matmul mode —
    // the math-engine state and the unpacker AB config both need resetting.
    // mm_block_init does the full HW reconfig (including pack data format
    // for cb_out, which is likely BF16/BFP8 and may differ from the
    // intermediate format we just packed into cb_activated).
    // =====================================================================
    mm_block_init(
        cb_activated,
        cb_in1_down,
        cb_out,
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
        d_in1_block_w,
        d_num_blocks,
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_subblock_num_tiles,
        d_out_block_num_tiles,
        /*apply_silu_in_pack=*/false>(cb_activated, cb_in1_down, cb_mm_partials_d, cb_out);

    // cb_out holds the final per-core y block; the writer kernel drains it
    // into the DRAM-interleaved output.
}
