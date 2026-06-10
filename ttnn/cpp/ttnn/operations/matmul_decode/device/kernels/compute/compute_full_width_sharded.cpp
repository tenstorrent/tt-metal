// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"  // deep-plan_7: copy_tile + copy_tile_to_dst_init_short_with_dt for fp32 c_4 down-cast

using std::uint32_t;

// Full-width-sharded matmul compute: C = A @ B per core.
//
// Inputs (published by reader_full_width_sharded):
//   full_in0_cb (c_3): full gathered A in sender-major K layout
//   in1_cb (c_1):      this core's B shard (buffer-backed, already in L1)
// Output:
//   out_cb (c_2):      this core's output shard (buffer-backed)
//
// Blocking: in0_block_w (K) = 1, out_block_h (M) = 1, out_block_w (N) = 1.
//
// M_tiles>1 fix: the original kernel hardcoded a single M-tile (out_block_h=1,
// in0 index without an M offset, pack always to out slot 0), so it only ever
// computed M-tile 0 and left M-rows 1..M_tiles-1 unwritten/garbage. We now loop
// over every M-tile (bh) x N-tile (bw), accumulating over K, and pack each
// (bh,bw) tile to its sequential output slot.
//
// full_in0 layout (built by reader_full_width_sharded): A is width(K)-sharded
// across `num_senders` sender cores, each holding a contiguous [M_tiles*32,
// inA_K_tiles_per_core*32] slice in TILE row-major order, and the reader copies
// each sender's whole slice into full_in0 at offset sender*shard_num_tiles. So
// full_in0 is SENDER-MAJOR: the tile for global (m, k_global) lives at
//   sender * (M_tiles * inA_K_tiles_per_core)  // sender base
//   + m * inA_K_tiles_per_core                 // M-row within the sender slice
//   + kc_local                                 // K-col within the sender slice
// where sender = k_global / inA_K_tiles_per_core, kc_local = k_global % ...
// (For M_tiles==1 this collapses to k_global, matching the old behaviour.)

using namespace ckernel;
void kernel_main() {
    // matmul_block PER-TILE geometry (DISTINCT from the DST M-block height below).
    // 1x1x1 = single-tile (32x32x32) matmul_block ops. The §6c widening experiment can
    // raise in0_block_w (K-reuse) / out_block_w (N-reuse) via NEW factory CT args [9]/[10]
    // (default 1) to fill the systolic array; default keeps the proven byte-identical path.
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(9);      // NEW (default 1)
    constexpr uint32_t mm_out_block_w = get_compile_time_arg_val(10);  // NEW (default 1)
    // deep-plan_7 §3 STEP-3/§7: rt_dim (DST M-block fill) lifted from a hardcoded 1 to a swept CT
    // arg [11] (default 1 => byte-identical matmul_block geometry). It controls how many M-tiles a
    // single matmul_block call fills into DST (the untried M-blocking systolic-fill lever).
    constexpr uint32_t rt_dim = get_compile_time_arg_val(11);  // NEW (default 1)
    constexpr uint32_t mm_out_block_h = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t last_out_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks_h = get_compile_time_arg_val(5);
    // K-STREAMING (plan_5 s3.5): when stream_k==1 A is delivered one K-slice at a time by
    // reader_full_width_sharded_stream; full_in0 holds only one CB-double-buffered slice of
    // K_slice_tiles columns, and the K-reduction accumulates across slices in DST (forced
    // fp32). When 0, the byte-identical one-shot gather path below is used (gate/up, O, QKV).
    constexpr uint32_t stream_k = get_compile_time_arg_val(6);
    constexpr uint32_t K_slice_tiles = get_compile_time_arg_val(7);
    // NEW CT arg [8] (deep-plan_6 §2.2): DST M-block height. The streamed branch tiles the
    // M loop into num_blocks_h blocks of <= out_block_h M-tiles so a single tile_regs_acquire
    // never holds more than out_block_h*N_tiles_per_core (<=8 = the proven fp32 DST cap) live
    // DST tiles across the K-stream. The factory shrinks out_block_h to max(1,8/Npc) for the
    // streamed Npc>1 (wide-N) case so block_h*Npc stays <=8.
    constexpr uint32_t out_block_h = get_compile_time_arg_val(8);

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t num_K_blocks = K_tiles / in0_block_w;
    // tiles per sender slice in full_in0 (== reader's shard_num_tiles)
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // deep-plan_7 §3 STEP-3 init: pass rt_dim so the unpacker/math are configured for the M-block
    // geometry actually used by the streamed K-OUTER-once path (rt_dim default 1 == iter-6 geometry).
    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, mm_out_block_w, rt_dim, in0_block_w);

    if constexpr (stream_k == 0) {
        // ---- Non-streamed: one-shot gather (full_in0 holds the WHOLE A, sender-major). ----
        constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;
        cb_wait_front(in0_cb_id, in0_num_tiles);
        cb_reserve_back(out_cb_id, M_tiles * N_tiles_per_core);
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < num_K_blocks; ++k) {
                    const uint32_t sender = k / inA_K_tiles_per_core;
                    const uint32_t kc_local = k - sender * inA_K_tiles_per_core;
                    const uint32_t in0_tile = sender * sender_slice_tiles + mt * inA_K_tiles_per_core + kc_local;
                    matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_tile,
                        k * N_tiles_per_core + bw,
                        0,
                        false,
                        mm_out_block_w,
                        mm_out_block_h,
                        in0_block_w);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb_id, mt * N_tiles_per_core + bw);
                tile_regs_release();
            }
        }
        cb_push_back(out_cb_id, M_tiles * N_tiles_per_core);
        cb_pop_front(in0_cb_id, in0_num_tiles);
    } else {
        // ================= deep-plan_7 STEP-3: K-slice OUTER-once / M-block+N INNER ============
        // iter-6 was M-block OUTER / K-slice INNER with cb_wait_front/pop_front(in0_cb) INSIDE the
        // mb loop, so K was RE-STREAMED num_blocks_h times (gate/up 9x, down 2x = the ~6110us
        // artifact). iter-7 INVERTS: the K-slice is the OUTER-once loop (each slice read/mcast
        // EXACTLY once, num_steps total), iterating ALL M-blocks (+ N-tiles) against each resident
        // slice INNER. Running partials live in a per-core fp32 L1 accumulator CB (c_4) that the
        // PACKER accumulates into in place (pack_reconfig_l1_acc(0/1)); DST holds only the <=8 live
        // transient tiles of one acquire (micro-tiling). After K completes, a single pass down-casts
        // the fp32 acc -> bf16 out_cb (c_2, the iter-6 buffer-backed contract, UNCHANGED).
        constexpr uint32_t num_steps = K_tiles / K_slice_tiles;
        constexpr uint32_t slice_tiles = M_tiles * K_slice_tiles;
        constexpr uint32_t num_out_tiles = M_tiles * N_tiles_per_core;
        constexpr uint32_t acc_cb_id = tt::CBIndex::c_4;  // NEW fp32 L1 accumulator (deep-plan_7)

        if (num_blocks_h == 1) {
            // ---- BYTE-IDENTICAL iter-6 single-acquire K-inner fast-path (M_tiles<=8: SigLIP M=8,
            //      every M<=96 regression shape). Reader streams K once (gstep==s) => K consumed
            //      once here too. PACKER_L1_ACC + acc_cb do NOT run for num_blocks_h==1. ----
            cb_reserve_back(out_cb_id, num_out_tiles);
            tile_regs_acquire();  // <= M_tiles*N_tiles_per_core (<=8) live DST tiles
            for (uint32_t s = 0; s < num_steps; ++s) {
                cb_wait_front(in0_cb_id, slice_tiles);  // slice s holds ALL M_tiles rows
                for (uint32_t bm = 0; bm < M_tiles; ++bm) {
                    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                        const uint32_t dst_idx = bm * N_tiles_per_core + bw;  // BLOCK-LOCAL DST slot
                        for (uint32_t kc = 0; kc < K_slice_tiles; kc += in0_block_w) {
                            const uint32_t k_global = s * K_slice_tiles + kc;
                            matmul_block(
                                in0_cb_id,
                                in1_cb_id,
                                bm * K_slice_tiles + kc,           // GLOBAL slice index (single sender)
                                k_global * N_tiles_per_core + bw,  // resident full-K B (global)
                                dst_idx,
                                false,
                                mm_out_block_w,
                                mm_out_block_h,
                                in0_block_w);
                        }
                    }
                }
                cb_pop_front(in0_cb_id, slice_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t bm = 0; bm < M_tiles; ++bm) {
                for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                    pack_tile<true>(bm * N_tiles_per_core + bw, out_cb_id, bm * N_tiles_per_core + bw);
                }
            }
            tile_regs_release();
            cb_push_back(out_cb_id, num_out_tiles);
        } else {
            // ---- iter-7 K-OUTER-ONCE + PACKER_L1_ACC into fp32 c_4 (num_blocks_h>=2: VLM M=288) ----
            cb_reserve_back(acc_cb_id, num_out_tiles);  // reserve fp32 accumulator region ONCE
            for (uint32_t s = 0; s < num_steps; ++s) {  // <-- K-slice OUTER, run ONCE (THE re-stream fix)
                cb_wait_front(in0_cb_id, slice_tiles);  // slice s = [M_tiles, K_slice_tiles]
                const bool first_slice = (s == 0);
                // Pack into the fp32 acc CB; overwrite on first slice, accumulate-in-place after.
                pack_reconfig_data_format(acc_cb_id);             // fp32 partials format (mirrors native)
                pack_reconfig_l1_acc(first_slice ? 0 : 1);        // s==0 overwrite; s>0 accumulate in L1
                for (uint32_t mb = 0; mb < num_blocks_h; ++mb) {  // <-- M-block INNER
                    const uint32_t block_h = (mb == num_blocks_h - 1) ? last_out_block_h : out_block_h;
                    const uint32_t mt0 = mb * out_block_h;
                    tile_regs_acquire();  // DST zero on acquire; <= block_h*Npc (<=8) live tiles
                    for (uint32_t bm = 0; bm < block_h; bm += rt_dim) {
                        const uint32_t rt = (rt_dim < (block_h - bm)) ? rt_dim : (block_h - bm);  // tail-safe
                        for (uint32_t bw = 0; bw < N_tiles_per_core; bw += mm_out_block_w) {
                            const uint32_t ct =
                                (mm_out_block_w < (N_tiles_per_core - bw)) ? mm_out_block_w : (N_tiles_per_core - bw);
                            const uint32_t dst_idx = bm * N_tiles_per_core + bw;  // BLOCK-LOCAL (<=7)
                            for (uint32_t kc = 0; kc < K_slice_tiles; kc += in0_block_w) {
                                const uint32_t k_global = s * K_slice_tiles + kc;
                                const uint32_t mt = mt0 + bm;  // GLOBAL M-tile
                                matmul_block(
                                    in0_cb_id,
                                    in1_cb_id,
                                    mt * K_slice_tiles + kc,           // GLOBAL A slice index
                                    k_global * N_tiles_per_core + bw,  // resident full-K B
                                    dst_idx,
                                    false,  // transpose flag (NOT accumulate; matmul_block always DST+=C)
                                    ct,
                                    rt,
                                    in0_block_w);  // ct_dim, rt_dim, kt_dim
                            }
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bm = 0; bm < block_h; ++bm) {
                        for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                            pack_tile<true>(
                                bm * N_tiles_per_core + bw,
                                acc_cb_id,
                                (mt0 + bm) * N_tiles_per_core + bw);  // absolute L1 slot
                        }
                    }
                    tile_regs_release();
                }
                cb_pop_front(in0_cb_id, slice_tiles);  // <-- K-slice consumed ONCE
            }
            // ---- final: down-cast the completed fp32 L1 partials -> bf16 out_cb (single pass) ----
            pack_reconfig_l1_acc(0);                                    // stop accumulating
            pack_reconfig_data_format(out_cb_id);                       // bf16 output format (native pattern)
            copy_tile_to_dst_init_short_with_dt(in1_cb_id, acc_cb_id);  // unpack src -> acc_cb (fp32)
            cb_reserve_back(out_cb_id, num_out_tiles);
            for (uint32_t mb = 0; mb < num_blocks_h; ++mb) {
                const uint32_t block_h = (mb == num_blocks_h - 1) ? last_out_block_h : out_block_h;
                const uint32_t mt0 = mb * out_block_h;
                tile_regs_acquire();
                for (uint32_t bm = 0; bm < block_h; ++bm) {
                    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                        copy_tile(acc_cb_id, (mt0 + bm) * N_tiles_per_core + bw, bm * N_tiles_per_core + bw);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t bm = 0; bm < block_h; ++bm) {
                    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
                        pack_tile<true>(bm * N_tiles_per_core + bw, out_cb_id, (mt0 + bm) * N_tiles_per_core + bw);
                    }
                }
                tile_regs_release();
            }
            cb_push_back(acc_cb_id, num_out_tiles);  // release acc region once at the very end
            cb_push_back(out_cb_id, num_out_tiles);  // out fully written
        }
    }
}
