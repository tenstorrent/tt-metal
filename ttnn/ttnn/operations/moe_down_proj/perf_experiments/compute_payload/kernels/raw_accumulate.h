// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf experiment: raw-LLK accumulate_round for moe_down_proj (bypasses compute_kernel_lib::matmul_block).
// Same math and pack order as the helper path (per round, per expert sub-block: acquire, k_block matmul_block
// LLK calls into DEST, pack TileRowMajor with packer L1 accumulation), so output is bit-identical. What the
// helper does per call that this path hoists:
//   - pack_reconfig_data_format(partials) + llk_pack_reconfig_l1_acc per subblock pack (both stall the packer);
//     here: format once per wave, l1_acc once per round.
//   - in0 wait/pop per call (WaitAndPopPerKBlock); here: indexed reads, one pop per round.
// Knobs:
//   CP_PER_SB_RECONFIG  restore the helper's per-subblock pack reconfig (for attribution).
//   CP_PAD_TAIL         compute a short tail with the preferred subblock height (junk rows computed, not packed)
//                       instead of a matmul_block_init for the tail shape (two re-inits per tail).
//   CP_SPLIT_RT         one DEST acquire per preferred (h x w) subblock, issued as h single-row (1 x w)
//                       matmul_block calls per k step; init is always 1 x w, so tails need no re-init.
//   CP_NO_PACK          diagnostic: skip pack_tile (output wrong).
//   CP_REPEAT=N         diagnostic: run each round's matmul work N times (output wrong for N > 1).
//   CP_KSPAN=N          diagnostic: run the k_block matmul sequence N times per DEST acquire (models a K span of
//                       N * k_block per pack; output wrong for N > 1).
//   CP_PACK_ROWS=N      diagnostic: pack only the first N subblock rows (output wrong).
//   CP_DST_OFF=N        diagnostic: DEST tile base of the subblock (default 0).
//   CP_NO_L1ACC         diagnostic: pack without L1 accumulation (output wrong).
//   SKIP_COMPUTE        diagnostic: skip the matmul LLK calls (pack only).

#pragma once

#ifndef CP_REPEAT
#define CP_REPEAT 1
#endif
#ifndef CP_DST_OFF
#define CP_DST_OFF 0
#endif
#ifndef CP_KSPAN
#define CP_KSPAN 1
#endif

template <
    uint32_t k_block_tiles,
    uint32_t num_k_rounds,
    uint32_t core_n_tiles,
    uint32_t core_row_tiles_max,
    uint32_t dest_limit,
    typename Buf>
ALWI void raw_accumulate_wave(
    Buf& x_buf,
    Buf& w_buf,
    const MoePlan& plan,
    uint32_t row_lo,
    uint32_t row_hi,
    uint32_t my_rows,
    uint32_t last_slot,
    Subblock preferred,
    uint32_t cb_x,
    uint32_t cb_w,
    uint32_t cb_partials) {
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;

    // First matmul of the wave follows the untilize (or boot): reconfigure both sides once.
    ckernel::reconfig_data_format(cb_w, cb_x);
    ckernel::pack_reconfig_data_format(cb_partials);
    Subblock init_sb{0, 0};

    MaybeDeviceZoneScope("compute_accumulate");
    for (uint32_t round = 0; round < num_k_rounds; ++round) {
        if (my_rows > 0) {
            MaybeDeviceZoneScope("compute_wait_inputs");
            x_buf.wait_front(my_rows * k_block_tiles);
            w_buf.wait_front(last_slot * w_chunk_tiles);
        }
        for (uint32_t rep = 0; rep < CP_REPEAT; ++rep) {
#ifdef CP_NO_L1ACC
            const uint32_t l1_acc = 0;
#else
            const uint32_t l1_acc = (round > 0 || rep > 0) ? 1 : 0;
#endif
#ifndef CP_PER_SB_RECONFIG
            pack_reconfig_l1_acc(l1_acc);
#endif
            uint32_t x_off = 0;
            for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
                uint32_t first;
                const uint32_t rows = plan.expert_rows_in(e, row_lo, row_hi, &first);
                if (rows == 0) {
                    continue;
                }
                const uint32_t w_base = e * w_chunk_tiles;
                const uint32_t main_rows = rows - rows % preferred.h;
                uint32_t call_r0 = first - row_lo;
                for (uint32_t part = 0; part < 2; ++part) {
                    const uint32_t call_rows = part == 0 ? main_rows : rows - main_rows;
                    if (call_rows == 0) {
                        continue;
                    }
#if defined(CP_SPLIT_RT)
                    const Subblock sb = preferred;
                    const uint32_t num_rb = (call_rows + sb.h - 1) / sb.h;
#elif defined(CP_PAD_TAIL)
                    const Subblock sb = preferred;
                    const uint32_t num_rb = (call_rows + sb.h - 1) / sb.h;
#else
                    const Subblock sb = part == 0 ? preferred : best_subblock(call_rows, core_n_tiles, dest_limit);
                    const uint32_t num_rb = call_rows / sb.h;
#endif
#ifdef CP_SPLIT_RT
                    const Subblock call_sb{1, sb.w};
#else
                    const Subblock call_sb = sb;
#endif
                    if (call_sb.h != init_sb.h || call_sb.w != init_sb.w) {
                        ckernel::matmul_block_init(cb_x, cb_w, true, call_sb.w, call_sb.h, k_block_tiles);
                        init_sb = call_sb;
                    }
                    for (uint32_t rb = 0; rb < num_rb; ++rb) {
                        const uint32_t rb_r0 = rb * sb.h;
                        const uint32_t pack_h = call_rows - rb_r0 < sb.h ? call_rows - rb_r0 : sb.h;
                        for (uint32_t cb_col = 0; cb_col < core_n_tiles; cb_col += sb.w) {
                            tile_regs_acquire();
#ifndef SKIP_COMPUTE
                            const uint32_t in0 = x_off + rb_r0 * k_block_tiles;
                            const uint32_t in1 = w_base + cb_col;
#ifdef CP_SPLIT_RT
                            for (uint32_t r = 0; r < pack_h; ++r) {
                                for (uint32_t kk = 0; kk < k_block_tiles * CP_KSPAN; ++kk) {
                                    ckernel::matmul_block(
                                        cb_x,
                                        cb_w,
                                        in0 + r * k_block_tiles + kk % k_block_tiles,
                                        in1 + (kk % k_block_tiles) * core_n_tiles,
                                        CP_DST_OFF + r * sb.w,
                                        true,
                                        sb.w,
                                        1,
                                        k_block_tiles);
                                }
                            }
#else
                            for (uint32_t kk = 0; kk < k_block_tiles * CP_KSPAN; ++kk) {
                                ckernel::matmul_block(
                                    cb_x,
                                    cb_w,
                                    in0 + kk % k_block_tiles,
                                    in1 + (kk % k_block_tiles) * core_n_tiles,
                                    CP_DST_OFF,
                                    true,
                                    sb.w,
                                    sb.h,
                                    k_block_tiles);
                            }
#endif
#endif
                            tile_regs_commit();
                            tile_regs_wait();
#ifdef CP_PER_SB_RECONFIG
                            PACK((pack_reconfig_data_format(cb_partials)));
                            PACK((llk_pack_reconfig_l1_acc(l1_acc)));
#endif
#ifndef CP_NO_PACK
                            const uint32_t out_base = (call_r0 + rb_r0) * core_n_tiles + cb_col;
#ifdef CP_PACK_ROWS
                            for (uint32_t r = 0; r < CP_PACK_ROWS; ++r) {
#else
                            for (uint32_t r = 0; r < pack_h; ++r) {
#endif
                                for (uint32_t c = 0; c < sb.w; ++c) {
                                    pack_tile<true>(
                                        CP_DST_OFF + r * sb.w + c, cb_partials, out_base + r * core_n_tiles + c);
                                }
                            }
#endif
                            tile_regs_release();
                        }
                    }
                    x_off += call_rows * k_block_tiles;
                    call_r0 += call_rows;
                }
            }
        }
        x_buf.wait_front(x_round_tiles);
        x_buf.pop_front(x_round_tiles);
        w_buf.wait_front(MOE_NUM_EXPERTS * w_chunk_tiles);
        w_buf.pop_front(MOE_NUM_EXPERTS * w_chunk_tiles);
    }
}
