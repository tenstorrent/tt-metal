// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v4) compute: batched per-row online-softmax visits. Resident rows keep Q, an
// O accumulator, and PING-PONG running max/sum slots in RAM-mode CBs. The reader windows the block
// stream (all slots of a window are freed together after its visits), so each visit is one row's
// batch of n windowed blocks and the flash machinery -- max reduce, corr, O/sum rescale, syncs --
// runs ONCE per batch instead of once per block:
//
//   1. QK    per block: two no-MOP column chains into the row's [Sqt x n*Skt] qk scratch (+ masks)
//   2. MAX   one runtime-width reduce over all n*Skt columns, eltwise-max with the old slot
//   3. CORR  corr = exp((old_max - new_max) * scale) (skipped on the row's first visit)
//   4. RESCALE O *= corr in place; old_sum * corr -> the NEW sum slot
//   5. EXP   probs = exp((qk - new_max) * scale) in place, col-batched; row sums L1-acc the NEW sum
//   6. PV    one acquire; per (col, block) no-MOP chains accumulate O (packer L1-acc, or overwrite)
//
// Ping-pong state comes from the reader's parity bit; MSG_WINDOW returns a window's slot credits.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#if defined(VSA_PROBE) && VSA_PROBE == 9
#include "api/debug/dprint.h"
#define VSA_TICK() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#endif
#include "api/compute/tile_move_copy.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

// Stamp a mask tile (L1-acc add of 0/-inf columns) onto absolute qk tile indices; the shared
// apply_partial_mask_lightweight wants a row base in row-tile units, which a ping-pong region
// base that is not a multiple of the runtime row width cannot express.
ALWI void vsa_stamp_mask(uint32_t mask_cb, uint32_t mask_idx, uint32_t out_cb, uint32_t tile0, uint32_t stride, uint32_t rows) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        copy_tile(mask_cb, mask_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, tile0 + row * stride);
        tile_regs_release();
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}

ALWI void stream_pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

constexpr uint32_t MSG_VISIT = 0;   // {type | n_blocks<<16, rowinfo, (slot | count<<8 | vmask<<15) x n}
constexpr uint32_t MSG_FLUSH = 1;   // {type, row_slot, parity}
constexpr uint32_t MSG_WINDOW = 2;  // {type, n_slots}: return the window's stream credits
constexpr uint32_t ROW_PARITY = 1u << 8;
constexpr uint32_t ROW_IS_FIRST = 1u << 9;

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t R_MAX = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);  // rows per phase group
    constexpr uint32_t block_size = get_compile_time_arg_val(6);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(7);

    constexpr uint32_t cb_q_res = get_compile_time_arg_val(8);
    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(9);
    constexpr uint32_t cb_v_stream = get_compile_time_arg_val(10);
    constexpr uint32_t cb_o_res = get_compile_time_arg_val(11);
    constexpr uint32_t cb_max_res = get_compile_time_arg_val(12);  // R_MAX x 2 x Sqt (ping-pong)
    constexpr uint32_t cb_sum_res = get_compile_time_arg_val(13);  // R_MAX x 2 x Sqt (ping-pong)
    constexpr uint32_t cb_corr = get_compile_time_arg_val(14);     // G x Sqt scratch
    constexpr uint32_t cb_qk = get_compile_time_arg_val(15);       // G x Sqt x Skt scratch
    constexpr uint32_t cb_scale = get_compile_time_arg_val(16);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(17);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(18);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(19);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(20);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(21);
    constexpr uint32_t cb_free = get_compile_time_arg_val(22);
    constexpr uint32_t cb_qdone = get_compile_time_arg_val(23);
    constexpr uint32_t cb_out = get_compile_time_arg_val(24);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(25);

    constexpr uint32_t k_tiles_per_block = Skt * DHt;
    constexpr uint32_t v_tiles_per_block = Skt * vDHt;
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    const uint32_t row_count = get_arg_val<uint32_t>(0);
    if (row_count == 0) {
        return;  // leader core (or a worker with no rows): its writer builds no persistent tiles
    }

    CircularBuffer ctrl_cb(cb_ctrl), free_cb(cb_free), qdone_cb(cb_qdone), out_cb(cb_out);
    CircularBuffer qk_cb(cb_qk), corr_cb(cb_corr);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_res, cb_k_stream, cb_qk);
    matmul_init(cb_q_res, cb_k_stream);

    // RAM-mode CBs: reserve capacity once, hold forever, index absolutely.
    const auto ram = [](uint32_t cb, uint32_t tiles) {
        CircularBuffer(cb).reserve_back(tiles);
        cb_push_back_hold_wr_ptr(cb, tiles);
    };
    ram(cb_max_res, R_MAX * 2 * Sqt);
    ram(cb_sum_res, R_MAX * 2 * Sqt);
    ram(cb_o_res, R_MAX * Sqt * vDHt);
    ram(cb_qk, G * Sqt * Skt);
    ram(cb_corr, G * Sqt);
    ram(cb_k_stream, stream_depth * k_tiles_per_block);
    ram(cb_v_stream, stream_depth * v_tiles_per_block);
    ram(cb_q_res, R_MAX * Sqt * DHt);
    ram(cb_vmask, stream_depth);  // slot-indexed ragged masks, freed with their window

    CircularBuffer(cb_scale).wait_front(1);
    CircularBuffer(cb_col_identity).wait_front(1);
    CircularBuffer(cb_neginf).wait_front(1);


#if defined(VSA_PROBE) && VSA_PROBE == 9
    // MATH-thread phase timers: elapsed wall clock between phase boundaries as seen by TRISC1.
    uint32_t t_wait = 0, t_qk = 0, t_max = 0, t_corr = 0, t_pv = 0, t_exp = 0, t_flush = 0;
    uint32_t n_visits = 0;
    const uint32_t t_begin = VSA_TICK();
    uint32_t tmark = t_begin;
    const auto lap = [&](uint32_t& acc) {
        const uint32_t now = VSA_TICK();
        acc += now - tmark;
        tmark = now;
    };
#else
    const auto lap = [](...) {};
    uint32_t t_wait = 0, t_qk = 0, t_max = 0, t_corr = 0, t_pv = 0, t_exp = 0, t_flush = 0;
    (void)t_wait; (void)t_qk; (void)t_max; (void)t_corr; (void)t_pv; (void)t_exp; (void)t_flush;
#endif

    // GROUP-MAJOR window engine: a window's visits are buffered and processed phase-major --
    // QK for every visit, ONE s1 sync, maxes batched four visits per DEST acquire, ONE s2 sync,
    // corr batched, ONE s3 sync, rescales, then exp -- instead of paying every sync and DEST
    // round-trip per visit (measured: ~2600 cycles/visit of non-math at ~4600 cycles/visit).
    // A window is processed in CHUNKS whose qk tiles fit one region of the double-buffered qk
    // scratch; each chunk's PV is DEFERRED to the next chunk (or flush), overlapping its
    // pack-thread exp with the next chunk's math, and the window's slot credits are stashed
    // until the deferred PV has consumed its V tiles.
    constexpr uint32_t kChunkCols = (stream_depth / 2) * Skt;         // qk region width in tiles
    constexpr uint32_t kRegionTiles = kChunkCols * Sqt;               // qk region size in tiles
    constexpr uint32_t kMaxVisits = 16;                               // >= rows per pass
    struct Visit {
        uint32_t row_slot;
        uint32_t flags;  // ROW_IS_FIRST / ROW_PARITY as sent by the reader
        uint32_t n;
        uint32_t tile_base;  // dense sub-block base within the qk region (tiles)
        uint32_t entries[stream_depth / 2 > 0 ? stream_depth / 2 : 1];
    };
    Visit vbuf[kMaxVisits];
    uint32_t vn = 0;

    // Deferred chunk: its probs live in the OTHER qk region; PV runs at the next chunk (or flush).
    Visit pend_v[stream_depth / 2 > 0 ? stream_depth / 2 : 1];
    uint32_t pend_n = 0;
    uint32_t pend_qk_base = 0;
    uint32_t pend_credits = 0;  // window credits held until the deferred PV consumed its V slots
    uint32_t qk_region = 0;

    const auto drain_pend = [&]() {
        if (pend_n == 0) {
            if (pend_credits != 0) {
                free_cb.reserve_back(pend_credits);
                free_cb.push_back(pend_credits);
                pend_credits = 0;
            }
            return;
        }
        stream_pack_to_unpack_sync();  // the chunk's probs must be visible (usually free by now)
        reconfig_data_format(cb_v_stream, cb_qk);
        for (uint32_t i = 0; i < pend_n; ++i) {
            const Visit& v = pend_v[i];
            const uint32_t qk_cols = v.n * Skt;
            mm_no_mop_init_short(cb_qk, cb_v_stream, /*transpose=*/false, 1, Sqt, qk_cols);
            tile_regs_acquire();
            for (uint32_t vd = 0; vd < vDHt; ++vd) {
                for (uint32_t b = 0; b < v.n; ++b) {
                    const uint32_t slot = v.entries[b] & 0xff;
                    for (uint32_t inner = 0; inner < Skt; ++inner) {
                        matmul_block_no_mop(
                            cb_qk,
                            cb_v_stream,
                            /*in0=*/pend_qk_base + v.tile_base + b * Skt + inner,
                            /*in1=*/slot * v_tiles_per_block + inner * vDHt + vd,
                            /*dst=*/vd * Sqt,
                            /*transpose=*/false,
                            /*w=*/1,
                            /*h=*/Sqt,
                            /*stride=*/qk_cols);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            const bool is_first = (v.flags & ROW_IS_FIRST) != 0;
            const uint32_t o_base_p = v.row_slot * Sqt * vDHt;
            configure_row_pack_width(cb_o_res, 1);
            PACK((llk_pack_reconfig_l1_acc(is_first ? 0 : 1)));
            for (uint32_t vd = 0; vd < vDHt; ++vd) {
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(vd * Sqt + sr, cb_o_res, o_base_p + sr * vDHt + vd);
                }
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            tile_regs_release();
        }
        pend_n = 0;
        if (pend_credits != 0) {
            free_cb.reserve_back(pend_credits);
            free_cb.push_back(pend_credits);
            pend_credits = 0;
        }
    };

    // Process one chunk of visits (sum of n over the chunk fits a qk region).
    const auto process_chunk = [&](Visit* vs, uint32_t nv) {
        const uint32_t qk_base = qk_region * kRegionTiles;
        {
            uint32_t off = 0;
            for (uint32_t i = 0; i < nv; ++i) {
                vs[i].tile_base = off;
                off += Sqt * vs[i].n * Skt;
            }
        }

        // Phase 1: QK for every visit (two blocks per DEST acquire), masks, ONE s1 sync.
        reconfig_data_format(cb_k_stream, cb_q_res);
        mm_no_mop_init_short(cb_q_res, cb_k_stream, /*transpose=*/true, 1, Sqt, DHt);
        pack_reconfig_data_format(cb_qk);
        configure_row_pack_width(cb_qk, 1);
        for (uint32_t i = 0; i < nv; ++i) {
            const Visit& v = vs[i];
            const uint32_t qk_cols = v.n * Skt;
            for (uint32_t b = 0; b < v.n; b += 2) {
                const uint32_t nb = (v.n - b < 2) ? (v.n - b) : 2;
                tile_regs_acquire();
                for (uint32_t j = 0; j < nb; ++j) {
                    const uint32_t slot = v.entries[b + j] & 0xff;
                    for (uint32_t c = 0; c < Skt; ++c) {
                        for (uint32_t inner = 0; inner < DHt; ++inner) {
                            matmul_block_no_mop(
                                cb_q_res, cb_k_stream, v.row_slot * Sqt * DHt + inner,
                                slot * k_tiles_per_block + c * DHt + inner, j * Skt * Sqt + c * Sqt,
                                /*transpose=*/true,
                                /*w=*/1, /*h=*/Sqt, /*stride=*/DHt);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t j = 0; j < nb; ++j) {
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        for (uint32_t c = 0; c < Skt; ++c) {
                            pack_tile<true>(
                                j * Skt * Sqt + c * Sqt + sr, cb_qk,
                                qk_base + v.tile_base + sr * qk_cols + (b + j) * Skt + c);
                        }
                    }
                }
                tile_regs_release();
            }
        }
        for (uint32_t i = 0; i < nv; ++i) {
            const Visit& v = vs[i];
            const uint32_t qk_cols = v.n * Skt;
            for (uint32_t b = 0; b < v.n; ++b) {
                const uint32_t count = (v.entries[b] >> 8) & 0x7f;
                if (count >= block_size) {
                    continue;
                }
                const uint32_t slot = v.entries[b] & 0xff;
                const uint32_t btile = count / keys_per_tile;
                const uint32_t bcol = count % keys_per_tile;
                uint32_t first_full = btile;
                if (bcol != 0) {
                    vsa_stamp_mask(cb_vmask, slot, cb_qk, qk_base + v.tile_base + b * Skt + btile, qk_cols, Sqt);
                    first_full = btile + 1;
                }
                for (uint32_t kt = first_full; kt < Skt; ++kt) {
                    vsa_stamp_mask(cb_neginf, 0, cb_qk, qk_base + v.tile_base + b * Skt + kt, qk_cols, Sqt);
                }
            }
        }
        stream_pack_to_unpack_sync();  // s1: every visit's qk (+masks) visible
        lap(t_qk);

        // Phase 2: running maxes, FOUR visits per DEST acquire, ONE s2 sync.
        reconfig_data_format(cb_qk, cb_scale);
        for (uint32_t g = 0; g < nv; g += 4) {
            const uint32_t ng = (nv - g < 4) ? (nv - g) : 4;
            tile_regs_acquire();
            for (uint32_t i = 0; i < ng; ++i) {
                const Visit& v = vs[g + i];
                const uint32_t parity = (v.flags & ROW_PARITY) ? 1u : 0u;
                const uint32_t old_st = (v.row_slot * 2 + (parity ^ 1u)) * Sqt;
                if (!(v.flags & ROW_IS_FIRST)) {
                    sdpa_reduce_copy_tile_to_dst_init_short(cb_max_res);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        copy_tile(cb_max_res, old_st + sr, i * Sqt + sr);
                    }
                }
                const uint32_t qk_cols = v.n * Skt;
                reduce_block_max_row_init_runtime(cb_max_res, qk_cols, cb_qk, cb_scale, false);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    reduce_block_max_row_runtime(
                        cb_qk, cb_scale, qk_base + v.tile_base + sr * qk_cols, i * Sqt + sr, false, false);
                }
                reduce_block_max_row_uninit_runtime(cb_qk, false, false);
            }
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(cb_max_res);
            PACK((llk_pack_reconfig_l1_acc(0)));
            for (uint32_t i = 0; i < ng; ++i) {
                const Visit& v = vs[g + i];
                const uint32_t parity = (v.flags & ROW_PARITY) ? 1u : 0u;
                const uint32_t new_st = (v.row_slot * 2 + parity) * Sqt;
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(i * Sqt + sr, cb_max_res, new_st + sr);
                }
            }
            tile_regs_release();
        }
        stream_pack_to_unpack_sync();  // s2: new maxes visible
        lap(t_max);

        // Phase 3: corr for every non-first visit, FOUR per acquire, ONE s3 sync (if any).
        bool any_corr = false;
        {
            uint32_t done = 0;
            exp_packthread_tile_init<EXP_APPROX_MODE>();
            sub_init(cb_max_res, cb_max_res);
            while (done < nv) {
                uint32_t taken = 0;
                tile_regs_acquire();
                uint32_t members[4];
                while (done < nv && taken < 4) {
                    const Visit& v = vs[done];
                    if (v.flags & ROW_IS_FIRST) {
                        ++done;
                        continue;
                    }
                    const uint32_t parity = (v.flags & ROW_PARITY) ? 1u : 0u;
                    const uint32_t new_st = (v.row_slot * 2 + parity) * Sqt;
                    const uint32_t old_st = (v.row_slot * 2 + (parity ^ 1u)) * Sqt;
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        sub_tiles(cb_max_res, cb_max_res, old_st + sr, new_st + sr, taken * Sqt + sr);
                    }
                    members[taken] = done;
                    ++taken;
                    ++done;
                }
                tile_regs_commit();
                tile_regs_wait();
                if (taken > 0) {
                    any_corr = true;
                    for (uint32_t i = 0; i < taken * Sqt; ++i) {
                        PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(i)));
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    configure_single_tile_pack(cb_corr);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t i = 0; i < taken; ++i) {
                        for (uint32_t sr = 0; sr < Sqt; ++sr) {
                            pack_tile<true>(i * Sqt + sr, cb_corr, members[i] * Sqt + sr);
                        }
                    }
                }
                tile_regs_release();
            }
        }
        if (any_corr) {
            stream_pack_to_unpack_sync();  // s3: corrs visible
        }

        // The PREVIOUS chunk's deferred PV: its exp/packs overlapped every phase above. It must
        // land (and be made visible) before rescale multiplies any O it accumulated into.
        drain_pend();
        stream_pack_to_unpack_sync();
        lap(t_pv);

        // Phase 4: rescale each non-first visit's O and sum by its corr.
        if (any_corr) {
            mul_bcast_cols_init(cb_o_res, cb_corr);
            reconfig_data_format(cb_o_res, cb_corr);
            for (uint32_t i = 0; i < nv; ++i) {
                const Visit& v = vs[i];
                if (v.flags & ROW_IS_FIRST) {
                    continue;
                }
                const uint32_t parity = (v.flags & ROW_PARITY) ? 1u : 0u;
                const uint32_t new_st = (v.row_slot * 2 + parity) * Sqt;
                const uint32_t old_st = (v.row_slot * 2 + (parity ^ 1u)) * Sqt;
                const uint32_t o_base = v.row_slot * Sqt * vDHt;
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        mul_tiles_bcast_cols(cb_o_res, cb_corr, o_base + sr * vDHt + j, i * Sqt + sr, j);
                    }
                    mul_tiles_bcast_cols(cb_sum_res, cb_corr, old_st + sr, i * Sqt + sr, vDHt);
                    tile_regs_commit();
                    tile_regs_wait();
                    const bool blocked_o = configure_row_pack_width(cb_o_res, vDHt);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    if (blocked_o) {
                        sdpa_pack_tile_ooo(0, cb_o_res, o_base + sr * vDHt);
                    } else {
                        for (uint32_t j = 0; j < vDHt; ++j) {
                            pack_tile<true>(j, cb_o_res, o_base + sr * vDHt + j);
                        }
                    }
                    configure_single_tile_pack(cb_sum_res);
                    pack_tile<true>(vDHt, cb_sum_res, new_st + sr);
                    tile_regs_release();
                }
            }
        }
        lap(t_corr);

        // Phase 5: probs = exp((qk - new_max) * scale) in place per visit, in DEST-sized column
        // batches; row sums L1-acc onto the NEW sum slot (overwrite on a first visit's first
        // batch).
        exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
        for (uint32_t i = 0; i < nv; ++i) {
            const Visit& v = vs[i];
            const bool is_first = (v.flags & ROW_IS_FIRST) != 0;
            const uint32_t parity = (v.flags & ROW_PARITY) ? 1u : 0u;
            const uint32_t new_st = (v.row_slot * 2 + parity) * Sqt;
            const uint32_t qk_cols = v.n * Skt;
            for (uint32_t kc = 0; kc < qk_cols; kc += 4) {
                const uint32_t w = (qk_cols - kc < 4) ? (qk_cols - kc) : 4;
                sub_bcast_cols_init_short_custom(cb_qk, cb_max_res, w);
                reconfig_data_format(cb_qk, cb_max_res);
                tile_regs_acquire();
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    sub_tiles_bcast_cols_custom(
                        cb_qk, cb_max_res, qk_base + v.tile_base + sr * qk_cols + kc, new_st + sr, sr * w, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_relu_config(ReluConfig::zero())));
                for (uint32_t t = 0; t < Sqt * w; ++t) {
                    exp_packthread_tile<true, false, InputClamping::None, 32>(t, VectorMode::None);
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                const bool blocked_p = configure_row_pack_width(cb_qk, w);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    if (blocked_p) {
                        sdpa_pack_tile_ooo(sr * w, cb_qk, qk_base + v.tile_base + sr * qk_cols + kc);
                    } else {
                        for (uint32_t c = 0; c < w; ++c) {
                            pack_tile<true>(sr * w + c, cb_qk, qk_base + v.tile_base + sr * qk_cols + kc + c);
                        }
                    }
                }
                configure_single_tile_pack(cb_sum_res);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    for (uint32_t c = 0; c < w; ++c) {
                        const bool overwrite = is_first && kc == 0 && c == 0;
                        PACK((llk_pack_reconfig_l1_acc(overwrite ? 0 : 1)));
                        pack_tile<true>(sr * w + c, cb_sum_res, new_st + sr);
                    }
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
                PACK((llk_pack_relu_config(ReluConfig::none())));
                tile_regs_release();
            }
        }
        lap(t_exp);

        // Defer this chunk's PV: its pack-thread exp overlaps the NEXT chunk's math.
        for (uint32_t i = 0; i < nv; ++i) {
            pend_v[i] = vs[i];
        }
        pend_n = nv;
        pend_qk_base = qk_base;
        qk_region ^= 1u;
#if defined(VSA_PROBE) && VSA_PROBE == 9
        n_visits += nv;
#endif
    };

    uint32_t rows_done = 0;
    while (rows_done < row_count) {
        const uint32_t pass_rows = (row_count - rows_done < R_MAX) ? (row_count - rows_done) : R_MAX;
        qdone_cb.wait_front(1);  // pass Q resident
        qdone_cb.pop_front(1);

        uint32_t flushed = 0;
        while (flushed < pass_rows) {
            lap(t_wait);
            ctrl_cb.wait_front(1);
            const uint32_t w0 = ckernel::read_tile_value(cb_ctrl, 0, 0);
            const uint32_t type = w0 & 0xff;
            lap(t_wait);

            if (type == MSG_FLUSH) {
                drain_pend();
                const uint32_t row_slot = ckernel::read_tile_value(cb_ctrl, 0, 1);
                const uint32_t parity = ckernel::read_tile_value(cb_ctrl, 0, 2);
                ctrl_cb.pop_front(1);
                stream_pack_to_unpack_sync();  // the row's last PV pack must be visible

                const uint32_t o_base = row_slot * Sqt * vDHt;
                const uint32_t sum_base = (row_slot * 2 + parity) * Sqt;
                constexpr uint32_t N = 1;
                out_cb.reserve_back(Sqt * vDHt);
                for (uint32_t s = 0; s < Sqt; ++s) {
                    matmul_block_init(cb_sum_res, cb_col_identity, 0, N, 1, N);
                    reconfig_data_format(cb_sum_res, cb_col_identity);
                    CircularBuffer(cb_recip_scratch).reserve_back(1);
                    tile_regs_acquire();
                    matmul_block(cb_sum_res, cb_col_identity, sum_base + s, 0, 0, 0, N, 1, N);
#ifdef ARCH_BLACKHOLE
                    recip_tile_init<false>();
                    MATH((recip_tile<false>(0, VectorMode::C)));
#else
                    recip_tile_init();
                    MATH((recip_tile_first_column_wh_idst0_direct()));
#endif
                    tile_regs_commit();
                    tile_regs_wait();
                    configure_single_tile_pack(cb_recip_scratch);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    pack_tile(0, cb_recip_scratch);
                    tile_regs_release();
                    CircularBuffer(cb_recip_scratch).push_back(1);
                    stream_pack_to_unpack_sync();

                    mul_bcast_cols_init(cb_o_res, cb_recip_scratch);
                    reconfig_data_format(cb_o_res, cb_recip_scratch);
                    CircularBuffer(cb_recip_scratch).wait_front(1);
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        mul_tiles_bcast_cols(cb_o_res, cb_recip_scratch, o_base + s * vDHt + j, 0, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    configure_row_pack_width(cb_out, 1);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        pack_tile<true>(j, cb_out, s * vDHt + j);
                    }
                    tile_regs_release();
                    CircularBuffer(cb_recip_scratch).pop_front(1);
                }
                out_cb.push_back(Sqt * vDHt);
                ++flushed;
                lap(t_flush);
                continue;
            }

            if (type == MSG_WINDOW) {
                const uint32_t n_slots = ckernel::read_tile_value(cb_ctrl, 0, 1);
                ctrl_cb.pop_front(1);
                // Process the buffered visits in region-sized chunks. All but the final chunk's
                // PV drains inside the loop; the final chunk's PV is deferred, so the window's
                // slot credits ride with it (V slots stay pinned until that PV consumed them).
                uint32_t i = 0;
                while (i < vn) {
                    uint32_t take = 0, cols = 0;
                    while (i + take < vn && cols + vbuf[i + take].n * Skt <= kChunkCols) {
                        cols += vbuf[i + take].n * Skt;
                        ++take;
                    }
                    process_chunk(&vbuf[i], take);
                    i += take;
                }
                vn = 0;
                pend_credits += n_slots;
                continue;
            }

            // ---- VISIT: buffer it; processing happens at MSG_WINDOW ----
#if defined(VSA_PROBE) && VSA_PROBE == 1
            ctrl_cb.pop_front(1);
            continue;  // probe 1: delivery floor -- consume the visit without any math
#else
            {
                Visit& v = vbuf[vn];
                v.n = w0 >> 16;
                const uint32_t info = ckernel::read_tile_value(cb_ctrl, 0, 1);
                v.row_slot = info & 0xff;
                v.flags = info & (ROW_IS_FIRST | ROW_PARITY);
                for (uint32_t b = 0; b < v.n; ++b) {
                    v.entries[b] = ckernel::read_tile_value(cb_ctrl, 0, 2 + b);
                }
                ctrl_cb.pop_front(1);
                ++vn;
            }
#endif
        }
        rows_done += pass_rows;
    }
#if defined(VSA_PROBE) && VSA_PROBE == 9
    MATH(({
        const uint32_t t_total = VSA_TICK() - t_begin;
        DPRINT(
            "VSAC v={} total={} wait={} qk={} max={} corr={} pv={} exp={} flush={}\n",
            n_visits, t_total, t_wait, t_qk, t_max, t_corr, t_pv, t_exp, t_flush);
    }));
#endif
}
