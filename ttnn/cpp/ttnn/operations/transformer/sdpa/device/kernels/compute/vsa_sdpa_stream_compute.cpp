// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming compute: batched per-row online-softmax visits. Resident rows keep Q, an O
// accumulator, and an ANCHOR max + THRESHOLD (anchor + T) tile pair in RAM-mode CBs; the exact row
// sums live on the writer core (vsa_sum_service.hpp). The reader windows the block stream (all slots
// of a window are freed together after its visits), so each visit is one row's batch of n windowed
// blocks and the flash machinery -- max reduce, corr, O rescale, syncs -- runs ONCE per batch:
//
//   1. QK    per block: two no-MOP column chains into the row's [Sqt x n*Skt] qk scratch (+ masks)
//   2. MAX   one runtime-width reduce over all n*Skt columns -> candidate c = max(anchor, visit);
//            the UNPACK RISC compares c with the threshold and broadcasts which visits MOVE
//   3. CORR  moved visits only: corr = exp((anchor - c) * scale), anchor := c, threshold := c + T
//   4. RESCALE moved visits only: O *= corr in place; CORR tile to the writer's row sum
//   5. EXP   probs = exp((qk - threshold) * scale) in place, col-batched (arguments never positive)
//   6. PV    deferred one chunk: O accumulates (packer L1-acc); per-visit partial row sums (probs @
//            ones column) stream to the writer, which accumulates them exactly
//
// MSG_WINDOW returns a window's slot credits; the reader's parity word is unused since the lazy
// rescale (kept in the message format).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#if defined(VSA_PROBE) && VSA_PROBE == 9
#include "api/debug/dprint.h"
#define VSA_TICK() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#endif
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
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

// Like stream_pack_to_unpack_sync, but the UNPACK *RISC* (not just its Tensix instruction stream) waits
// for the packer: needed before software L1 reads of freshly packed tiles. The Tensix SEMWAIT only orders
// later unpack instructions; the RISC keeps running C code past it. A dedicated semaphore (UNPACK_OPERAND_SYNC:
// unused by the LLKs) is polled from the RISC; the matching get is also waited for so the next pairing is clean.
ALWI void stream_pack_to_risc_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_OPERAND_SYNC)));
    UNPACK({
        while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) == 0) {
        }
        t6_semaphore_get<>(semaphore::UNPACK_OPERAND_SYNC);
        while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) != 0) {
        }
    });
}

// DEST[idst] += scalar over the WHOLE tile (32 iterations, VectorMode::None). The face-looped SFPU forms
// (add_unary_tile / the first-column exp: VectorMode RC / C) step 16 DEST rows per face, which in this
// kernel's 16-bit DEST (8 rows per face) skips faces and spills into the neighbouring tile: measured as
// garbage in rows 16..31 of the corr/threshold tiles. Full-tile forms (like the phase-5 exp) are exact.
ALWI void add_scalar_tile_full(uint32_t idst, uint32_t param) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (false, ADD_UNARY, 32, DST_ACCUM_MODE),
        idst,
        VectorMode::None,
        param));
}

__attribute__((noinline)) void vsa_trap_bad_ctrl() {
    for (;;) {
    }
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
    constexpr uint32_t chunk_slots = get_compile_time_arg_val(26);  // max blocks per visit = qk region width
    constexpr uint32_t cb_shdr = get_compile_time_arg_val(27);      // row-sum service to the writer
    constexpr uint32_t cb_stiles = get_compile_time_arg_val(28);
    constexpr uint32_t cb_sumback = get_compile_time_arg_val(29);

    constexpr uint32_t k_tiles_per_block = Skt * DHt;
    constexpr uint32_t v_tiles_per_block = Skt * vDHt;
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    // Lazy rescaling threshold: the running max moves only when a visit's max exceeds it by more than
    // 32 logits (exp(32) ~ 8e13 stays far inside fp32/bf16 range). Raw scores are unscaled, so the raw
    // threshold is 32 / scale.
#if defined(VSA_LAZY_T_OVERRIDE)
    constexpr float kLazyT =
        static_cast<float>(VSA_LAZY_T_OVERRIDE) / __builtin_bit_cast(float, scale_fp32);  // TT_VSA_LAZY_T
#else
    constexpr float kLazyT = 2.0f / __builtin_bit_cast(float, scale_fp32);
#endif
    constexpr uint32_t kLazyT_bits = __builtin_bit_cast(uint32_t, kLazyT);

    const uint32_t row_count = get_arg_val<uint32_t>(0);
    const uint32_t n_passes = get_arg_val<uint32_t>(1);  // pass row counts follow (args 2..)
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
    ram(cb_sum_res, 1);  // legacy placeholder (row sums live on the writer core)
    ram(cb_o_res, R_MAX * Sqt * vDHt);
    ram(cb_qk, G * Sqt * Skt);
    // cb_corr, indexed by ROW SLOT (one visit per row per chunk): the visit's candidate max tile, overwritten
    // by its corr once the anchor moved. A chunk can carry up to R_MAX visits (every row of the pass may list
    // a block of the window), more than the window's chunk_slots blocks: never index these by visit.
    ram(cb_corr, R_MAX * Sqt);
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
    uint32_t n_moved = 0, n_nonfirst = 0;
    (void)n_moved;
    (void)n_nonfirst;
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
    uint32_t n_moved = 0, n_nonfirst = 0;
    (void)n_moved;
    (void)n_nonfirst;
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
    constexpr uint32_t kChunkCols = chunk_slots * Skt;                // qk region width in tiles
    constexpr uint32_t kRegionTiles = kChunkCols * Sqt;               // qk region size in tiles
    constexpr uint32_t kMaxVisits = R_MAX;  // one visit per row of the pass at most (TRISC stack is small)
    struct Visit {
        uint32_t row_slot;
        uint32_t flags;  // ROW_IS_FIRST / ROW_PARITY as sent by the reader
        uint32_t n;
        uint32_t tile_base;  // dense sub-block base within the qk region (tiles)
        uint16_t entries[chunk_slots > 0 ? chunk_slots : 1];  // slot | count<<8 | vmask<<15
    };
    Visit vbuf[kMaxVisits];
    // row-sum service: header pages {kind, row, n_tiles, 0} written by the PACK thread (it also packs the
    // tiles that follow); see vsa_sum_service.hpp
    const auto push_hdr = [&](uint32_t kind, uint32_t row, uint32_t ntiles) {
#if defined(VSA_NO_SUMS)
        return;  // probe 10: no row-sum traffic (timing only, output garbage)
#endif
        PACK({
            cb_reserve_back(cb_shdr, 1);
            volatile tt_l1_ptr uint32_t* h =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_shdr).fifo_wr_ptr << 4);
            h[0] = kind;
            h[1] = row;
            h[2] = ntiles;
            h[3] = 0;
            cb_push_back(cb_shdr, 1);
        });
    };
    const uint32_t max_l1_base = get_tile_address(cb_max_res, 0);  // RAM CB: tile t at base + t * 2048
    const uint32_t corr_l1_base = get_tile_address(cb_corr, 0);
    uint32_t vn = 0;

    // Deferred chunk: its probs live in the OTHER qk region; PV runs at the next chunk (or flush).
    Visit pend_v[kMaxVisits];  // up to R_MAX visits per chunk (one per row of the pass)
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
        // MOP PV: one matmul_block per (block, inner) is a full Sqt x vDHt outer-product step
        // (8 tile-MACs per issue instead of 8 no-MOP issues). PV has no transpose, so the MOP path
        // is safe; kt_dim only sets the probs row stride, and a V slot's [Skt x vDHt] tile layout is
        // exactly the [kt x ct] row-major block the MOP expects. DEST ends up row-major
        // (sr * vDHt + vd), so each row's O tiles pack as ONE blocked call.
        uint32_t init_cols = 0;  // sparse listings make same-width (usually 1-block) visits common
        for (uint32_t i = 0; i < pend_n; ++i) {
            const Visit& v = pend_v[i];
            const uint32_t qk_cols = v.n * Skt;
            if (qk_cols != init_cols) {
                matmul_block_init(cb_qk, cb_v_stream, /*transpose=*/0, vDHt, Sqt, qk_cols);
                init_cols = qk_cols;
            }
            tile_regs_acquire();
            for (uint32_t b = 0; b < v.n; ++b) {
                const uint32_t slot = v.entries[b] & 0xff;
                for (uint32_t inner = 0; inner < Skt; ++inner) {
                    matmul_block(
                        cb_qk,
                        cb_v_stream,
                        /*in0=*/pend_qk_base + v.tile_base + b * Skt + inner,
                        /*in1=*/slot * v_tiles_per_block + inner * vDHt,
                        /*dst=*/0,
                        /*transpose=*/0,
                        /*ct=*/vDHt,
                        /*rt=*/Sqt,
                        /*kt=*/qk_cols);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            const bool is_first = (v.flags & ROW_IS_FIRST) != 0;
            const uint32_t o_base_p = v.row_slot * Sqt * vDHt;
            const bool blocked_o = configure_row_pack_width(cb_o_res, vDHt);
            PACK((llk_pack_reconfig_l1_acc(is_first ? 0 : 1)));
            for (uint32_t sr = 0; sr < Sqt; ++sr) {
                if (blocked_o) {
                    sdpa_pack_tile_ooo(sr * vDHt, cb_o_res, o_base_p + sr * vDHt);
                } else {
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        pack_tile<true>(sr * vDHt + vd, cb_o_res, o_base_p + sr * vDHt + vd);
                    }
                }
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            tile_regs_release();
        }
        // Per-row partial sums of every visit in the chunk: probs @ ones-column (fp32 inside the FPU,
        // one bf16 rounding), streamed to the writer, which accumulates them exactly (vsa_sum_service.hpp).
#if !defined(VSA_NO_SUMS)
        // Block form (ct=1, rt=Sqt, one call per k tile, like the PV): half the calls of a per-tile form.
        reconfig_data_format(cb_qk, cb_col_identity);
        uint32_t sum_init_cols = 0;
        // Up to kSumBatch visits per DEST acquire (the acquire/commit/release round trip dominated a
        // per-visit version); each visit still gets its own header + Sqt tiles in the FIFO.
        constexpr uint32_t kSumBatch = 8 / Sqt;
        for (uint32_t i0 = 0; i0 < pend_n; i0 += kSumBatch) {
            const uint32_t nb = (pend_n - i0 < kSumBatch) ? (pend_n - i0) : kSumBatch;
            tile_regs_acquire();
            for (uint32_t b = 0; b < nb; ++b) {
                const Visit& v = pend_v[i0 + b];
                const uint32_t qk_cols = v.n * Skt;
                if (qk_cols != sum_init_cols) {
                    matmul_block_init(cb_qk, cb_col_identity, /*transpose=*/0, /*ct=*/1, /*rt=*/Sqt, /*kt=*/qk_cols);
                    sum_init_cols = qk_cols;
                }
                for (uint32_t k = 0; k < qk_cols; ++k) {
                    matmul_block(
                        cb_qk,
                        cb_col_identity,
                        /*in0=*/pend_qk_base + v.tile_base + k,
                        /*in1=*/0,
                        /*dst=*/b * Sqt,
                        /*transpose=*/0,
                        /*ct=*/1,
                        /*rt=*/Sqt,
                        /*kt=*/qk_cols);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(cb_stiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
            for (uint32_t b = 0; b < nb; ++b) {
                const Visit& v = pend_v[i0 + b];
                if (v.flags & ROW_IS_FIRST) {
                    push_hdr(3 /*FIRST*/, v.row_slot, 0);
                }
                // stream the visit's partial to the writer as is: any bf16 aggregation of partials on the
                // compute side (packer L1-acc) truncates the smaller addend and biased the sums by ~2%
                push_hdr(0 /*PARTIAL*/, v.row_slot, Sqt);
                PACK((cb_reserve_back(cb_stiles, Sqt)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile(b * Sqt + sr, cb_stiles);
                }
                PACK((cb_push_back(cb_stiles, Sqt)));
            }
            tile_regs_release();
        }
#endif  // !VSA_NO_SUMS
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
        {
            // Walk (visit, block) units in a flat stream, filling DEST with two units per acquire
            // regardless of visit boundaries -- single-block visits (sparse listings) would
            // otherwise quarter-fill DEST and pay the acquire round-trip per block.
            uint32_t ui = 0, ub = 0;  // next unit: visit ui, block ub
            while (ui < nv) {
                uint32_t unit_v[2], unit_b[2];
                uint32_t nu = 0;
                while (nu < 2 && ui < nv) {
                    unit_v[nu] = ui;
                    unit_b[nu] = ub;
                    ++nu;
                    if (++ub == vs[ui].n) {
                        ub = 0;
                        ++ui;
                    }
                }
                tile_regs_acquire();
                for (uint32_t j = 0; j < nu; ++j) {
                    const Visit& v = vs[unit_v[j]];
                    const uint32_t slot = v.entries[unit_b[j]] & 0xff;
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
                for (uint32_t j = 0; j < nu; ++j) {
                    const Visit& v = vs[unit_v[j]];
                    const uint32_t qk_cols = v.n * Skt;
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        for (uint32_t c = 0; c < Skt; ++c) {
                            pack_tile<true>(
                                j * Skt * Sqt + c * Sqt + sr, cb_qk,
                                qk_base + v.tile_base + sr * qk_cols + unit_b[j] * Skt + c);
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

        // Phase 2: lazy running max (FlashAttention-3 style). Per row the ANCHOR max lives in max slot
        // (row*2)*Sqt + sr for the whole pass and slot (row*2+1)*Sqt + sr holds the row's THRESHOLD
        // tile anchor + T (T = kLazyT raw logits). Per visit and sub-row the FPU reduce forms the
        // candidate c = max(anchor, visit) in DEST (anchor copied in first) and packs it to the visit's
        // corr slot. The anchor moves only when some row's c exceeds the threshold (decided below on
        // UNPACK with integer bf16 compares): every other visit skips the corr/rescale entirely, so the
        // running sum and O are never rescaled through the 16-bit DEST for them.
        // No SFPU tile-to-tile op is used: sub_binary_tile indexes DEST with the 32-bit-mode tile stride
        // and misreads its second operand in 16-bit DEST mode.
        reconfig_data_format(cb_qk, cb_scale);
        {
            uint32_t g = 0;
            while (g < nv) {
                if (vs[g].flags & ROW_IS_FIRST) {
                    // First visit of a row: DEST[2sr] = DEST[2sr+1] = max(-inf, visit); +T on the odd copy.
                    const Visit& v = vs[g];
                    const uint32_t anc_st = (v.row_slot * 2) * Sqt;
                    const uint32_t thr_st = (v.row_slot * 2 + 1) * Sqt;
                    const uint32_t qk_cols = v.n * Skt;
                    tile_regs_acquire();
                    sdpa_reduce_copy_tile_to_dst_init_short(cb_neginf);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        copy_tile(cb_neginf, 0, sr * 2);
                        copy_tile(cb_neginf, 0, sr * 2 + 1);
                    }
                    reduce_block_max_row_init_runtime(cb_max_res, qk_cols, cb_qk, cb_scale, false);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        reduce_block_max_row_runtime(
                            cb_qk, cb_scale, qk_base + v.tile_base + sr * qk_cols, sr * 2, false, false);
                        reduce_block_max_row_runtime(
                            cb_qk, cb_scale, qk_base + v.tile_base + sr * qk_cols, sr * 2 + 1, false, false);
                    }
                    reduce_block_max_row_uninit_runtime(cb_qk, false, false);
                    MATH((binop_with_scalar_tile_init()));
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        add_scalar_tile_full(sr * 2 + 1, kLazyT_bits);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    configure_single_tile_pack(cb_max_res);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        pack_tile<true>(sr * 2, cb_max_res, anc_st + sr);      // the row's initial anchor
                        pack_tile<true>(sr * 2 + 1, cb_max_res, thr_st + sr);  // anchor + T
                    }
                    tile_regs_release();
                    ++g;
                    continue;
                }
                // up to 2 consecutive non-first visits per acquire: reduces into DEST tiles 0..3 only (4 visits
                // x Sqt = tiles 4..7 gave wrong candidates for the last two visits: see VSA_STREAM_DESIGN.md)
                uint32_t ng = 0;
                while (g + ng < nv && ng < 2 && !(vs[g + ng].flags & ROW_IS_FIRST)) {
                    ++ng;
                }
                tile_regs_acquire();
                for (uint32_t i = 0; i < ng; ++i) {
                    const Visit& v = vs[g + i];
                    const uint32_t anc_st = (v.row_slot * 2) * Sqt;
                    const uint32_t qk_cols = v.n * Skt;
                    // anchor copy immediately followed by its reduce: copying every visit's anchor first
                    // and reducing afterwards corrupted rows (the reduce init disturbs earlier DEST tiles)
                    sdpa_reduce_copy_tile_to_dst_init_short(cb_max_res);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        copy_tile(cb_max_res, anc_st + sr, i * Sqt + sr);
                    }
                    reduce_block_max_row_init_runtime(cb_max_res, qk_cols, cb_qk, cb_scale, false);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        reduce_block_max_row_runtime(
                            cb_qk, cb_scale, qk_base + v.tile_base + sr * qk_cols, i * Sqt + sr, false, false);
                    }
                    reduce_block_max_row_uninit_runtime(cb_qk, false, false);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(0)));
                configure_single_tile_pack(cb_corr);
                for (uint32_t i = 0; i < ng; ++i) {
                    const uint32_t cand_st = vs[g + i].row_slot * Sqt;
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        pack_tile<true>(i * Sqt + sr, cb_corr, cand_st + sr);  // candidate max
                    }
                }
                for (uint32_t i = 0; i < ng; ++i) {
                }
                tile_regs_release();
                g += ng;
            }
        }
        stream_pack_to_unpack_sync();  // s2: candidates/thresholds visible to the unpacker
        stream_pack_to_risc_sync();    // ... and to the UNPACK RISC's reads below
        // Which visits move their anchor? Decided once on UNPACK: a visit moves when any row's candidate
        // (column 0 of its corr-slot tile) exceeds the row's threshold tile, compared as order-preserving
        // integer keys of the bf16 bits. Broadcast by mailbox so all three threads branch identically.
        uint32_t updated = 0;
        UNPACK({
            // the packer rewrote these L1 tiles since this RISC last read the same addresses: drop the
            // RISC's L1 read cache first (stale candidates/thresholds gave path-dependent decisions)
            invalidate_l1_cache();
            auto bkey = [](uint32_t b) -> uint32_t {
                return (b & 0x8000u) ? (0x7FFFu - (b & 0x7FFFu)) : (b | 0x8000u);
            };
            for (uint32_t i = 0; i < nv; ++i) {
                const Visit& v = vs[i];
                if (v.flags & ROW_IS_FIRST) {
                    continue;
                }
                const uint32_t thr_st = (v.row_slot * 2 + 1) * Sqt;
                bool moved = false;
                for (uint32_t sr = 0; sr < Sqt && !moved; ++sr) {
                    volatile tt_l1_ptr uint16_t* cand =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(corr_l1_base + (v.row_slot * Sqt + sr) * 2048);
                    volatile tt_l1_ptr uint16_t* thr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(max_l1_base + (thr_st + sr) * 2048);
                    for (uint32_t r = 0; r < 32; ++r) {
                        const uint32_t off = (r < 16 ? 0u : 512u) + (r & 15) * 16;  // column 0, face-major (bf16 units)
                        if (bkey(cand[off]) > bkey(thr[off])) {
                            moved = true;
                            break;
                        }
                    }
                }
                if (moved) {
                    updated |= 1u << i;
                }
            }
            mailbox_write(ckernel::ThreadId::MathThreadId, updated);
            mailbox_write(ckernel::ThreadId::PackThreadId, updated);
        })
        MATH(updated = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(updated = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
#if defined(VSA_PROBE) && VSA_PROBE == 9
        n_moved += __builtin_popcount(updated & ((1u << nv) - 1u));
        n_nonfirst += nv;
#endif
        lap(t_max);

        // Phase 3: for each visit that moves its anchor (rare), corr = exp((anchor - candidate) * scale),
        // the candidate becomes the row's anchor and candidate + T its threshold. Two acquires per moved
        // visit, DEST tiles 0..3 only (see phase 2). The corr exp runs on the MATH thread: a face-looped SFPU
        // op on the PACK thread (exp_tile_first_column) displaced the packer's DEST reads for the packs that
        // followed it (garbage in rows 16..31 of the packed tiles). The corr overwrites the candidate slot,
        // so the threshold (which needs the candidate) is derived first.
        bool any_corr = false;
        {
            for (uint32_t i = 0; i < nv; ++i) {
                const Visit& v = vs[i];
                if ((v.flags & ROW_IS_FIRST) || !((updated >> i) & 1u)) {
                    continue;
                }
                any_corr = true;
                const uint32_t anc_st = (v.row_slot * 2) * Sqt;
                const uint32_t thr_st = (v.row_slot * 2 + 1) * Sqt;
                const uint32_t cand_st = v.row_slot * Sqt;  // candidate now, corr after (2)
                // (1) DEST[sr] = candidate + T -> new threshold
                tile_regs_acquire();
                reconfig_data_format_srca(cb_corr);
                copy_tile_to_dst_init_short(cb_corr);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    copy_tile(cb_corr, cand_st + sr, sr);
                }
                MATH((binop_with_scalar_tile_init()));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    add_scalar_tile_full(sr, kLazyT_bits);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(0)));
                configure_single_tile_pack(cb_max_res);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(sr, cb_max_res, thr_st + sr);  // new threshold
                }
                tile_regs_release();
                // (2) DEST[sr] = anchor - candidate -> corr (into the candidate slot); DEST[Sqt + sr] = candidate
                //     -> new anchor. The unpacker reads the candidate before the corr pack lands (same acquire).
                tile_regs_acquire();
                sub_init(cb_max_res, cb_corr);
                reconfig_data_format(cb_max_res, cb_corr);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    sub_tiles(cb_max_res, cb_corr, anc_st + sr, cand_st + sr, sr);
                }
                reconfig_data_format_srca(cb_corr);
                copy_tile_to_dst_init_short(cb_corr);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    copy_tile(cb_corr, cand_st + sr, Sqt + sr);
                }
                // exact exp with the scale applied inside the SFPU (one bf16 rounding, as the dense kernel)
                exp_tile_init<EXP_APPROX_MODE, 0x3F800000u, InputClamping::None>();
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    exp_tile<EXP_APPROX_MODE, true, InputClamping::None, 8>(sr, VectorMode::RC, scale_bf16);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(0)));
                configure_single_tile_pack(cb_corr);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(sr, cb_corr, cand_st + sr);  // corr
                }
                configure_single_tile_pack(cb_max_res);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(Sqt + sr, cb_max_res, anc_st + sr);  // new anchor
                }
                tile_regs_release();
            }
        }
        if (any_corr) {
            stream_pack_to_unpack_sync();  // s3: corrs and moved anchors visible
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
                if ((v.flags & ROW_IS_FIRST) || !((updated >> i) & 1u)) {
                    continue;
                }
                const uint32_t corr_st = v.row_slot * Sqt;  // the corr replaced the candidate (phase 3)
                const uint32_t o_base = v.row_slot * Sqt * vDHt;
                // the writer rescales its exact row sum by the same corr. This runs AFTER drain_pend(),
                // so the corr follows every earlier visit's partial sum (they belong to the old anchor).
                mul_bcast_cols_init(cb_o_res, cb_corr);
                reconfig_data_format(cb_o_res, cb_corr);
                push_hdr(1 /*CORR*/, v.row_slot, Sqt);
#if !defined(VSA_NO_SUMS)
                PACK((cb_reserve_back(cb_stiles, Sqt)));
#endif
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        mul_tiles_bcast_cols(cb_o_res, cb_corr, o_base + sr * vDHt + j, corr_st + sr, j);
                    }
                    reconfig_data_format_srca(cb_o_res, cb_corr);
                    copy_tile_to_dst_init_short(cb_corr);
                    copy_tile(cb_corr, corr_st + sr, vDHt);  // corr copy for the writer
                    mul_bcast_cols_init(cb_o_res, cb_corr);
                    reconfig_data_format(cb_o_res, cb_corr);
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
#if !defined(VSA_NO_SUMS)
                    configure_single_tile_pack(cb_stiles);
                    pack_tile(vDHt, cb_stiles);
#endif
                    tile_regs_release();
                }
#if !defined(VSA_NO_SUMS)
                PACK((cb_push_back(cb_stiles, Sqt)));
#endif
            }
        }
        lap(t_corr);

        // Phase 5: probs = exp((qk - threshold) * scale) in place per visit, in DEST-sized column batches.
        exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
        for (uint32_t i = 0; i < nv; ++i) {
            const Visit& v = vs[i];
            // exp reference = the row's THRESHOLD tile (anchor + T): while the anchor stands, every score is
            // <= anchor + T, so the exp argument stays <= 0 (the fast approx exp is only accurate there).
            // Online softmax is exact for any consistent reference; corr = exp(anchor_old - anchor_new)
            // is unchanged since both references carry the same +T.
            const uint32_t new_st = (v.row_slot * 2 + 1) * Sqt;
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
    for (uint32_t pass = 0; pass < n_passes; ++pass) {
        const uint32_t pass_rows = get_arg_val<uint32_t>(2 + pass);
        if (pass_rows == 0) {
            continue;  // a peer with nothing resident this pass (distributed dealing)
        }
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
                const uint32_t row_slot = ckernel::read_tile_value(cb_ctrl, 0, 1);  // word 2 (parity) unused
                ctrl_cb.pop_front(1);
                stream_pack_to_unpack_sync();  // the row's last PV pack must be visible

                const uint32_t o_base = row_slot * Sqt * vDHt;
                push_hdr(2 /*FLUSH*/, row_slot, 0);
#if defined(VSA_NO_SUMS)
                const uint32_t cb_sumsrc = cb_col_identity;  // probe 10: pretend sum = 1
#else
                CircularBuffer(cb_sumback).wait_front(Sqt);  // the writer's exact row sums (bf16 tiles, column 0)
                const uint32_t cb_sumsrc = cb_sumback;
#endif
                out_cb.reserve_back(Sqt * vDHt);
                for (uint32_t s = 0; s < Sqt; ++s) {
                    reconfig_data_format_srca(cb_sumsrc);
                    copy_tile_to_dst_init_short(cb_sumsrc);
                    CircularBuffer(cb_recip_scratch).reserve_back(1);
                    tile_regs_acquire();
                    copy_tile(cb_sumsrc, cb_sumsrc == cb_sumback ? s : 0, 0);
#ifdef ARCH_BLACKHOLE
                    recip_tile_init<false>();
                    MATH((recip_tile<false>(0, VectorMode::C)));
#else
                    recip_tile_init();
                    MATH((recip_tile_first_column_wh_idst0_direct()));
#endif
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_recip_scratch);
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
#if !defined(VSA_NO_SUMS)
                CircularBuffer(cb_sumback).pop_front(Sqt);
#endif
                out_cb.push_back(Sqt * vDHt);
                ++flushed;
                lap(t_flush);
                continue;
            }

            if (type == MSG_WINDOW) {
                const uint32_t n_slots = ckernel::read_tile_value(cb_ctrl, 0, 1);
                ctrl_cb.pop_front(1);
                if (n_slots == 0) {
                    drain_pend();  // NUDGE: the sender is gated on credits this deferred PV holds
                    continue;
                }
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
                if (pend_n == 0) {  // no deferred PV holds these V slots (e.g. probe-1 skips)
                    free_cb.reserve_back(n_slots);
                    free_cb.push_back(n_slots);
                } else {
                    pend_credits += n_slots;
                }
                continue;
            }

            // ---- VISIT: buffer it; processing happens at MSG_WINDOW ----
#if defined(VSA_PROBE) && VSA_PROBE == 1
            ctrl_cb.pop_front(1);
            continue;  // probe 1: delivery floor -- consume the visit without any math
#else
            {
                if (type != MSG_VISIT || vn >= kMaxVisits || (w0 >> 16) == 0 || (w0 >> 16) > chunk_slots) {
                    vsa_trap_bad_ctrl();
                }
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
    {
        const uint32_t t_total = VSA_TICK() - t_begin;
        DPRINT(
            "VSAC v={} total={} wait={} qk={} max={} corr={} pv={} exp={} flush={} moved={}/{}\n",
            n_visits,
            t_total,
            t_wait,
            t_qk,
            t_max,
            t_corr,
            t_pv,
            t_exp,
            t_flush,
            n_moved,
            n_nonfirst);
    }
#endif
}
