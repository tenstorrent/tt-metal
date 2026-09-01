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


    // Lag-1 PV pipeline: each visit's PV is deferred to the start of the next visit (or to the
    // next flush/window/pass boundary), so a visit's pack-thread exp overlaps the next visit's
    // QK and reduce math instead of serializing against them. qk scratch ping-pongs between two
    // regions so the deferred probs survive the next visit's QK packs.
    struct PendingPV {
        bool valid = false;
        uint32_t row_slot = 0;
        bool is_first = false;
        uint32_t qk_base = 0;   // tile offset of the probs region
        uint32_t qk_cols = 0;
        uint32_t n_blocks = 0;
        uint32_t slots[8];
    };
    PendingPV pend;
    uint32_t qk_region = 0;  // ping-pong: 0 or 1

    const auto drain_pending_pv = [&]() {
        if (!pend.valid) {
            return;
        }
        stream_pack_to_unpack_sync();  // probs of the pending visit must be visible (usually free)
        reconfig_data_format(cb_v_stream, cb_qk);
        mm_no_mop_init_short(cb_qk, cb_v_stream, /*transpose=*/false, 1, Sqt, pend.qk_cols);
        tile_regs_acquire();
        for (uint32_t vd = 0; vd < vDHt; ++vd) {
            for (uint32_t b = 0; b < pend.n_blocks; ++b) {
                for (uint32_t inner = 0; inner < Skt; ++inner) {
                    matmul_block_no_mop(
                        cb_qk,
                        cb_v_stream,
                        /*in0=*/pend.qk_base + b * Skt + inner,
                        /*in1=*/pend.slots[b] * v_tiles_per_block + inner * vDHt + vd,
                        /*dst=*/vd * Sqt,
                        /*transpose=*/false,
                        /*w=*/1,
                        /*h=*/Sqt,
                        /*stride=*/pend.qk_cols);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        const uint32_t o_base_p = pend.row_slot * Sqt * vDHt;
        configure_row_pack_width(cb_o_res, 1);
        PACK((llk_pack_reconfig_l1_acc(pend.is_first ? 0 : 1)));
        for (uint32_t vd = 0; vd < vDHt; ++vd) {
            for (uint32_t sr = 0; sr < Sqt; ++sr) {
                pack_tile<true>(vd * Sqt + sr, cb_o_res, o_base_p + sr * vDHt + vd);
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        tile_regs_release();
        pend.valid = false;
    };

    uint32_t rows_done = 0;
    while (rows_done < row_count) {
        const uint32_t pass_rows = (row_count - rows_done < R_MAX) ? (row_count - rows_done) : R_MAX;
        qdone_cb.wait_front(1);  // pass Q resident
        qdone_cb.pop_front(1);

        uint32_t flushed = 0;
        while (flushed < pass_rows) {
            ctrl_cb.wait_front(1);
            const uint32_t w0 = ckernel::read_tile_value(cb_ctrl, 0, 0);
            const uint32_t type = w0 & 0xff;

            if (type == MSG_FLUSH) {
                drain_pending_pv();
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
                continue;
            }

            if (type == MSG_WINDOW) {
                drain_pending_pv();  // the pending visit still reads this window's V slots
                const uint32_t n_slots = ckernel::read_tile_value(cb_ctrl, 0, 1);
                ctrl_cb.pop_front(1);
                free_cb.reserve_back(n_slots);
                free_cb.push_back(n_slots);
                continue;
            }

            // ---- VISIT: one resident row x n windowed blocks ----
            const uint32_t n_blocks = w0 >> 16;
            const uint32_t info = ckernel::read_tile_value(cb_ctrl, 0, 1);
            uint32_t entries[16];
            for (uint32_t b = 0; b < n_blocks; ++b) {
                entries[b] = ckernel::read_tile_value(cb_ctrl, 0, 2 + b);
            }
            ctrl_cb.pop_front(1);
#if defined(VSA_PROBE) && (VSA_PROBE == 1 || VSA_PROBE >= 3)
            continue;  // probe 1/3: delivery floor -- consume the visit without any math
#endif

            const uint32_t row_slot = info & 0xff;
            const bool is_first = (info & ROW_IS_FIRST) != 0;
            const uint32_t parity = (info & ROW_PARITY) ? 1u : 0u;
            const uint32_t new_st = (row_slot * 2 + parity) * Sqt;
            const uint32_t old_st = (row_slot * 2 + (parity ^ 1u)) * Sqt;
            const uint32_t o_base = row_slot * Sqt * vDHt;
            const uint32_t qk_cols = n_blocks * Skt;
            if (pend.valid && pend.row_slot == row_slot) {
                drain_pending_pv();  // same row back-to-back: its O and probs must land first
            }
            const uint32_t qk_half_tiles = (stream_depth / 2) * Skt * Sqt;  // tiles per qk region
            const uint32_t qk_base = qk_region * qk_half_tiles;

            // Phase 1: QK per block into the row's qk scratch (+ ragged masks), then sync.
            reconfig_data_format(cb_k_stream, cb_q_res);
            exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
            mm_no_mop_init_short(cb_q_res, cb_k_stream, /*transpose=*/true, 1, Sqt, DHt);
            pack_reconfig_data_format(cb_qk);
            configure_row_pack_width(cb_qk, 1);
            for (uint32_t b = 0; b < n_blocks; ++b) {
                const uint32_t slot = entries[b] & 0xff;
                tile_regs_acquire();
                for (uint32_t c = 0; c < Skt; ++c) {
                    for (uint32_t inner = 0; inner < DHt; ++inner) {
                        matmul_block_no_mop(
                            cb_q_res, cb_k_stream, row_slot * Sqt * DHt + inner,
                            slot * k_tiles_per_block + c * DHt + inner, c * Sqt, /*transpose=*/true,
                            /*w=*/1, /*h=*/Sqt, /*stride=*/DHt);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    for (uint32_t c = 0; c < Skt; ++c) {
                        pack_tile<true>(c * Sqt + sr, cb_qk, qk_base + sr * qk_cols + b * Skt + c);
                    }
                }
                tile_regs_release();
            }
#if defined(VSA_PROBE) && VSA_PROBE == 2
            // probe 2: math floor -- QK above and the deferred PV below, no softmax phases.
            drain_pending_pv();
#else
            bool any_mask = false;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                const uint32_t count = (entries[b] >> 8) & 0x7f;
                if (count >= block_size) {
                    continue;
                }
                const uint32_t slot = entries[b] & 0xff;
                const uint32_t btile = count / keys_per_tile;
                const uint32_t bcol = count % keys_per_tile;
                uint32_t first_full = btile;
                if (bcol != 0) {
                    vsa_stamp_mask(cb_vmask, slot, cb_qk, qk_base + b * Skt + btile, qk_cols, Sqt);
                    first_full = btile + 1;
                }
                for (uint32_t kt = first_full; kt < Skt; ++kt) {
                    vsa_stamp_mask(cb_neginf, 0, cb_qk, qk_base + b * Skt + kt, qk_cols, Sqt);
                }
                any_mask = true;
            }
            (void)any_mask;
            stream_pack_to_unpack_sync();  // s1: qk (+masks) visible

            // Phase 2: one runtime-width row max over all n*Skt columns -> the NEW slot.
            reconfig_data_format(cb_qk, cb_scale);
            {
                tile_regs_acquire();
                if (!is_first) {
                    sdpa_reduce_copy_tile_to_dst_init_short(cb_max_res);
                    for (uint32_t sr = 0; sr < Sqt; ++sr) {
                        copy_tile(cb_max_res, old_st + sr, sr);
                    }
                }
                reduce_block_max_row_init_runtime(cb_max_res, qk_cols, cb_qk, cb_scale, false);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    reduce_block_max_row_runtime(cb_qk, cb_scale, qk_base + sr * qk_cols, sr, false, false);
                }
                reduce_block_max_row_uninit_runtime(cb_qk, false, false);
                tile_regs_commit();
                tile_regs_wait();
                configure_single_tile_pack(cb_max_res);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(sr, cb_max_res, new_st + sr);
                }
                tile_regs_release();
            }
            stream_pack_to_unpack_sync();  // s2: new max visible

            // Phase 3+4: corr and the O/sum rescale, once per batch (skipped on a first visit).
            if (!is_first) {
                exp_packthread_tile_init<EXP_APPROX_MODE>();
                sub_init(cb_max_res, cb_max_res);
                tile_regs_acquire();
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    sub_tiles(cb_max_res, cb_max_res, old_st + sr, new_st + sr, sr);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(sr)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                configure_single_tile_pack(cb_corr);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    pack_tile<true>(sr, cb_corr, sr);
                }
                tile_regs_release();
                stream_pack_to_unpack_sync();  // s3: corr visible

                mul_bcast_cols_init(cb_o_res, cb_corr);
                reconfig_data_format(cb_o_res, cb_corr);
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        mul_tiles_bcast_cols(cb_o_res, cb_corr, o_base + sr * vDHt + j, sr, j);
                    }
                    mul_tiles_bcast_cols(cb_sum_res, cb_corr, old_st + sr, sr, vDHt);
                    tile_regs_commit();
                    tile_regs_wait();
                    configure_row_pack_width(cb_o_res, 1);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        pack_tile<true>(j, cb_o_res, o_base + sr * vDHt + j);
                    }
                    configure_single_tile_pack(cb_sum_res);
                    pack_tile<true>(vDHt, cb_sum_res, new_st + sr);
                    tile_regs_release();
                }
            }

            drain_pending_pv();  // previous visit's PV: its exp overlapped our math phases above

            // Phase 5: probs = exp((qk - new_max) * scale) in place, in DEST-sized column batches;
            // row sums L1-acc onto the NEW sum slot (overwrite on a first visit's first batch).
            exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
            for (uint32_t kc = 0; kc < qk_cols; kc += 4) {
                const uint32_t w = (qk_cols - kc < 4) ? (qk_cols - kc) : 4;
                sub_bcast_cols_init_short_custom(cb_qk, cb_max_res, w);
                reconfig_data_format(cb_qk, cb_max_res);
                tile_regs_acquire();
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    sub_tiles_bcast_cols_custom(cb_qk, cb_max_res, qk_base + sr * qk_cols + kc, new_st + sr, sr * w, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((llk_pack_relu_config(ReluConfig::zero())));
                for (uint32_t t = 0; t < Sqt * w; ++t) {
                    exp_packthread_tile<true, false, InputClamping::None, 32>(t, VectorMode::None);
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                configure_row_pack_width(cb_qk, 1);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t sr = 0; sr < Sqt; ++sr) {
                    for (uint32_t c = 0; c < w; ++c) {
                        pack_tile<true>(sr * w + c, cb_qk, qk_base + sr * qk_cols + kc + c);
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
#endif  // !(VSA_PROBE == 2)
            // No s4/s5 here: the PV of this visit is deferred to the next visit (or the next
            // flush/window boundary), so this visit's pack-thread exp overlaps the next visit's
            // QK and reduce math. drain_pending_pv() carries the probs-visibility sync.
            pend.valid = true;
            pend.row_slot = row_slot;
            pend.is_first = is_first;
            pend.qk_base = qk_base;
            pend.qk_cols = qk_cols;
            pend.n_blocks = n_blocks;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                pend.slots[b] = entries[b] & 0xff;
            }
            qk_region ^= 1u;
        }
        rows_done += pass_rows;
    }
}
