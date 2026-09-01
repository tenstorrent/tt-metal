// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) compute: fused multi-row online-softmax visit engine. Resident rows keep
// Q, an O accumulator, and PING-PONG running max/sum slots in RAM-mode CBs; each arriving KV block
// is visited by every resident row that lists it, phase-batched in groups of up to G rows:
//
//   1. QK    per row: one matmul_block [Sqt x Skt x DHt] into the row's qk scratch (+ ragged mask)
//   2. MAX   per row: eltwise-max reduce with the old max slot -> the NEW max slot (parity flip)
//   3. CORR  non-first rows: corr = exp((old_max - new_max) * scale)
//   4. RESCALE non-first rows: O *= corr in place; old_sum * corr -> the NEW sum slot
//   5. EXP   per row: probs = exp((qk - new_max) * scale) in place; row-sum L1-accs onto the NEW sum
//   6. PV    per row: one matmul_block [Sqt x vDHt x Skt]; packer-L1-acc onto O (overwrite on first)
//
// Ping-pong state slots eliminate every state copy; the visit parity comes from the reader. Five
// pack->unpack syncs order the phases; state CBs are indexed absolutely and never pushed/popped.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

ALWI void stream_pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

constexpr uint32_t MSG_VISIT = 0;
constexpr uint32_t MSG_FLUSH = 1;
constexpr uint32_t FLAG_HAS_VMASK = 1u << 8;
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

    CircularBuffer(cb_scale).wait_front(1);
    CircularBuffer(cb_col_identity).wait_front(1);
    CircularBuffer(cb_neginf).wait_front(1);

    // Hand the reader its initial stream-slot credits.
    free_cb.reserve_back(stream_depth);
    free_cb.push_back(stream_depth);

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

            // ---- VISIT ----
            const uint32_t n_rows = w0 >> 16;
            const bool has_vmask = (w0 & FLAG_HAS_VMASK) != 0;
            const uint32_t slot = ckernel::read_tile_value(cb_ctrl, 0, 1);
            const uint32_t count = ckernel::read_tile_value(cb_ctrl, 0, 2);
            const bool ragged = count < block_size;
            const uint32_t boundary_tile = count / keys_per_tile;
            const uint32_t boundary_col = count % keys_per_tile;
            uint32_t rowinfo[32];
            for (uint32_t i = 0; i < n_rows; ++i) {
                rowinfo[i] = ckernel::read_tile_value(cb_ctrl, 0, 3 + i);
            }
            ctrl_cb.pop_front(1);
            if (has_vmask) {
                CircularBuffer(cb_vmask).wait_front(1);
            }

            for (uint32_t g0 = 0; g0 < n_rows; g0 += G) {
                const uint32_t gn = (n_rows - g0 < G) ? (n_rows - g0) : G;

                // Phase 1: QK per row into its qk scratch region (+ ragged mask), then sync.
                // K blocks are stored [Skt x DHt] (one row per key tile): each key tile is its own
                // accumulation chain over contiguous in1 tiles, via the same no-MOP transposed
                // matmul the v1/MSA kernels use (the MOP matmul_block path does not honor
                // transpose).
                reconfig_data_format(cb_k_stream, cb_q_res);
                mm_no_mop_init_short(cb_q_res, cb_k_stream, /*transpose=*/true, 1, Sqt, DHt);
                pack_reconfig_data_format(cb_qk);
                configure_row_pack_width(cb_qk, 1);
                for (uint32_t i = 0; i < gn; ++i) {
                    const uint32_t row_slot = rowinfo[g0 + i] & 0xff;
                    for (uint32_t c = 0; c < Skt; ++c) {
                        blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/Skt>(
                            cb_q_res,
                            cb_k_stream,
                            cb_qk,
                            /*in0_index_start=*/row_slot * Sqt * DHt,
                            /*in1_index_start=*/slot * k_tiles_per_block + c * DHt,
                            /*row_subblock_idx=*/i,
                            /*out_col_offset=*/c,
                            /*subblock_w=*/1,
                            /*subblock_h=*/Sqt,
                            /*inner_dim=*/DHt,
                            /*matmul_stride=*/DHt,
                            /*skip_pack_configure=*/true);
                    }
                }
                if (ragged) {
                    uint32_t first_full = boundary_tile;
                    if (boundary_col != 0) {
                        first_full = boundary_tile + 1;
                    }
                    for (uint32_t i = 0; i < gn; ++i) {
                        const uint32_t row_base = i * Sqt;  // in row-tiles of the qk scratch
                        if (boundary_col != 0) {
                            apply_partial_mask_lightweight(cb_vmask, 0, cb_qk, boundary_tile, Skt, Sqt, row_base);
                        }
                        for (uint32_t kt = first_full; kt < Skt; ++kt) {
                            apply_partial_mask_lightweight(cb_neginf, 0, cb_qk, kt, Skt, Sqt, row_base);
                        }
                    }
                }
                stream_pack_to_unpack_sync();  // s1: qk (+mask) visible

                // Phase 2: running max -> the NEW slot (eltwise max with the old slot unless first).
                reconfig_data_format(cb_qk, cb_scale);
                for (uint32_t i = 0; i < gn; ++i) {
                    const uint32_t info = rowinfo[g0 + i];
                    const uint32_t row_slot = info & 0xff;
                    const bool is_first = (info & ROW_IS_FIRST) != 0;
                    const uint32_t parity = (info & ROW_PARITY) ? 1u : 0u;
                    const uint32_t new_max = (row_slot * 2 + parity) * Sqt;
                    const uint32_t old_max = (row_slot * 2 + (parity ^ 1u)) * Sqt;
                    tile_regs_acquire();
                    if (!is_first) {
                        sdpa_reduce_copy_tile_to_dst_init_short(cb_max_res);
                        for (uint32_t s = 0; s < Sqt; ++s) {
                            copy_tile(cb_max_res, old_max + s, s);
                        }
                    }
                    reduce_block_max_row_init<Skt>(cb_max_res);
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        reduce_block_max_row<Skt>(cb_qk, cb_scale, (i * Sqt + s) * Skt, s);
                    }
                    reduce_block_max_row_uninit(cb_qk);
                    tile_regs_commit();
                    tile_regs_wait();
                    configure_single_tile_pack(cb_max_res);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        pack_tile<true>(s, cb_max_res, new_max + s);
                    }
                    tile_regs_release();
                }
                stream_pack_to_unpack_sync();  // s2: new maxes visible

                // Phase 3: corr = exp((old_max - new_max) * scale) for non-first rows.
                bool any_rescale = false;
                exp_packthread_tile_init<EXP_APPROX_MODE>();
                for (uint32_t i = 0; i < gn; ++i) {
                    const uint32_t info = rowinfo[g0 + i];
                    if (info & ROW_IS_FIRST) {
                        continue;
                    }
                    any_rescale = true;
                    const uint32_t row_slot = info & 0xff;
                    const uint32_t parity = (info & ROW_PARITY) ? 1u : 0u;
                    const uint32_t new_max = (row_slot * 2 + parity) * Sqt;
                    const uint32_t old_max = (row_slot * 2 + (parity ^ 1u)) * Sqt;
                    sub_init(cb_max_res, cb_max_res);
                    tile_regs_acquire();
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        sub_tiles(cb_max_res, cb_max_res, old_max + s, new_max + s, s);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(s)));
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    configure_single_tile_pack(cb_corr);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        pack_tile<true>(s, cb_corr, i * Sqt + s);
                    }
                    tile_regs_release();
                }
                if (any_rescale) {
                    stream_pack_to_unpack_sync();  // s3: corr visible

                    // Phase 4: O *= corr (in place); old_sum * corr -> the NEW sum slot.
                    mul_bcast_cols_init(cb_o_res, cb_corr);
                    reconfig_data_format(cb_o_res, cb_corr);
                    for (uint32_t i = 0; i < gn; ++i) {
                        const uint32_t info = rowinfo[g0 + i];
                        if (info & ROW_IS_FIRST) {
                            continue;
                        }
                        const uint32_t row_slot = info & 0xff;
                        const uint32_t parity = (info & ROW_PARITY) ? 1u : 0u;
                        const uint32_t new_sum = (row_slot * 2 + parity) * Sqt;
                        const uint32_t old_sum = (row_slot * 2 + (parity ^ 1u)) * Sqt;
                        const uint32_t o_base = row_slot * Sqt * vDHt;
                        for (uint32_t s = 0; s < Sqt; ++s) {
                            tile_regs_acquire();
                            for (uint32_t j = 0; j < vDHt; ++j) {
                                mul_tiles_bcast_cols(cb_o_res, cb_corr, o_base + s * vDHt + j, i * Sqt + s, j);
                            }
                            mul_tiles_bcast_cols(cb_sum_res, cb_corr, old_sum + s, i * Sqt + s, vDHt);
                            tile_regs_commit();
                            tile_regs_wait();
                            configure_row_pack_width(cb_o_res, 1);
                            PACK((llk_pack_reconfig_l1_acc(0)));
                            for (uint32_t j = 0; j < vDHt; ++j) {
                                pack_tile<true>(j, cb_o_res, o_base + s * vDHt + j);
                            }
                            configure_single_tile_pack(cb_sum_res);
                            pack_tile<true>(vDHt, cb_sum_res, new_sum + s);
                            tile_regs_release();
                        }
                    }
                }

                // Phase 5: probs = exp((qk - new_max) * scale) in place; row sums onto the NEW sum
                // slot (packer L1-acc; plain overwrite for a first visit's fresh slot).
                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
                for (uint32_t i = 0; i < gn; ++i) {
                    const uint32_t info = rowinfo[g0 + i];
                    const uint32_t row_slot = info & 0xff;
                    const bool is_first = (info & ROW_IS_FIRST) != 0;
                    const uint32_t parity = (info & ROW_PARITY) ? 1u : 0u;
                    const uint32_t state = (row_slot * 2 + parity) * Sqt;
                    const uint32_t qk_base = i * Sqt;  // row-tiles

                    sub_bcast_cols_init_short_custom(cb_qk, cb_max_res, Skt);
                    tile_regs_acquire();
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        sub_tiles_bcast_cols_custom(cb_qk, cb_max_res, (qk_base + s) * Skt, state + s, s * Skt, Skt);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                    for (uint32_t t = 0; t < Sqt * Skt; ++t) {
                        exp_packthread_tile<true, false, InputClamping::None, 32>(t, VectorMode::None);
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    // probs back in place
                    configure_row_pack_width(cb_qk, 1);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t t = 0; t < Sqt * Skt; ++t) {
                        pack_tile<true>(t, cb_qk, qk_base * Skt + t);
                    }
                    // row sums: acc unless this is the row's first visit writing a fresh slot;
                    // a first visit overwrites with its first column then accs the rest.
                    configure_single_tile_pack(cb_sum_res);
                    uint32_t t = 0;
                    for (uint32_t s = 0; s < Sqt; ++s) {
                        for (uint32_t c = 0; c < Skt; ++c, ++t) {
                            const bool overwrite = is_first && c == 0;
                            PACK((llk_pack_reconfig_l1_acc(overwrite ? 0 : 1)));
                            pack_tile<true>(t, cb_sum_res, state + s);
                        }
                    }
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                    tile_regs_release();
                }
                stream_pack_to_unpack_sync();  // s4: probs visible

                // Phase 6: O += probs @ V (packer L1-acc; overwrite on a row's first visit),
                // via the v1/MSA per-output-column no-MOP matmul.
                reconfig_data_format(cb_v_stream, cb_qk);
                mm_no_mop_init_short(cb_qk, cb_v_stream, /*transpose=*/false, 1, Sqt, Skt);
                configure_row_pack_width(cb_o_res, 1);
                for (uint32_t i = 0; i < gn; ++i) {
                    const uint32_t info = rowinfo[g0 + i];
                    const uint32_t row_slot = info & 0xff;
                    const bool is_first = (info & ROW_IS_FIRST) != 0;
                    PACK((llk_pack_reconfig_l1_acc(is_first ? 0 : 1)));
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        blocked_matmul_and_pack<false, /*in1_stride=*/vDHt, /*out_num_cols=*/vDHt>(
                            cb_qk,
                            cb_v_stream,
                            cb_o_res,
                            /*in0_index_start=*/i * Sqt * Skt,
                            /*in1_index_start=*/slot * v_tiles_per_block + vd,
                            /*row_subblock_idx=*/row_slot,
                            /*out_col_offset=*/vd,
                            /*subblock_w=*/1,
                            /*subblock_h=*/Sqt,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/Skt,
                            /*skip_pack_configure=*/true);
                    }
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
                stream_pack_to_unpack_sync();  // s5: O state visible to the next visit's rescale
            }

            if (has_vmask) {
                CircularBuffer(cb_vmask).pop_front(1);
            }
            free_cb.reserve_back(1);
            free_cb.push_back(1);
        }
        rows_done += pass_rows;
    }
}
