// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v2) compute: resident-row online softmax. Up to R_MAX 64-token query rows of
// one head stay resident in L1 (Q, O accumulator, running max and sum) while the union of their
// listed KV blocks streams through ONCE in ascending block order. Each arriving block visits every
// resident row that lists it; online softmax is order-independent, so this computes exactly the
// per-row block-mask attention of the v1 kernel (bf16 rounding order aside).
//
// Resident and stream CBs are used as RAM: reserved/held once, indexed absolutely, never pushed.
// Ordering with the dataflow RISCs comes from the ctrl/credit CBs (push-release / pop-acquire),
// and packer->unpacker visibility within compute from explicit pack_to_unpack_sync points.

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

// ctrl message layout (uint32 words): {type|flags, row_slot, stream_slot, count}
constexpr uint32_t MSG_VISIT = 0;
constexpr uint32_t MSG_FLUSH = 1;
constexpr uint32_t FLAG_IS_FIRST = 1u << 8;        // row's first visit this pass: init state
constexpr uint32_t FLAG_LAST_OF_BLOCK = 1u << 9;   // free the stream slot after this visit
constexpr uint32_t FLAG_HAS_VMASK = 1u << 10;      // a partial-column mask tile accompanies the block

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);   // key tile-columns per block (2)
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);   // query tile-rows per row (2)
    constexpr uint32_t R_MAX = get_compile_time_arg_val(4); // resident rows per pass
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);

    constexpr uint32_t cb_q_res = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(8);
    constexpr uint32_t cb_v_stream = get_compile_time_arg_val(9);
    constexpr uint32_t cb_o_res = get_compile_time_arg_val(10);
    constexpr uint32_t cb_max_res = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sum_res = get_compile_time_arg_val(12);
    constexpr uint32_t cb_maxtmp = get_compile_time_arg_val(13);
    constexpr uint32_t cb_psum = get_compile_time_arg_val(14);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(15);
    constexpr uint32_t cb_qk = get_compile_time_arg_val(16);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(17);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(18);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(19);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(20);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(21);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(22);
    constexpr uint32_t cb_free = get_compile_time_arg_val(23);
    constexpr uint32_t cb_qdone = get_compile_time_arg_val(24);
    constexpr uint32_t cb_out = get_compile_time_arg_val(25);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(26);

    constexpr uint32_t k_tiles_per_block = Skt * DHt;
    constexpr uint32_t v_tiles_per_block = Skt * vDHt;
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;

    const uint32_t row_count = get_arg_val<uint32_t>(0);

    CircularBuffer ctrl_cb(cb_ctrl), free_cb(cb_free), qdone_cb(cb_qdone), out_cb(cb_out);
    CircularBuffer qk_cb(cb_qk), psum_cb(cb_psum), maxtmp_cb(cb_maxtmp), corr_cb(cb_corr);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_res, cb_k_stream, cb_qk);
    matmul_init(cb_q_res, cb_k_stream);

    // RAM-mode CBs: reserve the whole capacity once and hold; read/write via absolute tile indices.
    CircularBuffer(cb_max_res).reserve_back(R_MAX * Sqt);
    cb_push_back_hold_wr_ptr(cb_max_res, R_MAX * Sqt);  // fronted so helpers' wait_front is satisfied
    CircularBuffer(cb_sum_res).reserve_back(R_MAX * Sqt);
    cb_push_back_hold_wr_ptr(cb_sum_res, R_MAX * Sqt);
    CircularBuffer(cb_o_res).reserve_back(R_MAX * Sqt * vDHt);
    cb_push_back_hold_wr_ptr(cb_o_res, R_MAX * Sqt * vDHt);
    qk_cb.reserve_back(Sqt * Skt);
    cb_push_back_hold_wr_ptr(cb_qk, Sqt * Skt);
    maxtmp_cb.reserve_back(Sqt);
    cb_push_back_hold_wr_ptr(cb_maxtmp, Sqt);
    psum_cb.reserve_back(Sqt);
    cb_push_back_hold_wr_ptr(cb_psum, Sqt);
    corr_cb.reserve_back(Sqt);
    cb_push_back_hold_wr_ptr(cb_corr, Sqt);
    // Stream CBs are filled by NoC on the dataflow side; front them so absolute unpacks are legal.
    CircularBuffer(cb_k_stream).reserve_back(stream_depth * k_tiles_per_block);
    cb_push_back_hold_wr_ptr(cb_k_stream, stream_depth * k_tiles_per_block);
    CircularBuffer(cb_v_stream).reserve_back(stream_depth * v_tiles_per_block);
    cb_push_back_hold_wr_ptr(cb_v_stream, stream_depth * v_tiles_per_block);
    CircularBuffer(cb_q_res).reserve_back(R_MAX * Sqt * DHt);
    cb_push_back_hold_wr_ptr(cb_q_res, R_MAX * Sqt * DHt);

    CircularBuffer(cb_scale).wait_front(1);
    CircularBuffer(cb_col_identity).wait_front(1);
    CircularBuffer(cb_neginf).wait_front(1);

    // Hand the dataflow side its initial stream-slot credits.
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
            const uint32_t row_slot = ckernel::read_tile_value(cb_ctrl, 0, 1);
            const uint32_t slot = ckernel::read_tile_value(cb_ctrl, 0, 2);
            const uint32_t count = ckernel::read_tile_value(cb_ctrl, 0, 3);
            ctrl_cb.pop_front(1);
            const uint32_t type = w0 & 0xff;

            if (type == MSG_FLUSH) {
                // normalize: per-row scalar sum via matmul against col-identity, reciprocal, O *= 1/sum
                const uint32_t o_base = row_slot * Sqt * vDHt;
                constexpr uint32_t N = 1;
                out_cb.reserve_back(Sqt * vDHt);
                for (uint32_t s = 0; s < Sqt; ++s) {
                    // per-iteration: the mul_bcast below reconfigures the unpacker away from matmul
                    matmul_block_init(cb_sum_res, cb_col_identity, 0, N, 1, N);
                    reconfig_data_format(cb_sum_res, cb_col_identity);
                    CircularBuffer(cb_recip_scratch).reserve_back(1);
                    tile_regs_acquire();
                    matmul_block(cb_sum_res, cb_col_identity, row_slot * Sqt + s, 0, 0, 0, N, 1, N);
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

            // ---- visit ----
            const bool is_first = (w0 & FLAG_IS_FIRST) != 0;
            const bool last_of_block = (w0 & FLAG_LAST_OF_BLOCK) != 0;
            const bool has_vmask = (w0 & FLAG_HAS_VMASK) != 0;
            const bool ragged = count < block_size;

            // Phase 1: QK -> qk scratch (held; in-place mask + sub_exp follow)
            reconfig_data_format(cb_k_stream, cb_q_res);
            exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
            mm_no_mop_init_short(cb_q_res, cb_k_stream, /*transpose=*/true, 1, Sqt, DHt);
            pack_reconfig_data_format(cb_qk);
            configure_row_pack_width(cb_qk, 1);
            for (uint32_t kt = 0; kt < Skt; ++kt) {
                blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/Skt>(
                    cb_q_res,
                    cb_k_stream,
                    cb_qk,
                    /*in0_index_start=*/row_slot * Sqt * DHt,
                    /*in1_index_start=*/slot * k_tiles_per_block + kt * DHt,
                    /*row_subblock_idx=*/0,
                    /*out_col_offset=*/kt,
                    /*subblock_w=*/1,
                    /*subblock_h=*/Sqt,
                    /*inner_dim=*/DHt,
                    /*matmul_stride=*/DHt,
                    /*skip_pack_configure=*/true);
            }
            stream_pack_to_unpack_sync();

            // ragged block: -inf the pad columns before the max reduce
            if (ragged) {
                const uint32_t boundary_tile = count / keys_per_tile;
                const uint32_t boundary_col = count % keys_per_tile;
                uint32_t first_full = boundary_tile;
                if (boundary_col != 0) {
                    if (has_vmask) {
                        CircularBuffer(cb_vmask).wait_front(1);
                        apply_partial_mask_lightweight(cb_vmask, 0, cb_qk, boundary_tile, Skt, Sqt, 0);
                    }
                    first_full = boundary_tile + 1;
                }
                for (uint32_t kt = first_full; kt < Skt; ++kt) {
                    apply_partial_mask_lightweight(cb_neginf, 0, cb_qk, kt, Skt, Sqt, 0);
                }
                stream_pack_to_unpack_sync();
            }

            // Phase 2: row max of this block (eltwise max vs the resident max unless first)
            reconfig_data_format(cb_qk, cb_scale);
            {
                tile_regs_acquire();
                if (!is_first) {
                    sdpa_reduce_copy_tile_to_dst_init_short(cb_max_res);
                    for (uint32_t i = 0; i < Sqt; ++i) {
                        copy_tile(cb_max_res, row_slot * Sqt + i, i);
                    }
                }
                reduce_block_max_row_init<Skt>(cb_maxtmp);
                for (uint32_t i = 0; i < Sqt; ++i) {
                    reduce_block_max_row<Skt>(cb_qk, cb_scale, i * Skt, i);
                }
                reduce_block_max_row_uninit(cb_qk);
                tile_regs_commit();
                tile_regs_wait();
                configure_single_tile_pack(cb_maxtmp);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t i = 0; i < Sqt; ++i) {
                    pack_tile<true>(i, cb_maxtmp, i);
                }
                tile_regs_release();
            }
            stream_pack_to_unpack_sync();

            // Phase 3: probs = exp((qk - max) * scale) in place; partial row-sum -> cb_psum
            sub_exp_block_bcast_cols<false, scale_fp32>(
                cb_qk, cb_maxtmp, cb_psum, /*cols_in_row=*/Skt, /*q_subblock=*/0, /*global_col_base=*/0, Sqt, Skt);

            if (is_first) {
                stream_pack_to_unpack_sync();
                // state init: sum := psum, max := tmp (copy via DEST)
                copy_tile_init(cb_psum);
                reconfig_data_format_srca(cb_psum);
                tile_regs_acquire();
                for (uint32_t i = 0; i < Sqt; ++i) {
                    copy_tile(cb_psum, i, i);
                }
                copy_tile_init(cb_maxtmp);
                for (uint32_t i = 0; i < Sqt; ++i) {
                    copy_tile(cb_maxtmp, i, Sqt + i);
                }
                tile_regs_commit();
                tile_regs_wait();
                configure_single_tile_pack(cb_sum_res);
                PACK((llk_pack_reconfig_l1_acc(0)));
                for (uint32_t i = 0; i < Sqt; ++i) {
                    pack_tile<true>(i, cb_sum_res, row_slot * Sqt + i);
                }
                configure_single_tile_pack(cb_max_res);
                for (uint32_t i = 0; i < Sqt; ++i) {
                    pack_tile<true>(Sqt + i, cb_max_res, row_slot * Sqt + i);
                }
                tile_regs_release();
            } else {
                // corr = exp((old_max - new_max) * scale), per-row column vector
                exp_packthread_tile_init<EXP_APPROX_MODE>();
                sub_init(cb_max_res, cb_maxtmp);
                {
                    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < Sqt; ++i) {
                        sub_tiles(cb_max_res, cb_maxtmp, row_slot * Sqt + i, i, i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < Sqt; ++i) {
                        PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(i)));
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    configure_single_tile_pack(cb_corr);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t i = 0; i < Sqt; ++i) {
                        pack_tile<true>(i, cb_corr, i);
                    }
                    tile_regs_release();
                }
                stream_pack_to_unpack_sync();

                // O[row] *= corr (in place); sum[row] = sum[row] * corr, then += psum; max := tmp
                mul_bcast_cols_init(cb_o_res, cb_corr);
                reconfig_data_format(cb_o_res, cb_corr);
                const uint32_t o_base = row_slot * Sqt * vDHt;
                for (uint32_t i = 0; i < Sqt; ++i) {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        mul_tiles_bcast_cols(cb_o_res, cb_corr, o_base + i * vDHt + j, i, j);
                    }
                    mul_tiles_bcast_cols(cb_sum_res, cb_corr, row_slot * Sqt + i, i, vDHt);
                    tile_regs_commit();
                    tile_regs_wait();
                    configure_row_pack_width(cb_o_res, 1);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    for (uint32_t j = 0; j < vDHt; ++j) {
                        pack_tile<true>(j, cb_o_res, o_base + i * vDHt + j);
                    }
                    configure_single_tile_pack(cb_sum_res);
                    pack_tile<true>(vDHt, cb_sum_res, row_slot * Sqt + i);
                    tile_regs_release();
                }
                // sum += psum and max := tmp (copies through DEST, psum with packer L1-acc)
                copy_tile_init(cb_psum);
                reconfig_data_format_srca(cb_psum);
                tile_regs_acquire();
                for (uint32_t i = 0; i < Sqt; ++i) {
                    copy_tile(cb_psum, i, i);
                }
                copy_tile_init(cb_maxtmp);
                for (uint32_t i = 0; i < Sqt; ++i) {
                    copy_tile(cb_maxtmp, i, Sqt + i);
                }
                tile_regs_commit();
                tile_regs_wait();
                configure_single_tile_pack(cb_sum_res);
                PACK((llk_pack_reconfig_l1_acc(1)));
                for (uint32_t i = 0; i < Sqt; ++i) {
                    pack_tile<true>(i, cb_sum_res, row_slot * Sqt + i);
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
                configure_single_tile_pack(cb_max_res);
                for (uint32_t i = 0; i < Sqt; ++i) {
                    pack_tile<true>(Sqt + i, cb_max_res, row_slot * Sqt + i);
                }
                tile_regs_release();
            }
            stream_pack_to_unpack_sync();

            // Phase 4: O[row] += probs @ V (packer L1-acc; plain overwrite on the first visit)
            reconfig_data_format(cb_v_stream, cb_qk);
            mm_no_mop_init_short(cb_qk, cb_v_stream, /*transpose=*/false, 1, Sqt, Skt);
            configure_row_pack_width(cb_o_res, 1);
            PACK((llk_pack_reconfig_l1_acc(is_first ? 0 : 1)));
            for (uint32_t vd = 0; vd < vDHt; ++vd) {
                blocked_matmul_and_pack<false, /*in1_stride=*/vDHt, /*out_num_cols=*/vDHt>(
                    cb_qk,
                    cb_v_stream,
                    cb_o_res,
                    /*in0_index_start=*/0,
                    /*in1_index_start=*/slot * v_tiles_per_block + vd,
                    /*row_subblock_idx=*/row_slot,
                    /*out_col_offset=*/vd,
                    /*subblock_w=*/1,
                    /*subblock_h=*/Sqt,
                    /*inner_dim=*/Skt,
                    /*matmul_stride=*/Skt,
                    /*skip_pack_configure=*/true);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
            reconfig_data_format_srca(cb_qk);

            if (last_of_block) {
                if (has_vmask) {
                    CircularBuffer(cb_vmask).pop_front(1);
                }
                free_cb.reserve_back(1);
                free_cb.push_back(1);
            }
        }
        rows_done += pass_rows;
    }
    (void)dst_size;
}
