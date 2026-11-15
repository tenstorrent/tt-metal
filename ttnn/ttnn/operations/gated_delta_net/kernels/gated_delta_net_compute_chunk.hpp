// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// R-CHUNK chunk-step blocks: the local chunk step (gram, head prep, state part on one core) and the R-PREP prep
// core's head blocks and state core's prepped state part.

#include "gated_delta_net_compute_common.hpp"

namespace {

// ---------------------------------------------------------------- one chunk step (R-CHUNK)
// [Kn Kn^T; Qn Kn^T] -> cb_gram (2 tiles, left waited for head_prep); per key head, shared by its value heads.
ALWI void gram_block(Buffers& bufs) {
    DataflowBuffer gram{cb_gram};
    DataflowBuffer knT{cb_knT};
    ckl::matmul_block<
        false,
        false,
        ckl::LastBlockTarget::Out,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndRetainOnLastBlock,
        ckl::InputPolicy::NoWaitNoPop>(bufs.kq_rows, knT, gram, gram, ckl::MatmulBlockShape::of(2, 1, 1, 1, Kt, 1));
    cb_wait_front(cb_gram, 2);
}

// One value head's chunk operands from g and beta (column 0 of the writer's coefficient pages, 0 on rows that are
// no token, so c is flat there and those rows neither decay nor write) and the gram tiles. row_last: tile-local
// last token row. D, c, exp(c), exp(c_last - c), beta exp(c) -> cb_chunk_vec (every column of row s holds row s's
// value); L, QD -> cb_chunk_mat (local) or, on a prep core, the head's block of cb_prep_send (head_block).
// head_prep_vectors needs only the gates; head_prep_matrices then the gram tiles.
ALWI void head_prep_vectors(uint32_t row_last) {
    cb_wait_front(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
    cb_reserve_back(cb_chunk_vec, 2);
    tile_regs_acquire();
    begin_loads(cb_tok_coef);
    load(cb_tok_coef, CHUNK_G, 0);
    load(cb_tri_mask, 0, 3);
    cumsum_tile_init();
    cumsum_tile(0, true);  // column 0: c[s] = sum_{s' <= s} g[s'] (fp32)
    fill_tile_init();
    fill_tile(1, 0.0f);
    sfpu_bcast_col_init();
    sfpu_add_bcast_col(1, 0);  // C[s, s'] = c[s]
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(1, 2);
    wait_sfpu_before_fpu();
    transpose_dest_init<true>(cb_tok_coef);
    transpose_dest<true>(2);  // C^T[s, s'] = c[s']
    sub_binary_tile_init();
    sub_binary_tile(2, 1, 0);  // c[s'] - c[s]: >= 0 on and below the diagonal (g <= 0)
    relu_tile_init();
    relu_tile(0);  // the upper triangle (masked below) must not reach exp with a positive argument
    negative_tile_init();
    negative_tile(0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    mul_binary_tile_init();
    mul_binary_tile(0, 3, 0);  // D
    tile_regs_commit();
    tile_regs_wait();
    pack_first(0, cb_chunk_vec, CV_D);
    pack_at(1, cb_chunk_vec, CV_C);
    tile_regs_release();
    cb_push_back(cb_chunk_vec, 2);

    cb_wait_front(cb_chunk_vec, 2);
    const uint32_t c_last_bits = read_tile_value(cb_chunk_vec, CV_C, elem_index(row_last, 0));
    cb_reserve_back(cb_chunk_vec, GDN_CHUNK_VEC_PAGES - 2);
    tile_regs_acquire();
    begin_loads(cb_chunk_vec);
    load(cb_chunk_vec, CV_C, 0);
    load(cb_chunk_vec, CV_C, 1);
    load(cb_tok_coef, CHUNK_BETA, 2);
    load(cb_chunk_vec, CV_C, 3);
    binop_with_scalar_tile_init();
    rsub_unary_tile(1, c_last_bits);  // c_last - c[s] (<= 0 on token rows)
    exp_tile_init<false>();
    exp_tile<false>(0);  // exp(c)
    exp_tile<false>(1);  // exp(c_last - c)
    exp_tile<false>(3);
    sfpu_bcast_col_init();
    sfpu_mul_bcast_col(3, 2);  // beta exp(c)
    tile_regs_commit();
    tile_regs_wait();
    pack_first(0, cb_chunk_vec, CV_EC - 2);
    pack_at(1, cb_chunk_vec, CV_W - 2);
    pack_at(3, cb_chunk_vec, CV_BEC - 2);
    if constexpr (PREP) {
        cb_reserve_back(cb_prep_send, HB_M);  // exp(c), QD
        pack_at(0, cb_prep_send, HB_EC);
    }
    tile_regs_release();
    cb_push_back(cb_chunk_vec, GDN_CHUNK_VEC_PAGES - 2);
}

ALWI void head_prep_matrices() {
    cb_reserve_back(cb_chunk_mat, PREP ? 1 : GDN_CHUNK_MAT_PAGES);

    // L and QD: fp32 SFPU products of the gram tiles with D and beta.
    tile_regs_acquire();
    begin_loads(cb_gram);
    load(cb_gram, 0, 0);
    load(cb_gram, 1, 1);
    load(cb_chunk_vec, CV_D, 2);
    load(cb_tok_coef, CHUNK_BETA, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 2, 1);
    sfpu_bcast_col_init();
    sfpu_mul_bcast_col(0, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_first(0, cb_chunk_mat, CM_L);
    if constexpr (PREP) {
        pack_at(1, cb_prep_send, HB_QD);
        cb_push_back(cb_prep_send, HB_M);
    } else {
        pack_at(1, cb_chunk_mat, CM_QD);
    }
    tile_regs_release();
    cb_push_back(cb_chunk_mat, PREP ? 1 : GDN_CHUNK_MAT_PAGES);
    cb_wait_front(cb_chunk_vec, GDN_CHUNK_VEC_PAGES);
}

ALWI void head_prep(uint32_t row_last) {
    head_prep_vectors(row_last);
    head_prep_matrices();
}

// R-PREP prep core, after head_prep: the rest of value head `head`'s block. M = (I + L)^-1 (fp32 forward substitution
// on the identity), then the state part's row scalings folded into its operands, each an fp32 SFPU product:
// beta exp(c) (*) kn and exp(c) (*) qn (the read [kq] x S then yields beta exp(c) Rk and exp(c) Rq),
// kn^T (*) exp(c_last - c)^T (the update then multiplies U directly) and beta (*) v.
ALWI void head_block(uint32_t head) {
    cb_wait_front(cb_chunk_mat, 1);
    cb_reserve_back(cb_chunk_vec, 1);
    tile_regs_acquire();
    begin_loads(cb_tri_mask);
    load(cb_tri_mask, 1, 0);
    load(cb_chunk_vec, CV_W, 2);
    triangle_solve_tile_init();
    triangle_solve_tile<DataFormat::Float32>(cb_chunk_mat, CM_L, 0, 1);
    // exp(c_last - c)^T, kept fp32 (exp(c_last - c) is constant along a row of the vector tiles) ->
    // cb_chunk_vec[CV_WT].
    wait_sfpu_before_fpu();
    transpose_dest_init<true>(cb_chunk_vec);
    transpose_dest<true>(2);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_prep_send, 1);
    pack_first(1, cb_prep_send, 0);  // HB_M (pack indices count from the reservation)
    pack_at(2, cb_chunk_vec, 0);     // pack indices count from the reservation
    tile_regs_release();
    cb_pop_front(cb_chunk_mat, 1);
    cb_push_back(cb_chunk_vec, 1);
    cb_push_back(cb_prep_send, 1);

    cb_reserve_back(cb_prep_send, 2 * Kt);

    // [beta exp(c) (*) kn, exp(c) (*) qn], one key column tile of each per window. kn, qn, kn^T load through srcA
    // (19-bit, as every matmul reads them); the scalings stay fp32.
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        tile_regs_acquire();
        begin_loads(cb_chunk_vec);
        load(cb_chunk_vec, CV_BEC, 1);
        load(cb_chunk_vec, CV_EC, 3);
        begin_loads(cb_kq_rows);
        load(cb_kq_rows, kt, 0);
        load(cb_kq_rows, Kt + kt, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, kt);
        pack_at(2, cb_prep_send, Kt + kt);
        tile_regs_release();
    }
    cb_push_back(cb_prep_send, 2 * Kt);
    cb_reserve_back(cb_prep_send, Kt);
    cb_wait_front(cb_chunk_vec, CV_WT + 1);
    constexpr uint32_t BATCH = ckl::DEST_AUTO_LIMIT - 1;  // data tiles per window beside the scaling tile
    for (uint32_t kt0 = 0; kt0 < Kt; kt0 += BATCH) {
        const uint32_t n = (Kt - kt0) < BATCH ? (Kt - kt0) : BATCH;
        tile_regs_acquire();
        begin_loads(cb_chunk_vec);
        load(cb_chunk_vec, CV_WT, BATCH);
        begin_loads(cb_knT);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_knT, kt0 + i, i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            mul_binary_tile(i, BATCH, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, kt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_prep_send, kt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_prep_send, Kt);
    cb_reserve_back(cb_prep_send, HEAD_V_TILES);
    for (uint32_t vt0 = 0; vt0 < HEAD_V_TILES; vt0 += BATCH) {
        const uint32_t n = (HEAD_V_TILES - vt0) < BATCH ? (HEAD_V_TILES - vt0) : BATCH;
        tile_regs_acquire();
        begin_loads(cb_v_prep);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_v_prep, head * HEAD_V_TILES + vt0 + i, i);
        }
        load(cb_tok_coef, CHUNK_BETA, BATCH);
        sfpu_bcast_col_init();
        for (uint32_t i = 0; i < n; ++i) {
            sfpu_mul_bcast_col(i, BATCH);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_prep_send, vt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_prep_send, HEAD_V_TILES);
    cb_pop_front(cb_chunk_vec, CV_WT + 1);
    cb_pop_front(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
}

// The chunk step's state part on one core (local roles), on the head operands in cb_chunk_vec / cb_chunk_mat /
// cb_tok_coef and the kn / qn, kn^T, v tiles (waited by the caller); decay_bits = exp(c_last). Pops only its own
// intermediates.
ALWI void chunk_state(Buffers& bufs, bool first_in_pair, bool last_in_pair, uint32_t decay_bits) {
    read_state<cb_kq_rows>(bufs, first_in_pair);

    // U = (I + L)^-1 (beta V - beta exp(c) Rk), one fp32 SFPU solve per column tile.
    cb_wait_front(cb_reads, 2 * Vb);
    cb_wait_front(cb_chunk_mat, GDN_CHUNK_MAT_PAGES);
    cb_reserve_back(cb_u, Vb);
    cb_reserve_back(cb_u_f32, Vb);
    for (uint32_t vt = 0; vt < Vb; ++vt) {
        tile_regs_acquire();
        begin_loads(cb_v);
        load(cb_v, vt, 0);
        load(cb_reads, vt, 1);
        load(cb_tok_coef, CHUNK_BETA, 2);
        load(cb_chunk_vec, CV_BEC, 3);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        sfpu_mul_bcast_col(1, 3);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        triangle_solve_tile_init();
        triangle_solve_tile<DataFormat::Float32>(cb_chunk_mat, CM_L, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(1, cb_u, vt);
        tile_regs_release();
    }
    cb_push_back(cb_u, Vb);
    cb_push_back(cb_u_f32, Vb);
    cb_wait_front(cb_u, Vb);
    cb_wait_front(cb_u_f32, Vb);

    // Uw = exp(c_last - c) (*)col U (the update operand) and O = exp(c) (*)col Rq + QD x U, in batches
    // of DEST_AUTO_LIMIT - 1 column tiles plus the broadcast vector.
    constexpr uint32_t VEC_BATCH = ckl::DEST_AUTO_LIMIT - 1;
    cb_reserve_back(cb_uw, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += VEC_BATCH) {
        const uint32_t n = (Vb - vt0) < VEC_BATCH ? (Vb - vt0) : VEC_BATCH;
        tile_regs_acquire();
        begin_loads(cb_u_f32);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_u_f32, vt0 + i, i);
        }
        load(cb_chunk_vec, CV_W, VEC_BATCH);
        sfpu_bcast_col_init();
        for (uint32_t i = 0; i < n; ++i) {
            sfpu_mul_bcast_col(i, VEC_BATCH);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_uw, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_uw, vt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_uw, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += VEC_BATCH) {
        const uint32_t n = (Vb - vt0) < VEC_BATCH ? (Vb - vt0) : VEC_BATCH;
        tile_regs_acquire();
        begin_loads(cb_reads);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_reads, Vb + vt0 + i, i);
        }
        load(cb_chunk_vec, CV_EC, VEC_BATCH);
        sfpu_bcast_col_init();
        for (uint32_t i = 0; i < n; ++i) {
            sfpu_mul_bcast_col(i, VEC_BATCH);
        }
        wait_sfpu_before_fpu();
        reconfig_data_format(cb_u, cb_chunk_mat);
        matmul_init(cb_chunk_mat, cb_u);
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_chunk_mat, cb_u, CM_QD, vt0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_o_acc, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_o_acc, vt0 + i);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_reads, 2 * Vb);

    cb_wait_front(cb_uw, Vb);
    update_state<cb_knT>(first_in_pair, last_in_pair, cb_uw, decay_bits);
    cb_pop_front(cb_uw, Vb);
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
}

// R-PREP state core: the state part on the prescaled inbox operands (head_block), no broadcasts:
//   [beta exp(c) Rk; exp(c) Rq] = [kq] x S,  U = M (beta V - beta exp(c) Rk),  O = exp(c) Rq + QD x U,
//   S <- exp(c_last) S + (kn^T (*) exp(c_last - c)^T) x U.
ALWI void chunk_state_prepped(Buffers& bufs, bool first_in_pair, bool last_in_pair, uint32_t decay_bits) {
    read_state<cb_in_kq>(bufs, first_in_pair);
    cb_wait_front(cb_reads, 2 * Vb);
    cb_wait_front(cb_in_mat, GDN_IN_MAT_PAGES);
    cb_wait_front(cb_in_v, Vb);
    constexpr uint32_t PAIRS = ckl::DEST_AUTO_LIMIT / 2;  // rhs column tiles per window (operand pairs)
    cb_reserve_back(cb_uw, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += PAIRS) {
        const uint32_t n = (Vb - vt0) < PAIRS ? (Vb - vt0) : PAIRS;
        tile_regs_acquire();
        begin_loads(cb_in_v);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_in_v, vt0 + i, 2 * i);
            load(cb_reads, vt0 + i, 2 * i + 1);
        }
        sub_binary_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            sub_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_uw);
        for (uint32_t i = 0; i < n; ++i) {
            pack_at(2 * i, cb_uw, vt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_uw, Vb);
    cb_wait_front(cb_uw, Vb);
    cb_reserve_back(cb_u, Vb);
    cb_reserve_back(cb_u_f32, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        reconfig_data_format(cb_uw, cb_in_mat);
        matmul_init(cb_in_mat, cb_uw);
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_in_mat, cb_uw, IN_MAT_M, vt0 + i, i);  // M x rhs
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_u, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_u, vt0 + i);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_uw, Vb);
    cb_push_back(cb_u, Vb);
    cb_push_back(cb_u_f32, Vb);
    cb_wait_front(cb_u, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        begin_loads(cb_reads);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_reads, Vb + vt0 + i, i);
        }
        reconfig_data_format(cb_u, cb_in_mat);
        matmul_init(cb_in_mat, cb_u);
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_in_mat, cb_u, IN_MAT_QD, vt0 + i, i);  // + QD x U onto exp(c) Rq
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_o_acc, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_o_acc, vt0 + i);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_reads, 2 * Vb);
    update_state<cb_in_knt>(first_in_pair, last_in_pair, cb_u, decay_bits);
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
}

// Prep and state part on one core (local roles).
ALWI void chunk_step(Buffers& bufs, bool first_in_pair, bool last_in_pair, uint32_t row_last) {
    gram_block(bufs);
    head_prep(row_last);
    cb_pop_front(cb_gram, 2);
    chunk_state(bufs, first_in_pair, last_in_pair, read_tile_value(cb_chunk_vec, CV_EC, elem_index(row_last, 0)));
    cb_pop_front(cb_chunk_mat, GDN_CHUNK_MAT_PAGES);
    cb_pop_front(cb_chunk_vec, GDN_CHUNK_VEC_PAGES);
    cb_pop_front(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
}

}  // namespace
