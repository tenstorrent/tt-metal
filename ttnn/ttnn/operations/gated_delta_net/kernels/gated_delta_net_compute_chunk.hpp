// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// R-CHUNK chunk-step blocks: the local chunk step (gram, head prep, state part on one core) and the R-PREP prep
// core's head blocks and state core's prepped state part.

#include "gated_delta_net_compute_common.hpp"

namespace {

// R-PREP (host STATE_FUSED_RHS): the prep core sends -(beta exp(c)) (*) kn, so that the state core's k read
// accumulates onto beta V in DEST (the rhs in one window) and its q read is the start of the output's window.
// The writer then adds a coefficient page with -beta.
constexpr bool FUSED_RHS = GDN_STATE_FUSED_RHS != 0;
static_assert(!HOLD || FUSED_RHS, "STATE_HOLD counts the fused step's windows");
constexpr uint32_t CHUNK_BEC_BETA = PREP && FUSED_RHS ? 2 : CHUNK_BETA;
// Host PREP_SCALE_FPU, a mask: 1 beta exp(c) (*) kn, 2 exp(c) (*) qn, 4 kn^T (*) exp(c_last - c)^T as FPU products.
constexpr uint32_t SCALE_FPU = PREP ? GDN_PREP_SCALE_FPU : 0;
constexpr bool FPU_KN = SCALE_FPU % 2 == 1;
constexpr bool FPU_QN = SCALE_FPU / 2 % 2 == 1;
constexpr bool FPU_KNT = SCALE_FPU / 4 % 2 == 1;
// Host PREP_FPU_UNBIAS_BITS: the mean truncation loss of the FPU operands on the way to the state, scaled back in.
constexpr uint32_t FPU_UNBIAS_BITS = GDN_FPU_UNBIAS_BITS;
// Host PREP_M_BETA: beta as the columns of M (the solve runs on the writer's diag(beta) page), so v and kn go to the
// state core without their beta scaling (U = M diag(beta) (V - exp(c) Rk)): the vector for kn is exp(c) with
// FUSED_RHS's sign.
constexpr uint32_t CHUNK_DIAG_BETA = 2;
constexpr bool M_BETA = PREP && GDN_PREP_M_BETA != 0;
// Host STATE_FOLD: M folded into the state core's operands, so its step has no rhs and U and O do not wait for one
// another: U = M x V + Mk x S, O = (QD M) x V + q' x S with Mk = (M (*) -exp(c)^T) x kn and
// q' = exp(c) (*) qn + (QD (M (*) -exp(c)^T)) x kn from the prep core. The head block keeps its pages
// [exp(c) | QD M, M | Mk, q' | kn^T (*) w^T | v].
constexpr bool FOLD = GDN_STATE_FOLD;
static_assert(!FOLD || !PREP || (M_BETA && FPU_KN && FPU_QN), "STATE_FOLD folds the FPU-scaled operands");
constexpr uint32_t CM_M = 2;  // FOLD prep core: M, M (*) -exp(c)^T, QD x (M (*) -exp(c)^T) after L and QD
constexpr uint32_t CM_MT = 3;
constexpr uint32_t CM_QDMT = 4;
constexpr uint32_t FOLD_MAT_PAGES = 5;
// Host PREP_VEC_COL0 (folded prep core): exp(c), exp(c_last - c) and -exp(c) only in column 0 of their tiles (8 of
// 32 SFPU vectors each); their readers broadcast it: the FPU products by column / by row of the transposed tile, the
// M~ product by the row of the transposed tile.
constexpr bool VEC_COL0 = PREP && FOLD && FPU_KNT && GDN_VEC_COL0 != 0;
constexpr uint32_t KN_SCALE_BITS = (FOLD     ? GDN_FOLD_UNBIAS_BITS
                                    : FPU_KN ? FPU_UNBIAS_BITS
                                             : ONE_BITS) |
                                   (FUSED_RHS ? 0x80000000u : 0u);

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
    GDN_MARK("m_hv1");

    cb_wait_front(cb_chunk_vec, 2);
    const uint32_t c_last_bits = read_tile_value(cb_chunk_vec, CV_C, elem_index(row_last, 0));
    cb_reserve_back(cb_chunk_vec, GDN_CHUNK_VEC_PAGES - 2);
    tile_regs_acquire();
    begin_loads(cb_chunk_vec);
    load(cb_chunk_vec, CV_C, 0);
    load(cb_chunk_vec, CV_C, 1);
    if constexpr (!M_BETA) {
        load(cb_tok_coef, CHUNK_BEC_BETA, 2);
        load(cb_chunk_vec, CV_C, 3);
    }
    if constexpr (VEC_COL0) {
#ifdef TRISC_MATH
        binop_with_scalar_tile_init();
        sfpu_col0(
            1, [=] { ckernel::sfpu::calculate_binop_with_scalar<APPROX, RSUB_UNARY, 1, DST_ACCUM_MODE>(c_last_bits); });
        exp_tile_init<false>();
        sfpu_col0(0, [] { ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 1>(); });
        sfpu_col0(1, [] { ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 1>(); });
#endif
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 3);
#ifdef TRISC_MATH
        binop_with_scalar_tile_init();
        sfpu_col0(
            3, [] { ckernel::sfpu::calculate_binop_with_scalar<APPROX, MUL_UNARY, 1, DST_ACCUM_MODE>(KN_SCALE_BITS); });
#endif
    } else {
        binop_with_scalar_tile_init();
        rsub_unary_tile(1, c_last_bits);  // c_last - c[s] (<= 0 on token rows)
        exp_tile_init<false>();
        exp_tile<false>(0);  // exp(c)
        exp_tile<false>(1);  // exp(c_last - c)
    }
    if constexpr (VEC_COL0) {
    } else if constexpr (M_BETA) {
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 3);
        binop_with_scalar_tile_init();
        mul_unary_tile(3, KN_SCALE_BITS);  // (-)exp(c)
    } else {
        exp_tile<false>(3);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(3, 2);  // beta exp(c) (FUSED_RHS prep core: -beta exp(c))
        if constexpr (FPU_KN) {
            binop_with_scalar_tile_init();
            mul_unary_tile(3, FPU_UNBIAS_BITS);
        }
    }
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
    GDN_MARK("m_hv2");
}

ALWI void head_prep_matrices() {
    constexpr bool QD_LOCAL = !PREP || FOLD;  // QD stays on the core (FOLD: the operand of QD x M)
    cb_reserve_back(cb_chunk_mat, QD_LOCAL ? GDN_CHUNK_MAT_PAGES : 1);

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
    if constexpr (QD_LOCAL) {
        pack_at(1, cb_chunk_mat, CM_QD);
    } else {
        pack_at(1, cb_prep_send, HB_QD);
        cb_push_back(cb_prep_send, HB_M);
    }
    tile_regs_release();
    cb_push_back(cb_chunk_mat, QD_LOCAL ? GDN_CHUNK_MAT_PAGES : 1);
    cb_wait_front(cb_chunk_vec, GDN_CHUNK_VEC_PAGES);
    GDN_MARK("m_hm");
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
    cb_wait_front(cb_chunk_mat, FOLD ? GDN_CHUNK_MAT_PAGES : 1);
    cb_reserve_back(cb_chunk_vec, 1);
    if constexpr (FOLD) {
        cb_reserve_back(cb_chunk_mat, 2);
    }
    tile_regs_acquire();
    begin_loads(cb_tri_mask);
    if constexpr (M_BETA) {
        load(cb_tok_coef, CHUNK_DIAG_BETA, 0);
    } else {
        load(cb_tri_mask, 1, 0);
    }
    load(cb_chunk_vec, CV_W, 2);
    if constexpr (FOLD) {
        load(cb_chunk_vec, CV_BEC, 3);
    }
    triangle_solve_tile_init();
    triangle_solve_tile<DataFormat::Float32>(cb_chunk_mat, CM_L, 0, 1);
    if constexpr (FPU_KNT) {
        binop_with_scalar_tile_init();
        mul_unary_tile(2, FPU_UNBIAS_BITS);
    }
    // exp(c_last - c)^T, kept fp32 (exp(c_last - c) is constant along a row of the vector tiles) ->
    // cb_chunk_vec[CV_WT].
    wait_sfpu_before_fpu();
    transpose_dest_init<true>(cb_chunk_vec);
    transpose_dest<true>(2);
    if constexpr (FOLD) {
        transpose_dest<true>(3);
        if constexpr (VEC_COL0) {
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 0);
            sfpu_bcast_row_init();
            sfpu_mul_bcast_row(0, 3);  // row 0 of the transposed tile is -exp(c)^T
        } else {
            mul_binary_tile_init();
            mul_binary_tile(1, 3, 0);  // M (*) -exp(c)^T: kn's row scaling as M's column scaling (fp32)
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    if constexpr (FOLD) {
        cb_reserve_back(cb_prep_send, HB_M + 1);  // exp(c) is packed, nothing of the block pushed yet
        pack_first(1, cb_prep_send, HB_M);
        pack_at(1, cb_chunk_mat, 0);  // pack indices count from the reservation
        pack_at(0, cb_chunk_mat, 1);
    } else {
        cb_reserve_back(cb_prep_send, 1);
        pack_first(1, cb_prep_send, 0);  // HB_M (pack indices count from the reservation)
    }
    pack_at(2, cb_chunk_vec, 0);  // pack indices count from the reservation
    tile_regs_release();
    cb_push_back(cb_chunk_vec, 1);
    if constexpr (FOLD) {
        // QD x M (the state core's operand) and QD x (M (*) -exp(c)^T) (for q').
        cb_push_back(cb_chunk_mat, 2);
        cb_wait_front(cb_chunk_mat, CM_QDMT);
        cb_reserve_back(cb_chunk_mat, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_chunk_mat, cb_chunk_mat);
        matmul_init(cb_chunk_mat, cb_chunk_mat);
        matmul_tiles(cb_chunk_mat, cb_chunk_mat, CM_QD, CM_M, 0);
        matmul_tiles(cb_chunk_mat, cb_chunk_mat, CM_QD, CM_MT, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, HB_QD);
        pack_at(1, cb_chunk_mat, 0);
        tile_regs_release();
        cb_push_back(cb_chunk_mat, 1);
        cb_push_back(cb_prep_send, HB_M + 1);
        cb_wait_front(cb_chunk_mat, FOLD_MAT_PAGES);
    } else {
        cb_pop_front(cb_chunk_mat, 1);
        cb_push_back(cb_prep_send, 1);
    }
    GDN_MARK("m_hb_m");

    cb_reserve_back(cb_prep_send, 2 * Kt);
    // FOLD: Mk = (M (*) -exp(c)^T) x kn, then q' = exp(c) (*) qn + (QD (M (*) -exp(c)^T)) x kn, one window each.
    for (uint32_t part = 0; FOLD && part < 2; ++part) {
        tile_regs_acquire();
        if (part == 1) {
            reconfig_data_format(cb_kq_rows, cb_chunk_vec_mm);
            if constexpr (VEC_COL0) {
                mul_bcast_cols_init(cb_kq_rows, cb_chunk_vec_mm);
            } else {
                mul_init(cb_kq_rows, cb_chunk_vec_mm);
            }
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                if constexpr (VEC_COL0) {
                    mul_tiles_bcast_cols(cb_kq_rows, cb_chunk_vec_mm, Kt + kt, CV_EC, kt);
                } else {
                    mul_tiles(cb_kq_rows, cb_chunk_vec_mm, Kt + kt, CV_EC, kt);
                }
            }
        }
        reconfig_data_format(cb_kq_rows, cb_chunk_mat);
        matmul_init(cb_chunk_mat, cb_kq_rows);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            matmul_tiles(cb_chunk_mat, cb_kq_rows, part == 0 ? CM_MT : CM_QDMT, kt, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, part * Kt);
        for (uint32_t i = 1; i < Kt; ++i) {
            pack_at(i, cb_prep_send, part * Kt + i);
        }
        tile_regs_release();
    }
    if constexpr (FOLD) {
        cb_pop_front(cb_chunk_mat, FOLD_MAT_PAGES);
    }

    // [beta exp(c) (*) kn, exp(c) (*) qn]. FPU products: the Kt tiles of kn, then of qn, one window each, the scaling
    // vector read through cb_chunk_vec_mm: the same pages at the indices of cb_chunk_vec, whose read pointer is back
    // at the base at every head (a head pushes and pops exactly the buffer's pages).
    static_assert(Kt <= ckl::DEST_AUTO_LIMIT || SCALE_FPU == 0);
    for (uint32_t part = 0; !FOLD && part < 2; ++part) {
        if (!(part == 0 ? FPU_KN : FPU_QN)) {
            continue;
        }
        tile_regs_acquire();
        reconfig_data_format(cb_kq_rows, cb_chunk_vec_mm);
        mul_init(cb_kq_rows, cb_chunk_vec_mm);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            mul_tiles(cb_kq_rows, cb_chunk_vec_mm, part * Kt + kt, part == 0 ? CV_BEC : CV_EC, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, part * Kt);
        for (uint32_t i = 1; i < Kt; ++i) {
            pack_at(i, cb_prep_send, part * Kt + i);
        }
        tile_regs_release();
    }
    // SFPU products, one key column tile of each per window. kn, qn, kn^T load through srcA (19-bit, as every matmul
    // reads them); the scalings stay fp32.
    for (uint32_t kt = 0; !(FPU_KN && FPU_QN) && kt < Kt; ++kt) {
        tile_regs_acquire();
        begin_loads(cb_chunk_vec);
        load(cb_chunk_vec, CV_BEC, 1);
        load(cb_chunk_vec, CV_EC, 3);
        begin_loads(cb_kq_rows);
        load(cb_kq_rows, kt, 0);
        load(cb_kq_rows, Kt + kt, 2);
        mul_binary_tile_init();
        if constexpr (!FPU_KN) {
            mul_binary_tile(0, 1, 0);
        }
        if constexpr (!FPU_QN) {
            mul_binary_tile(2, 3, 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_prep_send);
        if constexpr (!FPU_KN) {
            pack_at(0, cb_prep_send, kt);
        }
        if constexpr (!FPU_QN) {
            pack_at(2, cb_prep_send, Kt + kt);
        }
        tile_regs_release();
    }
    cb_push_back(cb_prep_send, 2 * Kt);
    GDN_MARK("m_hb_kq");
    cb_reserve_back(cb_prep_send, Kt);
    cb_wait_front(cb_chunk_vec, CV_WT + 1);
    constexpr uint32_t BATCH = ckl::DEST_AUTO_LIMIT - 1;  // data tiles per window beside the scaling tile
    if constexpr (FPU_KNT) {
        tile_regs_acquire();
        reconfig_data_format(cb_knT, cb_chunk_vec_mm);
        if constexpr (VEC_COL0) {
            mul_bcast_rows_init(cb_knT, cb_chunk_vec_mm);
        } else {
            mul_init(cb_knT, cb_chunk_vec_mm);
        }
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            if constexpr (VEC_COL0) {
                mul_tiles_bcast_rows(cb_knT, cb_chunk_vec_mm, kt, CV_WT, kt);
            } else {
                mul_tiles(cb_knT, cb_chunk_vec_mm, kt, CV_WT, kt);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_prep_send, 0);
        for (uint32_t i = 1; i < Kt; ++i) {
            pack_at(i, cb_prep_send, i);
        }
        tile_regs_release();
    }
    for (uint32_t kt0 = 0; !FPU_KNT && kt0 < Kt; kt0 += BATCH) {
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
    GDN_MARK("m_hb_knt");
    cb_reserve_back(cb_prep_send, HEAD_V_TILES);
    constexpr uint32_t V_BATCH = M_BETA ? ckl::DEST_AUTO_LIMIT : BATCH;  // M_BETA: v as it is
    for (uint32_t vt0 = 0; vt0 < HEAD_V_TILES; vt0 += V_BATCH) {
        const uint32_t n = (HEAD_V_TILES - vt0) < V_BATCH ? (HEAD_V_TILES - vt0) : V_BATCH;
        tile_regs_acquire();
        begin_loads(cb_v_prep);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_v_prep, head * HEAD_V_TILES + vt0 + i, i);
        }
        if constexpr (!M_BETA) {
            load(cb_tok_coef, CHUNK_BETA, BATCH);
            sfpu_bcast_col_init();
            for (uint32_t i = 0; i < n; ++i) {
                sfpu_mul_bcast_col(i, BATCH);
            }
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
    GDN_MARK("m_hb_v");
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
// Host SEQ_SPLIT (see the reader): SEGS segments; a state core of a segment 1.. steps the H half of [P | H] and holds
// its tile rows of Oh in cb_reads, a P core (sequences SEGS..) steps its piece of P without the v terms.
constexpr uint32_t SEGS = GDN_SEQ_SPLIT;
constexpr bool SPLIT = SEGS > 1;
static_assert(!SPLIT || !STATE || FOLD, "SEQ_SPLIT runs the folded step");
constexpr uint32_t cb_seg_in = GDN_CB_SEG_IN;
constexpr uint32_t cb_seg_row = GDN_CB_SEG_ROW;
constexpr uint32_t SEG_S = Kt * Kt;  // cb_seg_in: P (Kt x Kt), then the entering state's piece

// STATE_FOLD: column tiles [vt0, vt0 + n) of U (part 0) or O (part 1) onto DEST slots d0.., on the folded operands.
// p_half (SEQ_SPLIT): no v terms.
ALWI void fold_part(uint32_t s_mm, uint32_t part, uint32_t vt0, uint32_t n, uint32_t d0, bool p_half) {
    reconfig_data_format(cb_in_v, cb_in_mat);
    matmul_init(cb_in_mat, cb_in_v);
    for (uint32_t i = 0; !p_half && i < n; ++i) {
        matmul_tiles(cb_in_mat, cb_in_v, part == 0 ? IN_MAT_M : IN_MAT_QD, vt0 + i, d0 + i);
    }
    reconfig_data_format(s_mm, cb_in_kq);
    matmul_init(cb_in_kq, s_mm);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            matmul_tiles(cb_in_kq, s_mm, part * Kt + kt, kt * Vb + vt0 + i, d0 + i);
        }
    }
}

// STATE_FOLD: U -> cb_u and O -> o_cb, one window when both fit a DEST half.
ALWI void state_fold(uint32_t s_mm, bool p_half, uint32_t o_cb) {
    constexpr bool ONE = 2 * Vb <= ckl::DEST_AUTO_LIMIT;
    cb_wait_front(cb_in_mat, GDN_IN_MAT_PAGES);
    cb_wait_front(cb_in_v, Vb);
    cb_reserve_back(cb_u, Vb);
    cb_reserve_back(cb_u_f32, Vb);
    if constexpr (ONE) {
        tile_regs_acquire();
        if constexpr (GDN_FOLD_ORDER != 0) {
            // Both parts' v products, then both parts' state products: one operand switch instead of three.
            reconfig_data_format(cb_in_v, cb_in_mat);
            matmul_init(cb_in_mat, cb_in_v);
            for (uint32_t i = 0; !p_half && i < 2 * Vb; ++i) {
                matmul_tiles(cb_in_mat, cb_in_v, i < Vb ? IN_MAT_M : IN_MAT_QD, i % Vb, i);
            }
            reconfig_data_format(s_mm, cb_in_kq);
            matmul_init(cb_in_kq, s_mm);
            for (uint32_t i = 0; i < 2 * Vb; ++i) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    matmul_tiles(cb_in_kq, s_mm, (i / Vb) * Kt + kt, kt * Vb + i % Vb, i);
                }
            }
        } else {
            fold_part(s_mm, 0, 0, Vb, 0, p_half);
            fold_part(s_mm, 1, 0, Vb, Vb, p_half);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_u, 0);
        for (uint32_t i = 1; i < Vb; ++i) {
            pack_at(i, cb_u, i);
        }
        pack_first(Vb, o_cb, 0);
        for (uint32_t i = 1; i < Vb; ++i) {
            pack_at(Vb + i, o_cb, i);
        }
        tile_regs_release();
        cb_push_back(cb_u, Vb);
        cb_push_back(cb_u_f32, Vb);
        GDN_MARK("m_st_o");
        cb_wait_front(cb_u, Vb);
        return;
    }
    for (uint32_t part = 0; part < 2; ++part) {
        for (uint32_t vt0 = 0; vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
            const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
            tile_regs_acquire();
            fold_part(s_mm, part, vt0, n, 0, p_half);
            if (HOLD && part == 0) {
                commit_stay();
            } else {
                tile_regs_commit();
            }
            tile_regs_wait();
            pack_first(0, part == 0 ? cb_u : o_cb, vt0);
            for (uint32_t i = 1; i < n; ++i) {
                pack_at(i, part == 0 ? cb_u : o_cb, vt0 + i);
            }
            if (HOLD && part == 0) {
                release_stay();
            } else {
                tile_regs_release();
            }
        }
        if (part == 0) {
            cb_push_back(cb_u, Vb);
            cb_push_back(cb_u_f32, Vb);
            GDN_MARK("m_st_u");
            cb_wait_front(cb_u, Vb);  // O's window starts behind U's release
        }
    }
    GDN_MARK("m_st_o");
}

// SEQ_SPLIT, a state core of a segment 1.. after its last step (H in cb_state): once P and the entering state S_in
// are in cb_seg_in, the final state P x S_in + H -> cb_state_out, then per tile row O = Oh_P x S_in + Oh_H (Oh_H held
// in cb_reads, Oh_P landing in cb_seg_row) -> cb_o_acc.
ALWI void seg_finish(uint32_t rows) {
    cb_wait_front(cb_seg_in, SEG_S + n_state);
    cb_wait_front(cb_state, n_state);
    cb_reserve_back(cb_state_out, n_state);
    for (uint32_t b = 0; b < n_state; b += UPDATE_BATCH) {
        const uint32_t n = (n_state - b) < UPDATE_BATCH ? (n_state - b) : UPDATE_BATCH;
        tile_regs_acquire();
        begin_loads(cb_state);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_state, b + i, i);
        }
        reconfig_data_format(cb_seg_in, cb_seg_in);
        matmul_init(cb_seg_in, cb_seg_in);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = b + i;
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_tiles(cb_seg_in, cb_seg_in, (t / Vb) * Kt + k, SEG_S + k * Vb + t % Vb, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_first(i, cb_state_out, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_state_out, n_state);
    cb_pop_front(cb_state, n_state);
    cb_pop_front(cb_state_mm, n_state);
#pragma GCC unroll 1
    for (uint32_t r = 0; r < rows; ++r) {
        cb_wait_front(cb_seg_row, Kt);
        cb_wait_front(cb_reads, Vb);
        cb_reserve_back(cb_o_acc, Vb);
        for (uint32_t vt0 = 0; vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
            const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
            tile_regs_acquire();
            begin_loads(cb_reads);
            for (uint32_t i = 0; i < n; ++i) {
                load(cb_reads, vt0 + i, i);
            }
            reconfig_data_format(cb_seg_in, cb_seg_row);
            matmul_init(cb_seg_row, cb_seg_in);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t k = 0; k < Kt; ++k) {
                    matmul_tiles(cb_seg_row, cb_seg_in, k, SEG_S + k * Vb + vt0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_first(0, cb_o_acc, vt0);
            for (uint32_t i = 1; i < n; ++i) {
                pack_at(i, cb_o_acc, vt0 + i);
            }
            tile_regs_release();
        }
        cb_push_back(cb_o_acc, Vb);
        cb_pop_front(cb_seg_row, Kt);
        cb_pop_front(cb_reads, Vb);
    }
    cb_pop_front(cb_seg_in, SEG_S + n_state);
}

// p_half, o_cb: SEQ_SPLIT (the P half's step; where the row's o goes).
ALWI void chunk_state_prepped(
    Buffers& bufs,
    bool first_in_pair,
    bool last_in_pair,
    uint32_t decay_bits,
    bool p_half = false,
    uint32_t o_cb = cb_o_acc) {
    // The state as a matmul operand (as read_state takes it).
    const uint32_t s_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    if constexpr (FOLD) {
        cb_wait_front(first_in_pair ? cb_state_in : cb_state, n_state);
        cb_wait_front(s_mm, n_state);
        cb_wait_front(cb_in_kq, 2 * Kt);
        state_fold(s_mm, p_half, o_cb);
        update_state<cb_in_knt>(first_in_pair, last_in_pair, cb_u, decay_bits);
        cb_pop_front(cb_u, Vb);
        cb_pop_front(cb_u_f32, Vb);
        return;
    }
    if constexpr (FUSED_RHS) {
        cb_wait_front(first_in_pair ? cb_state_in : cb_state, n_state);
        cb_wait_front(s_mm, n_state);
        cb_wait_front(cb_in_kq, 2 * Kt);
    } else {
        read_state<cb_in_kq>(bufs, first_in_pair);
        GDN_MARK("m_st_read");
        cb_wait_front(cb_reads, 2 * Vb);
    }
    cb_wait_front(cb_in_mat, GDN_IN_MAT_PAGES);
    cb_wait_front(cb_in_v, Vb);
    constexpr uint32_t PAIRS = ckl::DEST_AUTO_LIMIT / 2;  // rhs column tiles per window (operand pairs)
    cb_reserve_back(cb_uw, Vb);
    for (uint32_t vt0 = 0; FUSED_RHS && vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        begin_loads(cb_in_v);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_in_v, vt0 + i, i);
        }
        reconfig_data_format(s_mm, cb_in_kq);
        matmul_init(cb_in_kq, s_mm);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(cb_in_kq, s_mm, kt, kt * Vb + vt0 + i, i);  // - beta exp(c) Rk onto beta V
            }
        }
        if constexpr (HOLD) {
            commit_stay();
        } else {
            tile_regs_commit();
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb_uw);
        pack_first(0, cb_uw, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_uw, vt0 + i);
        }
        if constexpr (HOLD) {
            release_stay();
        } else {
            tile_regs_release();
        }
    }
    for (uint32_t vt0 = 0; !FUSED_RHS && vt0 < Vb; vt0 += PAIRS) {
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
    GDN_MARK("m_st_rhs");
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
        if constexpr (HOLD) {
            commit_stay();
        } else {
            tile_regs_commit();
        }
        tile_regs_wait();
        pack_first(0, cb_u, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_u, vt0 + i);
        }
        if constexpr (HOLD) {
            release_stay();
        } else {
            tile_regs_release();
        }
    }
    cb_pop_front(cb_uw, Vb);
    cb_push_back(cb_u, Vb);
    cb_push_back(cb_u_f32, Vb);
    GDN_MARK("m_st_u");
    cb_wait_front(cb_u, Vb);
    for (uint32_t vt0 = 0; vt0 < Vb; vt0 += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (Vb - vt0) < ckl::DEST_AUTO_LIMIT ? (Vb - vt0) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        if constexpr (FUSED_RHS) {
            reconfig_data_format(s_mm, cb_in_kq);
            matmul_init(cb_in_kq, s_mm);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    matmul_tiles(cb_in_kq, s_mm, Kt + kt, kt * Vb + vt0 + i, i);  // exp(c) Rq
                }
            }
        } else {
            begin_loads(cb_reads);
            for (uint32_t i = 0; i < n; ++i) {
                load(cb_reads, Vb + vt0 + i, i);
            }
        }
        reconfig_data_format(cb_u, cb_in_mat);
        matmul_init(cb_in_mat, cb_u);
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(cb_in_mat, cb_u, IN_MAT_QD, vt0 + i, i);  // + QD x U onto exp(c) Rq
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, o_cb, vt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, o_cb, vt0 + i);
        }
        tile_regs_release();
    }
    if constexpr (!FUSED_RHS) {
        cb_pop_front(cb_reads, 2 * Vb);
    }
    GDN_MARK("m_st_o");
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
