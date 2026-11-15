// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute, local role (GDN_ROLE_LOCAL): prep and state of a unit on one core. R-STEP, R-VSPLIT,
// R-CHUNK, R-BCAST (q / k injectors and receivers) and the R-QKCORE pair cores.

#include "gated_delta_net_compute_common.hpp"
#include "gated_delta_net_compute_chunk.hpp"

namespace {

static_assert(ROLE == GDN_ROLE_LOCAL, "local-role kernel");

// ---------------------------------------------------------------- one token step
// p: the token's row in its tile row.
ALWI void token_step(
    Buffers& bufs,
    bool first_in_pair,
    bool last_in_pair,
    bool first_in_row,
    bool last_in_row,
    uint32_t p,
    RowScope scope,
    uint32_t rot = 0) {
    // read (5a): [rk; ro] = [kn; qn] x S_{t-1}
    read_state<cb_kq_rows>(bufs, first_in_pair);
    GDN_MARK("m_read");
    // token vector (5b): w = v - alpha rk, u = beta (sel (*)col w), o = sel (*)col (alpha ro) + d u, with sel the
    // row-p selector and alpha, beta, d fp32 scalars of token p. Shared-q/k builds (the decode regimes) run one DEST
    // window per column tile and, for a tile row with this token alone, also pack o^2 (the RMS-norm square); the
    // others run u and o as two windows (the fused window does not fit their math thread's kernel-config text).
    cb_wait_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
    const uint32_t alpha_bits = read_tile_value(cb_tok_coef, COEF_ALPHA_SEL, elem_index(p, 0));
    const uint32_t d_bits = read_tile_value(cb_d, D_PAGES - 1, elem_index(p, p));
    const bool square_o = QK_SHARED && first_in_row && last_in_row;
    cb_wait_front(cb_reads, 2 * Vb);
    cb_reserve_back(cb_u, Vb);
    cb_reserve_back(cb_u_f32, Vb);
    if (square_o) {
        cb_reserve_back(cb_sq_o, Vb);
    }
    if constexpr (QK_SHARED) {
        const uint32_t beta_bits = read_tile_value(cb_tok_coef, COEF_BETA_SEL, elem_index(p, 0));
        // QK_RAW: cb_kq_rows held y_k, y_q, so rk, ro, d are raw. With the inverse norms ik, iq: rk = ik rk_raw,
        // ro = iq ro_raw, and the update operand u' = ik u (cb_knT holds y_k^T), so d u = iq d_raw u'.
        uint32_t ik_bits = ONE_BITS;
        uint32_t iq_bits = ONE_BITS;
        if constexpr (QK_RAW) {
            cb_wait_front(cb_rinv_in, 2);
            ik_bits = read_tile_value(cb_rinv_in, 0, elem_index(p, 0));
            iq_bits = read_tile_value(cb_rinv_in, 1, elem_index(p, 0));
        }
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, vt, 0);       // rk
            load(cb_v, vt, 1);           // v
            load(cb_reads, Vb + vt, 2);  // ro
            load(cb_tok_coef, COEF_SEL, 3);
            if constexpr (GDN_TOKEN_VECTOR_FUSED) {
                if (square_o) {
                    token_vector_band<true>(alpha_bits, beta_bits, d_bits, ik_bits, iq_bits, scope);
                } else {
                    token_vector_band<false>(alpha_bits, beta_bits, d_bits, ik_bits, iq_bits, scope);
                }
            } else {
                binop_with_scalar_tile_init();
                binop_scalar_scoped<MUL_UNARY>(0, alpha_bits, scope);
                binop_scalar_scoped<MUL_UNARY>(2, alpha_bits, scope);
                if constexpr (QK_RAW) {
                    binop_scalar_scoped<MUL_UNARY>(0, ik_bits, scope);
                    binop_scalar_scoped<MUL_UNARY>(2, iq_bits, scope);
                }
                // w in the v slot: off the scope's rows u keeps v's zeros (rk there may hold other sequences' rows).
                sub_binary_tile_init();
                binary_scoped<ckernel::BinaryOp::SUB>(1, 0, 1, scope);
                sfpu_bcast_col_init();
                mul_bcast_col_scoped(1, 3, scope);
                mul_bcast_col_scoped(2, 3, scope);
                binop_with_scalar_tile_init();
                binop_scalar_scoped<MUL_UNARY>(1, beta_bits, scope);  // u
                if constexpr (QK_RAW) {
                    binop_scalar_scoped<MUL_UNARY>(1, ik_bits, scope);  // u'
                }
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(1, 0);
                binop_with_scalar_tile_init();
                binop_scalar_scoped<MUL_UNARY>(0, d_bits, scope);
                if constexpr (QK_RAW) {
                    binop_scalar_scoped<MUL_UNARY>(0, iq_bits, scope);
                }
                add_binary_tile_init();
                binary_scoped<ckernel::BinaryOp::ADD>(2, 0, 2, scope);  // o
                if (square_o) {
                    mul_binary_tile_init();
                    binary_scoped<ckernel::BinaryOp::MUL>(2, 2, 0, scope);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_first(1, cb_u, vt);
            if (square_o) {
                pack_at(0, cb_sq_o, vt);
            }
            if (!first_in_row) {
                pack_reconfig_l1_acc(1);
            }
            pack_at(2, cb_o_acc, vt);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
        cb_push_back(cb_u, Vb);
        cb_push_back(cb_u_f32, Vb);
        cb_wait_front(cb_u, Vb);  // the update's matmul operand
        cb_wait_front(cb_u_f32, Vb);
    } else {
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, vt, 0);
            load(cb_tok_coef, COEF_BETA_SEL, 3);
            load(cb_v, vt, 2);
            binop_with_scalar_tile_init();
            binop_scalar_scoped<MUL_UNARY>(0, alpha_bits, scope);
            sub_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::SUB>(2, 0, 2, scope);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(2, 3, scope);
            tile_regs_commit();
            tile_regs_wait();
            pack_first(2, cb_u, vt);
            tile_regs_release();
        }
        cb_push_back(cb_u, Vb);
        cb_push_back(cb_u_f32, Vb);
        cb_wait_front(cb_u, Vb);
        cb_wait_front(cb_u_f32, Vb);
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, Vb + vt, 0);
            load(cb_tok_coef, COEF_ALPHA_SEL, 1);
            load(cb_u_f32, vt, 2);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(0, 1, scope);
            binop_with_scalar_tile_init();
            binop_scalar_scoped<MUL_UNARY>(2, d_bits, scope);
            add_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::ADD>(0, 2, 0, scope);
            tile_regs_commit();
            tile_regs_wait();
            if (!first_in_row) {
                pack_reconfig_l1_acc(1);
            }
            pack_first(0, cb_o_acc, vt);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
    }
    if (square_o) {
        cb_push_back(cb_sq_o, Vb);
    }
    cb_pop_front(cb_reads, 2 * Vb);
    if constexpr (PIECES > 1) {
        // o of the tile row is final: its partial sum o^2 goes to the peers now, so the all-gather overlaps the update.
        if (last_in_row) {
            cb_push_back(cb_o_acc, Vb);
            osum_partial(bufs, scope, square_o);
        }
    }
    // update (5c): S_t = alpha S_{t-1} (SFPU, fp32) + kn^T u (matmul accumulated onto DEST)
    GDN_MARK("m_tokvec");
    update_state<cb_knT>(first_in_pair, last_in_pair, cb_u, alpha_bits, rot);
    GDN_MARK("m_update");
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
    cb_pop_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
}

}  // namespace

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t v_eps_bits = get_arg_val<uint32_t>(3);  // V * epsilon
    // Compile-time for a group of receivers (descriptor GDN_GROUP_QK_ROLE): their binary drops the q / k part. A
    // compile-time injector role unrolls that part and grows the binary, so injectors keep the runtime arg.
    const uint32_t qk_role =
        GDN_GROUP_QK_ROLE == GDN_QK_RECEIVER ? uint32_t{GDN_QK_RECEIVER} : get_arg_val<uint32_t>(4);
    const uint32_t unit_stride = get_arg_val<uint32_t>(7);  // units first_unit, + stride, ... (R-QKCORE: Hv)

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_state_mm, cb_reads);
    Buffers bufs;
    constexpr bool qk_shared = QK_SHARED;  // every core of an R-BCAST program has a shared role
    // Compile-time outside R-BCAST (every q/k column), so those builds fold the column mapping away.
    const UnitCols cols = !QK_SHARED
                              ? UnitCols{0, 2 * Kt}
                              : UnitCols{qk_role == GDN_QK_K_INJECTOR ? Kt : 0, qk_role == GDN_QK_RECEIVER ? 0 : Kt};
    // FPU conv reads raw conv_weight tiles (tap j at row j); the SFPU conv one row-0 tile per tap.
    const uint32_t conv_w_pages = (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * cols.count();

    if constexpr (CHUNKED) {
        cb_wait_front(cb_tri_mask, 2);  // reader-built once (mask, identity), held for the kernel
    }

    const bool hold_qk = HOLD_QK && qk_role == GDN_QK_RECEIVER && unit_stride == Hv * PIECES && T == 1;
    const bool lag = UNIT_LAG && hold_qk && num_units > 1;
    RowScope lag_scope = RowScope::of(0, 1);
    auto lagged_rms = [&](RowScope rms_scope) {
        GDN_ZONE("gdn_rmsnorm");
        cb_wait_front(cb_norm_w, Vb);
        rmsnorm_gate_block<cb_o_acc, 1, !CHUNKED>(bufs, v_eps_bits, rms_scope, QK_SHARED);
        cb_pop_front(cb_norm_w, Vb);
    };
    for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
        const uint32_t unit = first_unit + unit_idx * unit_stride;
        const bool last_unit = unit_idx + 1 == num_units;
        const uint32_t seq_lo = (unit / PIECES / Hv) * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        cb_wait_front(cb_conv_w, conv_w_pages);
        cb_wait_front(cb_gate_w, 2);
        if (!lag) {
            cb_wait_front(cb_norm_w, Vb);
        }

        uint32_t token = 0;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            const RowScope scope = RowScope::of(lo - base_row, hi - base_row);
            {
                GDN_ZONE("gdn_prep");  // gates, conv, q / k norms
                if constexpr (qk_shared) {
                    // An injector convolves and normalizes its q/k part before its v columns: the part is what every
                    // core of the key head waits for.
                    const uint32_t split = cols.qk_cols != 0 ? cols.qk_cols : cols.count();
#pragma GCC unroll 1
                    for (uint32_t pass = 0; pass < (split < cols.count() ? 2 : 1); ++pass) {
                        const uint32_t first = pass == 0 ? 0 : split;
                        const uint32_t last = pass == 0 ? split : cols.count();
                        if constexpr (CONV_ON_FPU) {
                            conv_block_fpu(scope, cols, first, last);
                        } else {
                            conv_block(scope, cols, first, last);
                        }
                        GDN_MARK("m_conv");
                        if (pass == 0 && cols.qk_cols != 0) {
                            qk_part_block(bufs, qk_role == GDN_QK_Q_INJECTOR, scope);
                            GDN_MARK("m_qk_norm");
                        }
                    }
                    // After an injector's q / k part, which the key head waits for; only the token step needs them.
                    gates_block(scope);
                    GDN_MARK("m_gates");
                    if (!hold_qk || unit_idx == 0) {
                        qk_finish_block(bufs);
                        GDN_MARK("m_finish");
                    }
                } else {
                    gates_block(scope);
                    if constexpr (CONV_ON_FPU) {
                        conv_block_fpu(scope, cols, 0, cols.count());
                    } else {
                        conv_block(scope, cols, 0, cols.count());
                    }
                    l2norm_block(bufs, scope);
                }
            }
            cb_reserve_back(cb_o_acc, Vb);
            {
                GDN_ZONE("gdn_step");
                if constexpr (CHUNKED) {
                    chunk_step(bufs, token == 0, token + (hi - lo) == T, hi - 1 - base_row);
                    token += hi - lo;
                } else {
                    for (uint32_t row = lo; row < hi; ++row, ++token) {
                        token_step(
                            bufs,
                            token == 0,
                            token == T - 1,
                            row == lo,
                            row + 1 == hi,
                            row - base_row,
                            scope,
                            (unit / PIECES * STATE_STAGGER) % n_state);
                    }
                    if (!hold_qk || last_unit) {
                        cb_pop_front(cb_d, D_PAGES);
                    }
                }
            }
            if (CHUNKED || PIECES == 1) {
                cb_push_back(cb_o_acc, Vb);  // else pushed by the token step
            }
            if (!hold_qk || last_unit) {
                cb_pop_front(cb_kq_rows, 2 * Kt);
                if constexpr (qk_shared) {
                    cb_pop_front(cb_kq_rows_f32, 2 * Kt);
                }
                if constexpr (QK_RAW) {
                    cb_pop_front(cb_rinv_in, 2);
                }
                cb_pop_front(cb_knT, Kt);
            }
            cb_pop_front(cb_v, Vb);
            if (lag) {
                // T = 1: this tile row is the unit's only one.
                if (unit_idx > 0) {
                    lagged_rms(lag_scope);
                }
                lag_scope = scope;
                continue;
            }
            GDN_ZONE("gdn_rmsnorm");
            rmsnorm_gate_block<cb_o_acc, 1, !CHUNKED>(
                bufs, v_eps_bits, scope, QK_SHARED && hi - lo == 1);  // token_step's square_o
            GDN_MARK("m_rms");
        }
        cb_pop_front(cb_conv_w, conv_w_pages);
        cb_pop_front(cb_gate_w, 2);
        if (!lag) {
            cb_pop_front(cb_norm_w, Vb);
        }
    }
    if (lag) {
        lagged_rms(lag_scope);
    }
}
