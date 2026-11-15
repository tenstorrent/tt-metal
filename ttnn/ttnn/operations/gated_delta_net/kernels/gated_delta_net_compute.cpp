// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute (regime R-STEP, op_design.md).
//
// Per pair, per tile row: gates_block -> conv_block -> l2norm_block -> recurrence_block (one
// token per step: read matmul, token vector, state update) -> rmsnorm_gate_block.
//
// Helpers: matmul_block (read [kn; qn] x S) and reduce (row sums, Accurate fp32). Raw LLK where the
// design names a custom block op, with the reasons:
//   - row scope (sfpu_tile_scope): in decode a tile row holds one or a few token rows; the row-local
//     SFPU ops (squares, rsqrt post-op, gates, silu, broadcasts) run only on the 4-row band holding
//     them. eltwise_chain / the *_tile APIs run whole tiles, so squares are a raw loop over
//     DEST_AUTO_LIMIT-tile windows and the band variants call the LLK bodies with ITERATIONS=2;
//   - every row / column broadcast must stay fp32 (alpha, the L2 scale, S): the only broadcast in
//     eltwise_chain is BinaryFpu (srcB, 19-bit), so these use sfpu_*_bcast_{row,col} in DEST;
//   - conv_block_fpu (bf16 tokens): the four taps of a column are FPU row-broadcast multiplies
//     (srcB = conv weight row) summed in one fp32 DEST tile, which ELWMUL accumulates into; no
//     helper accumulates FPU binaries into DEST. fp32 tokens keep the SFPU path (srcA/srcB are 19-bit);
//   - the state update accumulates the increment matmul onto an SFPU-decayed fp32 state already in
//     DEST; matmul_block owns DEST acquire/pack and cannot seed DEST with it.
// DEST discipline in the raw windows: every unpack-to-DEST load of a window is issued before its
// first SFPU op (an unpack-to-DEST load may run ahead of SFPU work on the math thread), and the
// math thread waits for the SFPU before the FPU accumulates onto SFPU results.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"

namespace ckl = compute_kernel_lib;

namespace {

// CB indices and GDN_* spec constants are host defines (gated_delta_net_program_descriptor.py).
constexpr uint32_t cb_state_in = GDN_CB_STATE_IN;
constexpr uint32_t cb_state_in_mm = GDN_CB_STATE_IN_MM;
constexpr uint32_t cb_state = GDN_CB_STATE;
constexpr uint32_t cb_state_mm = GDN_CB_STATE_MM;
constexpr uint32_t cb_state_out = GDN_CB_STATE_OUT;
constexpr uint32_t cb_conv_w = GDN_CB_CONV_W;
constexpr uint32_t cb_window = GDN_CB_WINDOW;
constexpr uint32_t cb_y_qk = GDN_CB_Y_QK;
constexpr uint32_t cb_v = GDN_CB_V;
constexpr uint32_t cb_sq = GDN_CB_SQ;
constexpr uint32_t cb_rinv = GDN_CB_RINV;
constexpr uint32_t cb_kq_rows = GDN_CB_KQ_ROWS;
constexpr uint32_t cb_knT = GDN_CB_KNT;
constexpr uint32_t cb_d = GDN_CB_D;
constexpr uint32_t cb_gate_in = GDN_CB_GATE_IN;
constexpr uint32_t cb_gate_w = GDN_CB_GATE_W;
constexpr uint32_t cb_gates = GDN_CB_GATES;
constexpr uint32_t cb_tok_coef = GDN_CB_TOK_COEF;
constexpr uint32_t cb_reads = GDN_CB_READS;
constexpr uint32_t cb_u = GDN_CB_U;
constexpr uint32_t cb_o_acc = GDN_CB_O_ACC;
constexpr uint32_t cb_z = GDN_CB_Z;
constexpr uint32_t cb_norm_w = GDN_CB_NORM_W;
constexpr uint32_t cb_out_rows = GDN_CB_OUT_ROWS;
constexpr uint32_t cb_reduce_scaler = GDN_CB_REDUCE_SCALER;
constexpr uint32_t cb_u_f32 = GDN_CB_U_F32;
constexpr uint32_t cb_conv_acc = GDN_CB_CONV_ACC;
constexpr uint32_t cb_sq_o = GDN_CB_SQ_O;
constexpr uint32_t cb_rinv_o = GDN_CB_RINV_O;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t COEF_ALPHA = 0;      // alpha_p in every row
constexpr uint32_t COEF_BETA_SEL = 1;   // beta_p at row p, 0 elsewhere
constexpr uint32_t COEF_ALPHA_SEL = 2;  // alpha_p at row p, 0 elsewhere

constexpr uint32_t ONE_BITS = 0x3f800000u;
constexpr uint32_t SOFTPLUS_THRESHOLD_BITS = 0x41a00000u;  // 20.0f
constexpr uint32_t L2_EPS_BITS = 0x358637bdu;              // 1e-6f

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t Hv = get_compile_time_arg_val(2);
constexpr uint32_t MM_SUBBLOCK_W = get_compile_time_arg_val(3);
constexpr uint32_t SQRT_V_BITS = get_compile_time_arg_val(4);
constexpr uint32_t Q_SCALE_BITS = get_compile_time_arg_val(5);
constexpr bool CONV_ON_FPU = get_compile_time_arg_val(6) != 0;
// FPU conv reads raw conv_weight tiles (tap j at row j); the SFPU conv one row-0 tile per tap.
constexpr uint32_t conv_w_pages = CONV_ON_FPU ? (2 * Kt + Vt) : GDN_CONV_TAPS * (2 * Kt + Vt);
constexpr uint32_t n_state = Kt * Vt;
constexpr uint32_t pair_cols = 2 * Kt + Vt;
// State tiles per update batch: every DEST slot of a window.
constexpr uint32_t UPDATE_BATCH = ckl::DEST_AUTO_LIMIT < n_state ? ckl::DEST_AUTO_LIMIT : n_state;
static_assert(ckl::DEST_AUTO_LIMIT >= 4, "raw windows use 4 DEST slots");
static_assert(Vt % MM_SUBBLOCK_W == 0, "read-matmul subblock must divide Vt");

// Unpack setup for a run of copy_tile loads: issued once per run of CBs that share a data format and
// unpack-to-DEST mode (descriptor TOKEN_CBS / U2D_CBS), not per CB.
ALWI void begin_loads(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

ALWI void load(uint32_t cb, uint32_t tile, uint32_t dst) { copy_tile(cb, tile, dst); }

// Each window packs one format (Float32, or the token dtype for cb_out_rows), so the packer is
// reconfigured once per window (pack_first),
// not per tile.
ALWI void pack_first(uint32_t dst, uint32_t cb, uint32_t idx) {
    pack_reconfig_data_format(cb);
    pack_tile<true>(dst, cb, idx);
}

ALWI void pack_at(uint32_t dst, uint32_t cb, uint32_t idx) { pack_tile<true>(dst, cb, idx); }

ALWI void wait_sfpu_before_fpu() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::WAIT_SFPU);
#endif
}

// Valid token rows [row_lo, row_hi) of a tile row (tile-local). When they sit in one 4-row band, the
// row-local SFPU ops run on that band of both faces only (4 of 32 vector ops, sfpu_tile_scope); rows
// outside it keep their pre-op DEST values, which no valid row reads.
struct RowScope {
    uint32_t band;  // band index 0..7, or FULL_TILE
    static constexpr uint32_t FULL_TILE = 0xffffffffu;
    static RowScope of(uint32_t row_lo, uint32_t row_hi) {
        return {(row_lo >> 2) == ((row_hi - 1) >> 2) ? (row_lo >> 2) : FULL_TILE};
    }
};

// One SFPU body of ITERATIONS=2 (one 4-row band, both column parities) on `band` of both faces of its half.
template <typename Body>
ALWI void sfpu_band(uint32_t dst, uint32_t band, Body body) {
#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(dst);
    if (band >= 4) {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    for (uint32_t b = 0; b < (band & 3); ++b) {
        math::inc_dst_addr<4>();
    }
    body();
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    body();
    _llk_math_eltwise_sfpu_done_();
#endif
}

ALWI void silu_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        silu_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, 2>(); });
#endif
    }
}

// Row-local unary SFPU ops on the scope's rows (full-tile API call when unscoped).
ALWI void softplus_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        softplus_tile(dst, ONE_BITS, ONE_BITS, SOFTPLUS_THRESHOLD_BITS);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] {
            ckernel::sfpu::calculate_softplus<APPROX, DST_ACCUM_MODE, 2>(ONE_BITS, ONE_BITS, SOFTPLUS_THRESHOLD_BITS);
        });
#endif
    }
}

ALWI void exp_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        exp_tile<false>(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 2>(); });
#endif
    }
}

ALWI void negative_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        negative_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::_calculate_negative_<APPROX, 2>(); });
#endif
    }
}

ALWI void sigmoid_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        sigmoid_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::calculate_sigmoid<false, DST_ACCUM_MODE, 2>(); });
#endif
    }
}

ALWI void square_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        square_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::calculate_square<APPROX, DST_ACCUM_MODE, 2>(); });
#endif
    }
}

template <int BINOP_MODE>
ALWI void binop_scalar_scoped(uint32_t dst, uint32_t scalar_bits, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        if constexpr (BINOP_MODE == ADD_UNARY) {
            add_unary_tile(dst, scalar_bits);
        } else {
            mul_unary_tile(dst, scalar_bits);
        }
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [=] {
            ckernel::sfpu::calculate_binop_with_scalar<APPROX, BINOP_MODE, 2, DST_ACCUM_MODE>(scalar_bits);
        });
#endif
    }
}

ALWI void rsqrt_scoped(uint32_t dst, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        rsqrt_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope.band, [] { ckernel::sfpu::calculate_rsqrt<APPROX, 2, DST_ACCUM_MODE, false>(); });
#endif
    }
}

// fp32 SFPU binary (DEST[d0] op DEST[d1] -> DEST[out]) on the scope's rows.
template <ckernel::BinaryOp OP>
ALWI void binary_scoped(uint32_t d0, uint32_t d1, uint32_t out, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        if constexpr (OP == ckernel::BinaryOp::ADD) {
            add_binary_tile(d0, d1, out);
        } else if constexpr (OP == ckernel::BinaryOp::SUB) {
            sub_binary_tile(d0, d1, out);
        } else {
            mul_binary_tile(d0, d1, out);
        }
    } else {
#ifdef TRISC_MATH
        sfpu_band(
            0, scope.band, [=] { ckernel::sfpu::calculate_sfpu_binary<APPROX, OP, 2, DST_ACCUM_MODE>(d0, d1, out); });
#endif
    }
}

// DEST[data] *= column 0 of DEST[col] (broadcast along the row), on the scope's rows.
// Band path: the LLK's own per-band body of sfpu_mul_bcast_col (needs sfpu_bcast_col_init).
ALWI void mul_bcast_col_scoped(uint32_t data, uint32_t col, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        sfpu_mul_bcast_col(data, col);
    } else {
#ifdef TRISC_MATH
        using namespace ckernel::sfpu;
        _llk_math_eltwise_sfpu_start_(0);
        const uint32_t left = (scope.band >= 4 ? FACE2_BASE : FACE0_BASE) + (scope.band & 3) * ROW_BAND_STRIDE;
        const uint32_t right = left + (FACE1_BASE - FACE0_BASE);
        _process_col_bcast_row_band_<ckernel::BinaryOp::MUL>(
            col * DEST_TILE_SIZE_RAW + left, left, right, data * DEST_TILE_SIZE_RAW, data * DEST_TILE_SIZE_RAW);
        _llk_math_eltwise_sfpu_done_();
#endif
    }
}

// DEST[data] op= row 0 of DEST[row] (broadcast down the column), on the scope's rows.
// Band path: the LLK's own per-band body of sfpu_*_bcast_row (needs sfpu_bcast_row_init).
template <ckernel::BinaryOp OP>
ALWI void bcast_row_scoped(uint32_t data, uint32_t row, RowScope scope) {
    if (scope.band == RowScope::FULL_TILE) {
        if constexpr (OP == ckernel::BinaryOp::ADD) {
            sfpu_add_bcast_row(data, row);
        } else {
            sfpu_mul_bcast_row(data, row);
        }
    } else {
#ifdef TRISC_MATH
        using namespace ckernel::sfpu;
        constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
        _llk_math_eltwise_sfpu_start_(0);
        const uint32_t row0 = row * DEST_TILE_SIZE_RAW + FACE0_BASE;
        TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_7, row0 + COL_GROUP_OFFSETS[0]);
        TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_7, row0 + COL_GROUP_OFFSETS[1]);
        TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_7, row0 + COL_GROUP_OFFSETS[2]);
        TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_7, row0 + COL_GROUP_OFFSETS[3]);
        _process_row_bcast_data_band_<OP>(
            data * DEST_TILE_SIZE_RAW,
            data * DEST_TILE_SIZE_RAW,
            scope.band >= 4 ? FACE2_BASE : FACE0_BASE,
            (scope.band & 3) * ROW_BAND_STRIDE);
        _llk_math_eltwise_sfpu_done_();
#endif
    }
}

struct Buffers {
    DataflowBuffer state_in_mm{cb_state_in_mm};
    DataflowBuffer state_mm{cb_state_mm};
    DataflowBuffer y_qk{cb_y_qk};
    DataflowBuffer sq{cb_sq};
    DataflowBuffer rinv{cb_rinv};
    DataflowBuffer sq_o{cb_sq_o};
    DataflowBuffer rinv_o{cb_rinv_o};
    DataflowBuffer d{cb_d};
    DataflowBuffer kq_rows{cb_kq_rows};
    DataflowBuffer reads{cb_reads};
    DataflowBuffer o_acc{cb_o_acc};
    DataflowBuffer scaler{cb_reduce_scaler};
};

// ---------------------------------------------------------------- gates_block
// alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b), on the token rows of the gate tile.
ALWI void gates_block(RowScope scope) {
    cb_wait_front(cb_gate_in, 2);
    tile_regs_acquire();
    begin_loads(cb_gate_in);
    load(cb_gate_in, 1, 0);  // a
    load(cb_gate_in, 0, 3);  // b
    load(cb_gate_w, 1, 1);  // dt_bias (row 0)
    load(cb_gate_w, 0, 2);  // A_log (row 0)
    sfpu_bcast_row_init();
    bcast_row_scoped<ckernel::BinaryOp::ADD>(0, 1, scope);
    softplus_tile_init();
    softplus_scoped(0, scope);
    // Pass 0: exp(A_log), then g = -exp(A_log) * softplus; pass 1: alpha = exp(g). One exp call site.
#pragma GCC unroll 1
    for (uint32_t pass = 0; pass < 2; ++pass) {
        exp_tile_init<false>();
        // A_log is a broadcast row (row 0 only); alpha is per token row.
        exp_scoped(pass == 0 ? 2 : 0, pass == 0 ? RowScope{0} : scope);
        if (pass == 0) {
            sfpu_bcast_row_init();
            bcast_row_scoped<ckernel::BinaryOp::MUL>(0, 2, scope);
            negative_tile_init();
            negative_scoped(0, scope);
        }
    }
    sigmoid_tile_init<false>();
    sigmoid_scoped(3, scope);
    tile_regs_commit();
    cb_reserve_back(cb_gates, 2);
    tile_regs_wait();
    pack_first(0, cb_gates, 0);
    pack_at(3, cb_gates, 1);
    tile_regs_release();
    cb_push_back(cb_gates, 2);
    cb_pop_front(cb_gate_in, 2);
}

// Two taps of one column tile: DEST0 = X_t0 (*)row W_t0 + X_t1 (*)row W_t1.
ALWI void conv_two_taps(uint32_t col, uint32_t tap0) {
    tile_regs_acquire();
    begin_loads(cb_window);
    load(cb_window, tap0, 0);
    load(cb_window, tap0 + 1, 2);
    load(cb_conv_w, GDN_CONV_TAPS * col + tap0, 1);
    load(cb_conv_w, GDN_CONV_TAPS * col + tap0 + 1, 3);
    sfpu_bcast_row_init();
    sfpu_mul_bcast_row(0, 1);
    sfpu_mul_bcast_row(2, 3);
    add_binary_tile_init();
    add_binary_tile(0, 2, 0);
    tile_regs_commit();
}

// ---------------------------------------------------------------- conv_block
// y = silu(sum_j X_j (*)row W_j) per column tile of the pair; q, k -> cb_y_qk, v -> cb_v.
ALWI void conv_block(RowScope scope) {
    cb_reserve_back(cb_y_qk, 2 * Kt);
    cb_reserve_back(cb_v, Vt);
    for (uint32_t col = 0; col < pair_cols; ++col) {
        cb_wait_front(cb_window, GDN_CONV_TAPS);
        cb_reserve_back(cb_conv_acc, 1);
        // Taps (0, 1) seed the partial sum, taps (2, 3) add onto it by packer L1 accumulation.
#pragma GCC unroll 1
        for (uint32_t tap0 = 0; tap0 < GDN_CONV_TAPS; tap0 += 2) {
            conv_two_taps(col, tap0);
            tile_regs_wait();
            pack_reconfig_l1_acc(tap0 == 0 ? 0 : 1);
            pack_first(0, cb_conv_acc, 0);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
        cb_push_back(cb_conv_acc, 1);
        cb_pop_front(cb_window, GDN_CONV_TAPS);

        cb_wait_front(cb_conv_acc, 1);
        tile_regs_acquire();
        begin_loads(cb_conv_acc);
        load(cb_conv_acc, 0, 0);
        silu_tile_init();
        silu_scoped(0, scope);
        tile_regs_commit();
        tile_regs_wait();
        if (col < 2 * Kt) {
            pack_first(0, cb_y_qk, col);
        } else {
            pack_first(0, cb_v, col - 2 * Kt);
        }
        tile_regs_release();
        cb_pop_front(cb_conv_acc, 1);
    }
    cb_push_back(cb_y_qk, 2 * Kt);
    cb_push_back(cb_v, Vt);
}

// ---------------------------------------------------------------- conv_block_fpu
// Same contract as conv_block: DEST = sum_j X_j (*)row W_j by FPU ELWMUL accumulation (DEST is zero
// after acquire), then silu in the same window. Exact for bf16 operands; products accumulate in fp32.
ALWI void conv_block_fpu(RowScope scope) {
    cb_reserve_back(cb_y_qk, 2 * Kt);
    cb_reserve_back(cb_v, Vt);
    for (uint32_t col = 0; col < pair_cols; ++col) {
        cb_wait_front(cb_window, GDN_CONV_TAPS);
        tile_regs_acquire();
        reconfig_data_format(cb_window, cb_conv_w);
        mul_bcast_rows_init(cb_window, cb_conv_w);
        for (uint32_t tap = 0; tap < GDN_CONV_TAPS; ++tap) {
            mul_tiles_bcast_rows(cb_window, cb_conv_w, tap, col, 0, tap);
        }
        silu_tile_init();
        silu_scoped(0, scope);
        tile_regs_commit();
        tile_regs_wait();
        if (col < 2 * Kt) {
            pack_first(0, cb_y_qk, col);
        } else {
            pack_first(0, cb_v, col - 2 * Kt);
        }
        tile_regs_release();
        cb_pop_front(cb_window, GDN_CONV_TAPS);
    }
    cb_push_back(cb_y_qk, 2 * Kt);
    cb_push_back(cb_v, Vt);
}

// Squares of `count` tiles of an fp32 buffer starting at `base` (the input is kept) -> sq_cb, DEST_AUTO_LIMIT
// tiles per window, on the scope's rows.
template <uint32_t in_cb, uint32_t sq_cb>
ALWI void square_block(uint32_t base, uint32_t count, RowScope scope) {
    cb_wait_front(in_cb, base + count);
    cb_reserve_back(sq_cb, count);
    for (uint32_t first = 0; first < count; first += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (count - first) < ckl::DEST_AUTO_LIMIT ? (count - first) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        begin_loads(in_cb);
        for (uint32_t i = 0; i < n; ++i) {
            load(in_cb, base + first + i, i);
        }
        square_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            square_scoped(i, scope);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, sq_cb, first);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, sq_cb, first + i);
        }
        tile_regs_release();
    }
    cb_push_back(sq_cb, count);
}

// Post-reduce op on the column-0 row sum: rsqrt(sum + add) * post_mul.
struct RsqrtPost {
    uint32_t add_bits;
    uint32_t post_mul_bits;
    RowScope scope;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        binop_scalar_scoped<ADD_UNARY>(dst, add_bits, scope);
        rsqrt_tile_init();
        rsqrt_scoped(dst, scope);
        binop_with_scalar_tile_init();
        binop_scalar_scoped<MUL_UNARY>(dst, post_mul_bits, scope);
    }
};

template <uint32_t sq_cb, uint32_t out_cb, typename PostOp>
ALWI void row_sum(Buffers& bufs, DataflowBuffer& sq_buf, DataflowBuffer& out_buf, uint32_t width, PostOp post_op) {
    ckl::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        sq_cb,
        cb_reduce_scaler,
        out_cb,
        ckl::ReduceInputPolicy::BulkWaitBulkPop,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(
        sq_buf,
        bufs.scaler,
        out_buf,
        ckl::ReduceInputBlockShape::row(width),
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::NoAccumulation{},
        post_op);
}

// rsqrt(sum of squares of the Kt tiles of cb_y_qk at `base` + 1e-6) * post_mul -> cb_rinv.
ALWI void l2_inverse_norm(Buffers& bufs, uint32_t base, uint32_t post_mul_bits, RowScope scope) {
    square_block<cb_y_qk, cb_sq>(base, Kt, scope);
    row_sum<cb_sq, cb_rinv>(bufs, bufs.sq, bufs.rinv, Kt, RsqrtPost{L2_EPS_BITS, post_mul_bits, scope});
}

// ---------------------------------------------------------------- l2norm_block
// rinv_q = K^-1/2 rsqrt(sum q^2 + 1e-6), rinv_k = rsqrt(sum k^2 + 1e-6);
// kn -> cb_kq_rows[0, Kt), qn -> cb_kq_rows[Kt, 2Kt), d = rowsum(qn * kn) -> cb_d, kn^T -> cb_knT.
ALWI void l2norm_block(Buffers& bufs, RowScope scope) {
    // q then k: one loop body keeps a single expansion of the square chain and the reduce.
#pragma GCC unroll 1
    for (uint32_t part = 0; part < 2; ++part) {
        l2_inverse_norm(bufs, part * Kt, part == 0 ? Q_SCALE_BITS : ONE_BITS, scope);
    }

    cb_wait_front(cb_rinv, 2);
    cb_reserve_back(cb_kq_rows, 2 * Kt);
    cb_reserve_back(cb_sq, Kt);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        tile_regs_acquire();
        begin_loads(cb_y_qk);
        load(cb_y_qk, kt, 0);       // q
        load(cb_y_qk, Kt + kt, 2);  // k
        load(cb_rinv, 0, 1);
        load(cb_rinv, 1, 3);
        sfpu_bcast_col_init();
        mul_bcast_col_scoped(0, 1, scope);  // qn
        mul_bcast_col_scoped(2, 3, scope);  // kn
        mul_binary_tile_init();
        binary_scoped<ckernel::BinaryOp::MUL>(0, 2, 1, scope);  // qn * kn
        tile_regs_commit();
        tile_regs_wait();
        pack_first(2, cb_kq_rows, kt);
        pack_at(0, cb_kq_rows, Kt + kt);
        pack_at(1, cb_sq, kt);
        tile_regs_release();
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_push_back(cb_sq, Kt);
    cb_pop_front(cb_rinv, 2);
    cb_pop_front(cb_y_qk, 2 * Kt);

    row_sum<cb_sq, cb_d>(bufs, bufs.sq, bufs.d, Kt, ckl::NoOp{});

    cb_wait_front(cb_kq_rows, 2 * Kt);
    cb_reserve_back(cb_knT, Kt);
    reconfig_data_format_srca(cb_kq_rows);
    transpose_init(cb_kq_rows);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        tile_regs_acquire();
        transpose_tile(cb_kq_rows, kt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_knT, kt);
        tile_regs_release();
    }
    cb_push_back(cb_knT, Kt);
    cb_wait_front(cb_d, 1);
    cb_wait_front(cb_v, Vt);
    cb_wait_front(cb_knT, Kt);
}

// ---------------------------------------------------------------- one token step
ALWI void token_step(Buffers& bufs, bool first_in_pair, bool last_in_pair, bool first_in_row, RowScope scope) {
    const uint32_t src = first_in_pair ? cb_state_in : cb_state;
    const uint32_t src_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    const uint32_t dst = last_in_pair ? cb_state_out : cb_state;
    DataflowBuffer& src_mm_buf = first_in_pair ? bufs.state_in_mm : bufs.state_mm;

    // read (5a): [rk; ro] = [kn; qn] x S_{t-1}
    {
        DeviceZoneScopedN("Z_MM");
        cb_wait_front(src, n_state);
        cb_wait_front(src_mm, n_state);
        ckl::matmul_block<
            false,
            false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::InputPolicy::NoWaitNoPop>(
            bufs.kq_rows,
            src_mm_buf,
            bufs.reads,
            bufs.reads,
            ckl::MatmulBlockShape::of(2, Vt / MM_SUBBLOCK_W, 1, MM_SUBBLOCK_W, Kt, 1));
    }
    {
        DeviceZoneScopedN("Z_UO");
        // token vector (5b): u = beta_sel (*)col (v - alpha (*)col rk); o = alpha_sel (*)col ro + d (*)col u
        cb_wait_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
        cb_wait_front(cb_reads, 2 * Vt);
        cb_reserve_back(cb_u, Vt);
        cb_reserve_back(cb_u_f32, Vt);
        for (uint32_t vt = 0; vt < Vt; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, vt, 0);
            load(cb_tok_coef, COEF_ALPHA, 1);
            load(cb_tok_coef, COEF_BETA_SEL, 3);
            load(cb_v, vt, 2);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(0, 1, scope);
            sub_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::SUB>(2, 0, 2, scope);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(2, 3, scope);
            tile_regs_commit();
            tile_regs_wait();
            pack_first(2, cb_u, vt);
            tile_regs_release();
        }
        cb_push_back(cb_u, Vt);
        cb_push_back(cb_u_f32, Vt);
        cb_wait_front(cb_u, Vt);
        cb_wait_front(cb_u_f32, Vt);
        for (uint32_t vt = 0; vt < Vt; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, Vt + vt, 0);
            load(cb_tok_coef, COEF_ALPHA_SEL, 1);
            load(cb_u_f32, vt, 2);
            load(cb_d, 0, 3);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(0, 1, scope);
            mul_bcast_col_scoped(2, 3, scope);
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
        cb_pop_front(cb_reads, 2 * Vt);
    }
    DeviceZoneScopedN("Z_UPD");
    // update (5c): S_t = alpha S_{t-1} (SFPU, fp32) + kn^T u (matmul accumulated onto DEST)
    // alpha is one scalar per token (element (0, 0) of the alpha page): an fp32 scalar SFPU multiply,
    // with no broadcast tile in DEST.
    const uint32_t alpha_bits = read_tile_value(cb_tok_coef, COEF_ALPHA, 0);
    const bool in_place = (src == dst);
    if (in_place) {
        // One allocation of exactly n_state pages: after the pop the read pointer wraps back onto
        // the same tiles, which are read by each batch before its packs overwrite them.
        cb_pop_front(src, n_state);
        cb_pop_front(src_mm, n_state);
    }
    cb_reserve_back(dst, n_state);
    if (dst == cb_state) {
        cb_reserve_back(cb_state_mm, n_state);
    }
    for (uint32_t first = 0; first < n_state; first += UPDATE_BATCH) {
        const uint32_t count = (n_state - first) < UPDATE_BATCH ? (n_state - first) : UPDATE_BATCH;
        tile_regs_acquire();
        begin_loads(src);
        for (uint32_t i = 0; i < count; ++i) {
            load(src, first + i, i);
        }
        binop_with_scalar_tile_init();
        for (uint32_t i = 0; i < count; ++i) {
            mul_unary_tile(i, alpha_bits);
        }
        wait_sfpu_before_fpu();
        reconfig_data_format(cb_u, cb_knT);
        matmul_init(cb_knT, cb_u);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = first + i;
            matmul_tiles(cb_knT, cb_u, tile / Vt, tile % Vt, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < count; ++i) {
            pack_first(i, dst, first + i);
        }
        tile_regs_release();
    }
    cb_push_back(dst, n_state);
    if (dst == cb_state) {
        cb_push_back(cb_state_mm, n_state);
    }
    if (!in_place) {
        cb_pop_front(src, n_state);
        cb_pop_front(src_mm, n_state);
    }
    cb_pop_front(cb_u, Vt);
    cb_pop_front(cb_u_f32, Vt);
    cb_pop_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
}

// ---------------------------------------------------------------- rmsnorm_gate_block
// out = ((o (*)col rsqrt(mean o^2 + eps)) (*)row w) * silu(z), with
// rsqrt(mean o^2 + eps) = sqrt(V) * rsqrt(sum o^2 + V eps).
ALWI void rmsnorm_gate_block(Buffers& bufs, uint32_t v_eps_bits, RowScope scope) {
    square_block<cb_o_acc, cb_sq_o>(0, Vt, scope);
    row_sum<cb_sq_o, cb_rinv_o>(bufs, bufs.sq_o, bufs.rinv_o, Vt, RsqrtPost{v_eps_bits, SQRT_V_BITS, scope});
    cb_wait_front(cb_rinv_o, 1);
    cb_wait_front(cb_z, Vt);
    cb_reserve_back(cb_out_rows, Vt);
    for (uint32_t vt = 0; vt < Vt; ++vt) {
        tile_regs_acquire();
        begin_loads(cb_o_acc);
        load(cb_o_acc, vt, 0);
        begin_loads(cb_rinv_o);
        load(cb_rinv_o, 0, 1);
        begin_loads(cb_norm_w);
        load(cb_norm_w, vt, 2);
        load(cb_z, vt, 3);
        sfpu_bcast_col_init();
        mul_bcast_col_scoped(0, 1, scope);
        sfpu_bcast_row_init();
        bcast_row_scoped<ckernel::BinaryOp::MUL>(0, 2, scope);
        silu_tile_init();
        silu_scoped(3, scope);
        mul_binary_tile_init();
        binary_scoped<ckernel::BinaryOp::MUL>(0, 3, 0, scope);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_out_rows, vt);
        tile_regs_release();
    }
    cb_push_back(cb_out_rows, Vt);
    cb_pop_front(cb_o_acc, Vt);
    cb_pop_front(cb_z, Vt);
    cb_pop_front(cb_rinv_o, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t first_pair = get_arg_val<uint32_t>(0);
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t v_eps_bits = get_arg_val<uint32_t>(3);  // V * epsilon

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_state_mm, cb_reads);
    Buffers bufs;

    for (uint32_t pair = first_pair; pair < first_pair + num_pairs; ++pair) {
        const uint32_t seq_lo = (pair / Hv) * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        cb_wait_front(cb_conv_w, conv_w_pages);
        cb_wait_front(cb_gate_w, 2);
        cb_wait_front(cb_norm_w, Vt);

        uint32_t token = 0;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            const RowScope scope = RowScope::of(lo - base_row, hi - base_row);
            {
                DeviceZoneScopedN("Z_GATES");
                gates_block(scope);
            }
            {
                DeviceZoneScopedN("Z_CONV");
                if constexpr (CONV_ON_FPU) {
                    conv_block_fpu(scope);
                } else {
                    conv_block(scope);
                }
            }
            {
                DeviceZoneScopedN("Z_L2N");
                l2norm_block(bufs, scope);
            }

            cb_reserve_back(cb_o_acc, Vt);
            for (uint32_t row = lo; row < hi; ++row, ++token) {
                DeviceZoneScopedN("Z_STEP");
                token_step(bufs, token == 0, token == T - 1, row == lo, scope);
            }
            cb_push_back(cb_o_acc, Vt);
            cb_pop_front(cb_kq_rows, 2 * Kt);
            cb_pop_front(cb_knT, Kt);
            cb_pop_front(cb_d, 1);
            cb_pop_front(cb_v, Vt);

            {
                DeviceZoneScopedN("Z_RMSN");
                rmsnorm_gate_block(bufs, v_eps_bits, scope);
            }
        }
        cb_pop_front(cb_conv_w, conv_w_pages);
        cb_pop_front(cb_gate_w, 2);
        cb_pop_front(cb_norm_w, Vt);
    }
}
