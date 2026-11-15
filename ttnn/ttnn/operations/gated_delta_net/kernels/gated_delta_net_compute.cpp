// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute (regimes R-STEP, R-VSPLIT and R-CHUNK, op_design.md).
//
// Per unit (pair, piece of Vb state columns; R-STEP is one piece of Vt), per tile row: gates_block ->
// conv_block -> l2norm_block -> recurrence_block -> rmsnorm_gate_block. recurrence_block is one token
// per step (read matmul, token vector, state update), or with CHUNKED (R-CHUNK, multi-token calls) one
// closed-form chunk step over the tile row (F1, math.md):
//   c = cumsum(g), D[s, s'] = exp(c[s] - c[s']) (s' <= s), L = diag(beta) (D * Kn Kn^T) strictly lower,
//   (I + L) U = diag(beta) (V - diag(exp(c)) Kn S),  O = diag(exp(c)) Qn S + ((Qn Kn^T) * D) U,
//   S <- exp(c_last) S + Kn^T (diag(exp(c_last - c)) U).
// c, D, the chunk vectors and the solve (triangle_solve_tile, forward substitution on the SFPU) are fp32;
// the state decay is the fp32 SFPU scalar multiply of the token step; the products with the state and
// with U run on the matrix engine like the token step's reads and increment.
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
// R-VSPLIT combine: the RMS-norm sum of o^2 over V is the sum of the pieces' partial row sums, which the
// writer all-gathers; the PIECES partial tiles are added in one SFPU window in piece order, so every
// piece of a pair derives the same rsqrt.

#include <cstdint>

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
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/cumsum.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/triangle_solve.h"
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
constexpr uint32_t cb_osum_part = GDN_CB_OSUM_PART;
constexpr uint32_t cb_osum_gather = GDN_CB_OSUM_GATHER;
constexpr uint32_t cb_gram = GDN_CB_GRAM;
constexpr uint32_t cb_chunk_vec = GDN_CB_CHUNK_VEC;
constexpr uint32_t cb_chunk_mat = GDN_CB_CHUNK_MAT;
constexpr uint32_t cb_uw = GDN_CB_UW;
constexpr uint32_t cb_tri_mask = GDN_CB_TRI_MASK;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t COEF_ALPHA = 0;      // alpha_p in every row
constexpr uint32_t COEF_BETA_SEL = 1;   // beta_p at row p, 0 elsewhere
constexpr uint32_t COEF_ALPHA_SEL = 2;  // alpha_p at row p, 0 elsewhere
// R-CHUNK pages. Chunk coefficients (writer, column 0, 0 on rows that are no token):
constexpr uint32_t CHUNK_G = 0;     // g
constexpr uint32_t CHUNK_BETA = 1;  // beta
// Chunk vectors (compute), in push order:
constexpr uint32_t CV_D = 0;    // D[s, s'] = exp(c[s] - c[s']) for s' <= s, else 0
constexpr uint32_t CV_C = 1;    // c[s] in every column of row s
constexpr uint32_t CV_EC = 2;   // exp(c[s])
constexpr uint32_t CV_W = 3;    // exp(c_last - c[s])
constexpr uint32_t CV_BEC = 4;  // beta[s] exp(c[s]) (column 0)
static_assert(CV_BEC + 1 == GDN_CHUNK_VEC_PAGES, "chunk vector pages");
// Chunk matrices (compute):
constexpr uint32_t CM_L = 0;   // beta[s] D[s, s'] (kn[s] . kn[s']): the solve reads only its strict lower part
constexpr uint32_t CM_QD = 1;  // (qn[s] . kn[s']) D[s, s']
static_assert(CM_QD + 1 == GDN_CHUNK_MAT_PAGES, "chunk matrix pages");

constexpr uint32_t ONE_BITS = 0x3f800000u;
constexpr uint32_t SOFTPLUS_THRESHOLD_BITS = 0x41a00000u;  // 20.0f
constexpr uint32_t L2_EPS_BITS = 0x358637bdu;              // 1e-6f

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vb = get_compile_time_arg_val(1);  // v / state column tiles of this core's piece
constexpr uint32_t Hv = get_compile_time_arg_val(2);
constexpr uint32_t MM_SUBBLOCK_W = get_compile_time_arg_val(3);
constexpr uint32_t SQRT_V_BITS = get_compile_time_arg_val(4);
constexpr uint32_t Q_SCALE_BITS = get_compile_time_arg_val(5);
constexpr bool CONV_ON_FPU = get_compile_time_arg_val(6) != 0;
constexpr uint32_t PIECES = get_compile_time_arg_val(7);             // column pieces per pair (R-VSPLIT when > 1)
constexpr uint32_t WINDOW_GROUP_COLS = get_compile_time_arg_val(8);  // columns per cb_window hand-off
constexpr bool CHUNKED = get_compile_time_arg_val(9) != 0;           // R-CHUNK: one chunk step per tile row

constexpr uint32_t largest_divisor_at_most(uint32_t n, uint32_t cap) {
    uint32_t best = 1;
    for (uint32_t d = 1; d <= n && d <= cap; ++d) {
        if (n % d == 0) {
            best = d;
        }
    }
    return best;
}
// FPU conv columns per DEST window: a divisor of the window group, so a batch never straddles the
// cb_window wrap (tiles are addressed by index from the read pointer).
constexpr uint32_t CONV_BATCH = largest_divisor_at_most(WINDOW_GROUP_COLS, ckl::DEST_AUTO_LIMIT);
static_assert(PIECES <= ckl::DEST_AUTO_LIMIT, "the partial-sum combine holds every piece in one DEST window");
// FPU conv reads raw conv_weight tiles (tap j at row j); the SFPU conv one row-0 tile per tap.
constexpr uint32_t conv_w_pages = CONV_ON_FPU ? (2 * Kt + Vb) : GDN_CONV_TAPS * (2 * Kt + Vb);
constexpr uint32_t n_state = Kt * Vb;
constexpr uint32_t pair_cols = 2 * Kt + Vb;
// State tiles per update batch: every DEST slot of a window.
constexpr uint32_t UPDATE_BATCH = ckl::DEST_AUTO_LIMIT < n_state ? ckl::DEST_AUTO_LIMIT : n_state;
static_assert(ckl::DEST_AUTO_LIMIT >= 4, "raw windows use 4 DEST slots");
static_assert(Vb % MM_SUBBLOCK_W == 0, "read-matmul subblock must divide Vb");

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

// Word index of element (r, c) of an fp32 tile (four 16x16 faces).
constexpr uint32_t elem_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

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
    // R-CHUNK runs whole tiles: its calls are prefill (tile rows mostly full), and dropping the band
    // variants keeps the math thread's code inside the kernel-config text budget.
    static RowScope of(uint32_t row_lo, uint32_t row_hi) {
        if constexpr (CHUNKED) {
            return {FULL_TILE};
        }
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
// CHUNKED: g = -exp(A_log) * softplus(a + dt_bias) instead of alpha (the chunk step takes cumulative sums of g).
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
    for (uint32_t pass = 0; pass < (CHUNKED ? 1 : 2); ++pass) {
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
    cb_reserve_back(cb_v, Vb);
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
    cb_push_back(cb_v, Vb);
}

// ---------------------------------------------------------------- conv_block_fpu
// Same contract as conv_block: DEST = sum_j X_j (*)row W_j by FPU ELWMUL accumulation (DEST is zero
// after acquire), then silu in the same window. Exact for bf16 operands; products accumulate in fp32.
// CONV_BATCH columns per window: one init / reconfig / sync per batch, not per column.
ALWI void conv_block_fpu(RowScope scope) {
    cb_reserve_back(cb_y_qk, 2 * Kt);
    cb_reserve_back(cb_v, Vb);
    static_assert(pair_cols % CONV_BATCH == 0, "window groups tile the unit's columns");
    constexpr uint32_t n = CONV_BATCH;
    for (uint32_t col0 = 0; col0 < pair_cols; col0 += n) {
        cb_wait_front(cb_window, GDN_CONV_TAPS * n);
        tile_regs_acquire();
        reconfig_data_format(cb_window, cb_conv_w);
        mul_bcast_rows_init(cb_window, cb_conv_w);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t tap = 0; tap < GDN_CONV_TAPS; ++tap) {
                mul_tiles_bcast_rows(cb_window, cb_conv_w, GDN_CONV_TAPS * i + tap, col0 + i, i, tap);
            }
        }
        silu_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            silu_scoped(i, scope);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_y_qk);  // cb_y_qk and cb_v are both Float32
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t col = col0 + i;
            if (col < 2 * Kt) {
                pack_at(i, cb_y_qk, col);
            } else {
                pack_at(i, cb_v, col - 2 * Kt);
            }
        }
        tile_regs_release();
        cb_pop_front(cb_window, GDN_CONV_TAPS * n);
    }
    cb_push_back(cb_y_qk, 2 * Kt);
    cb_push_back(cb_v, Vb);
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
    if constexpr (!CHUNKED) {
        cb_reserve_back(cb_sq, Kt);
    }
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
        if constexpr (!CHUNKED) {
            mul_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::MUL>(0, 2, 1, scope);  // qn * kn
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(2, cb_kq_rows, kt);
        pack_at(0, cb_kq_rows, Kt + kt);
        if constexpr (!CHUNKED) {
            pack_at(1, cb_sq, kt);
        }
        tile_regs_release();
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_pop_front(cb_rinv, 2);
    cb_pop_front(cb_y_qk, 2 * Kt);
    if constexpr (!CHUNKED) {
        // d = qn . kn per token (the token step's output term); the chunk step has it inside Qn Kn^T.
        cb_push_back(cb_sq, Kt);
        row_sum<cb_sq, cb_d>(bufs, bufs.sq, bufs.d, Kt, ckl::NoOp{});
    }

    cb_wait_front(cb_kq_rows, 2 * Kt);
    cb_reserve_back(cb_knT, Kt);
    reconfig_data_format_srca(cb_kq_rows);
    transpose_init(cb_kq_rows);
    for (uint32_t kt0 = 0; kt0 < Kt; kt0 += ckl::DEST_AUTO_LIMIT) {
        const uint32_t n = (Kt - kt0) < ckl::DEST_AUTO_LIMIT ? (Kt - kt0) : ckl::DEST_AUTO_LIMIT;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            transpose_tile(cb_kq_rows, kt0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_knT, kt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(i, cb_knT, kt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_knT, Kt);
    if constexpr (!CHUNKED) {
        cb_wait_front(cb_d, 1);
    }
    cb_wait_front(cb_v, Vb);
    cb_wait_front(cb_knT, Kt);
}

// ---------------------------------------------------------------- state read / update (shared)
// [rk; ro] = [kn; qn] x S_{in} -> cb_reads (2 x Vb tiles). S_in is S0 for the pair's first step.
ALWI void read_state(Buffers& bufs, bool first_in_pair) {
    const uint32_t src = first_in_pair ? cb_state_in : cb_state;
    const uint32_t src_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    DataflowBuffer& src_mm_buf = first_in_pair ? bufs.state_in_mm : bufs.state_mm;
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
        ckl::MatmulBlockShape::of(2, Vb / MM_SUBBLOCK_W, 1, MM_SUBBLOCK_W, Kt, 1));
}

// S_out = decay S_in (SFPU, fp32 scalar) + kn^T x u_cb (matmul accumulated onto DEST); S_out goes to
// cb_state_out after the pair's last step, else back into cb_state. Pops S_in.
ALWI void update_state(bool first_in_pair, bool last_in_pair, uint32_t u_cb, uint32_t decay_bits) {
    const uint32_t src = first_in_pair ? cb_state_in : cb_state;
    const uint32_t src_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    const uint32_t dst = last_in_pair ? cb_state_out : cb_state;
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
            mul_unary_tile(i, decay_bits);
        }
        wait_sfpu_before_fpu();
        reconfig_data_format(u_cb, cb_knT);
        matmul_init(cb_knT, u_cb);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = first + i;
            matmul_tiles(cb_knT, u_cb, tile / Vb, tile % Vb, i);
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
}

// ---------------------------------------------------------------- one token step
ALWI void token_step(Buffers& bufs, bool first_in_pair, bool last_in_pair, bool first_in_row, RowScope scope) {
    // read (5a): [rk; ro] = [kn; qn] x S_{t-1}
    read_state(bufs, first_in_pair);
    // token vector (5b): u = beta_sel (*)col (v - alpha rk); o = alpha_sel (*)col ro + d (*)col u
    cb_wait_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
    // alpha is one scalar per token (element (0, 0) of the alpha page): fp32 scalar SFPU multiplies
    // here and in the state decay, with no broadcast tile in DEST.
    const uint32_t alpha_bits = read_tile_value(cb_tok_coef, COEF_ALPHA, 0);
    cb_wait_front(cb_reads, 2 * Vb);
    cb_reserve_back(cb_u, Vb);
    cb_reserve_back(cb_u_f32, Vb);
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
    cb_pop_front(cb_reads, 2 * Vb);
    // update (5c): S_t = alpha S_{t-1} (SFPU, fp32) + kn^T u (matmul accumulated onto DEST)
    update_state(first_in_pair, last_in_pair, cb_u, alpha_bits);
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
    cb_pop_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
}

// ---------------------------------------------------------------- one chunk step (R-CHUNK)
// Chunk vectors from g and beta (column 0 of the writer's coefficient pages, 0 on rows that are no token,
// so c is flat there and those rows neither decay nor write). row_last: tile-local last token row.
// Returns exp(c_last), the chunk's state decay.
ALWI uint32_t chunk_vectors(uint32_t row_last) {
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
    tile_regs_release();
    cb_push_back(cb_chunk_vec, GDN_CHUNK_VEC_PAGES - 2);
    cb_wait_front(cb_chunk_vec, GDN_CHUNK_VEC_PAGES);
    return read_tile_value(cb_chunk_vec, CV_EC, elem_index(row_last, 0));
}

ALWI void chunk_step(Buffers& bufs, bool first_in_pair, bool last_in_pair, uint32_t row_last) {
    const uint32_t decay_bits = chunk_vectors(row_last);

    // [Kn Kn^T; Qn Kn^T] (2 tiles), then L and QD (fp32 SFPU products with D and beta).
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
    cb_reserve_back(cb_chunk_mat, GDN_CHUNK_MAT_PAGES);
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
    pack_at(1, cb_chunk_mat, CM_QD);
    tile_regs_release();
    cb_push_back(cb_chunk_mat, GDN_CHUNK_MAT_PAGES);
    cb_pop_front(cb_gram, 2);

    read_state(bufs, first_in_pair);

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
    update_state(first_in_pair, last_in_pair, cb_uw, decay_bits);
    cb_pop_front(cb_uw, Vb);
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
    cb_pop_front(cb_chunk_mat, GDN_CHUNK_MAT_PAGES);
    cb_pop_front(cb_chunk_vec, GDN_CHUNK_VEC_PAGES);
    cb_pop_front(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
}

// ---------------------------------------------------------------- rmsnorm_gate_block
// out = ((o (*)col rsqrt(mean o^2 + eps)) (*)row w) * silu(z), with
// rsqrt(mean o^2 + eps) = sqrt(V) * rsqrt(sum o^2 + V eps); sum over the whole V (all pieces).
ALWI void rmsnorm_gate_block(Buffers& bufs, uint32_t v_eps_bits, RowScope scope) {
    square_block<cb_o_acc, cb_sq_o>(0, Vb, scope);
    const RsqrtPost rsqrt_post{v_eps_bits, SQRT_V_BITS, scope};
    if constexpr (PIECES == 1) {
        row_sum<cb_sq_o, cb_rinv_o>(bufs, bufs.sq_o, bufs.rinv_o, Vb, rsqrt_post);
    } else {
        DataflowBuffer osum_part{cb_osum_part};
        row_sum<cb_sq_o, cb_osum_part>(bufs, bufs.sq_o, osum_part, Vb, ckl::NoOp{});
        cb_wait_front(cb_osum_gather, PIECES);
        cb_reserve_back(cb_rinv_o, 1);
        tile_regs_acquire();
        begin_loads(cb_osum_gather);
        for (uint32_t s = 0; s < PIECES; ++s) {
            load(cb_osum_gather, s, s);
        }
        add_binary_tile_init();
        for (uint32_t s = 1; s < PIECES; ++s) {
            binary_scoped<ckernel::BinaryOp::ADD>(0, s, 0, scope);
        }
        rsqrt_post(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_rinv_o, 0);
        tile_regs_release();
        cb_push_back(cb_rinv_o, 1);
        cb_pop_front(cb_osum_gather, PIECES);
    }
    cb_wait_front(cb_rinv_o, 1);
    cb_wait_front(cb_z, Vb);
    cb_reserve_back(cb_out_rows, Vb);
    for (uint32_t vt = 0; vt < Vb; ++vt) {
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
    cb_push_back(cb_out_rows, Vb);
    cb_pop_front(cb_o_acc, Vb);
    cb_pop_front(cb_z, Vb);
    cb_pop_front(cb_rinv_o, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t v_eps_bits = get_arg_val<uint32_t>(3);  // V * epsilon

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_state_mm, cb_reads);
    Buffers bufs;
    if constexpr (CHUNKED) {
        cb_wait_front(cb_tri_mask, 1);  // reader-built once, held for the kernel
    }

    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t seq_lo = (unit / PIECES / Hv) * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        cb_wait_front(cb_conv_w, conv_w_pages);
        cb_wait_front(cb_gate_w, 2);
        cb_wait_front(cb_norm_w, Vb);

        uint32_t token = 0;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            const RowScope scope = RowScope::of(lo - base_row, hi - base_row);
            gates_block(scope);
            if constexpr (CONV_ON_FPU) {
                conv_block_fpu(scope);
            } else {
                conv_block(scope);
            }
            l2norm_block(bufs, scope);

            cb_reserve_back(cb_o_acc, Vb);
            if constexpr (CHUNKED) {
                chunk_step(bufs, token == 0, token + (hi - lo) == T, hi - 1 - base_row);
                token += hi - lo;
            } else {
                for (uint32_t row = lo; row < hi; ++row, ++token) {
                    token_step(bufs, token == 0, token == T - 1, row == lo, scope);
                }
                cb_pop_front(cb_d, 1);
            }
            cb_push_back(cb_o_acc, Vb);
            cb_pop_front(cb_kq_rows, 2 * Kt);
            cb_pop_front(cb_knT, Kt);
            cb_pop_front(cb_v, Vb);

            rmsnorm_gate_block(bufs, v_eps_bits, scope);
        }
        cb_pop_front(cb_conv_w, conv_w_pages);
        cb_pop_front(cb_gate_w, 2);
        cb_pop_front(cb_norm_w, Vb);
    }
}
