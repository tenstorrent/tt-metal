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
// R-BCAST (decode R-VSPLIT): the key head's q (k) injector convolves and L2-normalizes only q (k) and hands qn
// (kn) to its writer for the multicast; a receiver convolves only its v columns. On every such core the writer
// lands kn / qn in cb_kq_rows, and compute derives d = qn . kn and kn^T from them (qk_finish_block).
// R-PREP (ROLE, chunked R-VSPLIT): a prep core runs, per tile row of its key head, gates, conv, L2 norm, gram_block
// and head_prep for each of the r value heads, packing one prescaled head block per head into cb_prep_send, then
// (one row later) the RMS norm of the state cores' o; a state core runs chunk_state_prepped on its inbox rows
// (landed by its reader): the same F1 step with the row scalings and (I + L)^-1 applied on the prep core.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/copy_dest_values.h"
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
constexpr uint32_t D_PAGES = GDN_D_PAGES;
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
constexpr uint32_t cb_qk_stage = GDN_CB_QK_STAGE;
constexpr uint32_t cb_kq_rows_f32 = GDN_CB_KQ_ROWS_F32;
constexpr uint32_t cb_prep_send = GDN_CB_PREP_SEND;
constexpr uint32_t cb_in_kq = GDN_CB_IN_KQ;
constexpr uint32_t cb_in_knt = GDN_CB_IN_KNT;
constexpr uint32_t cb_in_mat = GDN_CB_IN_MAT;
constexpr uint32_t cb_in_vec = GDN_CB_IN_VEC;
constexpr uint32_t cb_in_v = GDN_CB_IN_V;
constexpr uint32_t cb_v_prep = GDN_CB_V_PREP;
constexpr uint32_t cb_o_in = GDN_CB_O_IN;
constexpr uint32_t cb_zw = GDN_CB_ZW;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t COEF_ALPHA_SEL = 0;  // alpha_p at row p, 0 elsewhere
constexpr uint32_t COEF_BETA_SEL = 1;   // beta_p at row p, 0 elsewhere
constexpr uint32_t COEF_SEL = 2;        // 1 at row p, 0 elsewhere
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
constexpr uint32_t CV_WT = GDN_CHUNK_VEC_PAGES;  // R-PREP prep core: exp(c_last - c)^T after the chunk vectors
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
constexpr bool QK_SHARED = get_compile_time_arg_val(10) != 0;        // R-BCAST: q/k prepped once per key head
constexpr uint32_t ROLE = get_compile_time_arg_val(11);   // GDN_ROLE_*
constexpr uint32_t RATIO = get_compile_time_arg_val(12);  // R-PREP: value heads per prep group
constexpr bool PREP = ROLE == GDN_ROLE_PREP;              // Vb is then the group's RATIO * Vt v columns
constexpr bool STATE = ROLE == GDN_ROLE_STATE;
constexpr bool QK_CORE = ROLE == GDN_ROLE_QK;  // R-QKCORE: conv + L2 norm of one key head's q or k, every sequence
// R-PREP q / k core (PREP_SHARE_QK): conv + L2 norm of one key head's q or k for its tile rows; the prep cores it
// writes qn / kn to have QK_SHARED (they convolve only their v columns and take kn / qn from cb_kq_rows).
constexpr bool PREP_QK = ROLE == GDN_ROLE_PREP_QK;
constexpr bool PREP_SHARED_QK = PREP && QK_SHARED;
static_assert(!(QK_SHARED && CHUNKED) || PREP, "R-BCAST is a decode regime");
// R-QKCORE pair cores whose units share one key head: kn^T and d are computed once and held for every unit (host knob
// SHARE_QK_FINISH); with UNIT_LAG a unit's RMS norm runs after the next unit's token step, so the last state is final
// before the norms (host knob UNIT_LAG; needs two units of o, z and norm weights in flight).
constexpr bool HOLD_QK = GDN_HOLD_QK != 0;
constexpr bool UNIT_LAG = GDN_UNIT_LAG != 0;
// Final-state tiles of pair `pair` start at tile (pair * STATE_STAGGER) mod n_state (rotated through the banks); the
// writer applies the same rotation. 0 everywhere but R-QKCORE pair cores.
constexpr uint32_t STATE_STAGGER = GDN_STATE_STAGGER;
static_assert(CHUNKED || ROLE == GDN_ROLE_LOCAL || QK_CORE, "R-PREP is a chunked regime");
// A prep core's conv output v is scaled by beta per head (head_prep) before it is sent.
constexpr uint32_t cb_v_out = PREP ? cb_v_prep : cb_v;
// R-PREP head block (descriptor IN_MAT_PAGES), in pack order: [exp(c) | QD, M | beta exp(c) (*) kn, exp(c) (*) qn |
// kn^T (*) exp(c_last - c)^T | beta (*) v]; the state core's inboxes hold these parts. Each part is pushed as it is
// packed, so the writer sends it while compute packs the next.
constexpr uint32_t HB_EC = 0;
constexpr uint32_t HB_QD = HB_EC + GDN_IN_VEC_PAGES;
constexpr uint32_t HB_M = HB_QD + 1;
constexpr uint32_t HB_KQ = HB_QD + GDN_IN_MAT_PAGES;
constexpr uint32_t HB_KNT = HB_KQ + 2 * Kt;
constexpr uint32_t HB_V = HB_KNT + Kt;
constexpr uint32_t HEAD_V_TILES = PREP ? Vb / RATIO : 1;  // a prep core's Vt columns of one value head
constexpr uint32_t HEAD_BLOCK = HB_V + HEAD_V_TILES;
// State core inbox cb_in_mat pages (the head block's QD, M).
constexpr uint32_t IN_MAT_QD = 0;
constexpr uint32_t IN_MAT_M = HB_M - HB_QD;

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

// The unit's columns: q/k columns [qk_lo, qk_lo + qk_cols) of the pair's 2Kt (all of them unless R-BCAST),
// then the Vb v columns; cb_y_qk holds the qk_cols prepped q/k tiles.
struct UnitCols {
    uint32_t qk_lo;
    uint32_t qk_cols;
    uint32_t count() const { return qk_cols + Vb; }
    uint32_t pair_col(uint32_t local) const { return local < qk_cols ? qk_lo + local : 2 * Kt + (local - qk_cols); }
};

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

// Valid token rows [row_lo, row_hi) of a tile row (tile-local). When they sit in fewer than all eight 4-row bands,
// the row-local SFPU ops run on those bands of both faces only (4 of 32 vector ops per band, sfpu_tile_scope); rows
// outside them keep their pre-op DEST values, which no valid row reads.
struct RowScope {
    uint32_t band;       // first band index 0..7, or FULL_TILE
    uint32_t bands = 1;  // bands [band, band + bands)
    static constexpr uint32_t FULL_TILE = 0xffffffffu;
    // R-CHUNK runs whole tiles: its calls are prefill (tile rows mostly full), and dropping the band
    // variants keeps the math thread's code inside the kernel-config text budget.
    // R-BCAST is decode (one token row per tile row): always one band, so its builds drop the whole-tile variants.
    static RowScope of(uint32_t row_lo, uint32_t row_hi) {
        if constexpr (CHUNKED) {
            return {FULL_TILE};
        }
        if constexpr (QK_SHARED) {
            return {row_lo >> 2};
        }
        const uint32_t bands = ((row_hi - 1) >> 2) - (row_lo >> 2) + 1;
        if (bands == 1 || (QK_CORE && bands < 8)) {
            return {row_lo >> 2, bands};
        }
        return {FULL_TILE};
    }
    bool full() const { return CHUNKED || (!QK_SHARED && band == FULL_TILE); }
    // Several bands only on a q / k core (rows 0..B-1): every other build scopes one band (else the whole tile) and
    // takes it at compile time, as its math thread sits near the kernel-config text budget.
    uint32_t band_end() const { return QK_CORE ? band + bands : band + 1; }
};

// One SFPU body of ITERATIONS=2 (one 4-row band, both column parities) on each band of the scope, both faces of its
// half.
template <typename Body>
ALWI void sfpu_band(uint32_t dst, RowScope scope, Body body) {
#ifdef TRISC_MATH
    for (uint32_t band = scope.band; band < scope.band_end(); ++band) {
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
    }
#endif
}

ALWI void silu_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        silu_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, 2>(); });
#endif
    }
}

// Row-local unary SFPU ops on the scope's rows (full-tile API call when unscoped).
ALWI void softplus_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        softplus_tile(dst, ONE_BITS, ONE_BITS, SOFTPLUS_THRESHOLD_BITS);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] {
            ckernel::sfpu::calculate_softplus<APPROX, DST_ACCUM_MODE, 2>(ONE_BITS, ONE_BITS, SOFTPLUS_THRESHOLD_BITS);
        });
#endif
    }
}

ALWI void exp_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        exp_tile<false>(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 2>(); });
#endif
    }
}

ALWI void negative_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        negative_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::_calculate_negative_<APPROX, 2>(); });
#endif
    }
}

ALWI void sigmoid_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        sigmoid_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::calculate_sigmoid<false, DST_ACCUM_MODE, 2>(); });
#endif
    }
}

ALWI void square_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        square_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::calculate_square<APPROX, DST_ACCUM_MODE, 2>(); });
#endif
    }
}

template <int BINOP_MODE>
ALWI void binop_scalar_scoped(uint32_t dst, uint32_t scalar_bits, RowScope scope) {
    if (scope.full()) {
        if constexpr (BINOP_MODE == ADD_UNARY) {
            add_unary_tile(dst, scalar_bits);
        } else {
            mul_unary_tile(dst, scalar_bits);
        }
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [=] {
            ckernel::sfpu::calculate_binop_with_scalar<APPROX, BINOP_MODE, 2, DST_ACCUM_MODE>(scalar_bits);
        });
#endif
    }
}

ALWI void rsqrt_scoped(uint32_t dst, RowScope scope) {
    if (scope.full()) {
        rsqrt_tile(dst);
    } else {
#ifdef TRISC_MATH
        sfpu_band(dst, scope, [] { ckernel::sfpu::calculate_rsqrt<APPROX, 2, DST_ACCUM_MODE, false>(); });
#endif
    }
}

// fp32 SFPU binary (DEST[d0] op DEST[d1] -> DEST[out]) on the scope's rows.
template <ckernel::BinaryOp OP>
ALWI void binary_scoped(uint32_t d0, uint32_t d1, uint32_t out, RowScope scope) {
    if (scope.full()) {
        if constexpr (OP == ckernel::BinaryOp::ADD) {
            add_binary_tile(d0, d1, out);
        } else if constexpr (OP == ckernel::BinaryOp::SUB) {
            sub_binary_tile(d0, d1, out);
        } else {
            mul_binary_tile(d0, d1, out);
        }
    } else {
#ifdef TRISC_MATH
        sfpu_band(0, scope, [=] { ckernel::sfpu::calculate_sfpu_binary<APPROX, OP, 2, DST_ACCUM_MODE>(d0, d1, out); });
#endif
    }
}

// DEST[data] *= column 0 of DEST[col] (broadcast along the row), on the scope's rows.
// Band path: the LLK's own per-band body of sfpu_mul_bcast_col (needs sfpu_bcast_col_init).
ALWI void mul_bcast_col_scoped(uint32_t data, uint32_t col, RowScope scope) {
    if (scope.full()) {
        sfpu_mul_bcast_col(data, col);
    } else {
#ifdef TRISC_MATH
        using namespace ckernel::sfpu;
        for (uint32_t band = scope.band; band < scope.band_end(); ++band) {
            _llk_math_eltwise_sfpu_start_(0);
            const uint32_t left = (band >= 4 ? FACE2_BASE : FACE0_BASE) + (band & 3) * ROW_BAND_STRIDE;
            const uint32_t right = left + (FACE1_BASE - FACE0_BASE);
            _process_col_bcast_row_band_<ckernel::BinaryOp::MUL>(
                col * DEST_TILE_SIZE_RAW + left, left, right, data * DEST_TILE_SIZE_RAW, data * DEST_TILE_SIZE_RAW);
            _llk_math_eltwise_sfpu_done_();
        }
#endif
    }
}

// DEST[data] op= row 0 of DEST[row] (broadcast down the column), on the scope's rows.
// Band path: the LLK's own per-band body of sfpu_*_bcast_row (needs sfpu_bcast_row_init).
template <ckernel::BinaryOp OP>
ALWI void bcast_row_scoped(uint32_t data, uint32_t row, RowScope scope) {
    if (scope.full()) {
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
        for (uint32_t band = scope.band; band < scope.band_end(); ++band) {
            _process_row_bcast_data_band_<OP>(
                data * DEST_TILE_SIZE_RAW,
                data * DEST_TILE_SIZE_RAW,
                band >= 4 ? FACE2_BASE : FACE0_BASE,
                (band & 3) * ROW_BAND_STRIDE);
        }
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
// y = silu(sum_j X_j (*)row W_j) per column tile of the unit; q, k -> cb_y_qk, v -> cb_v.
ALWI void conv_block(RowScope scope, UnitCols cols, uint32_t first, uint32_t last) {
    const bool qk_range = first < cols.qk_cols;
    const bool v_range = last > cols.qk_cols;
    if (qk_range) {
        cb_reserve_back(cb_y_qk, cols.qk_cols);
    }
    if (v_range) {
        cb_reserve_back(cb_v_out, Vb);
    }
    for (uint32_t local = first; local < last; ++local) {
        const uint32_t col = cols.pair_col(local);
        cb_wait_front(cb_window, GDN_CONV_TAPS);
        cb_reserve_back(cb_conv_acc, 1);
        // Taps (0, 1) seed the partial sum, taps (2, 3) add onto it by packer L1 accumulation.
#pragma GCC unroll 1
        for (uint32_t tap0 = 0; tap0 < GDN_CONV_TAPS; tap0 += 2) {
            conv_two_taps(local, tap0);
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
            pack_first(0, cb_y_qk, col - cols.qk_lo);
        } else {
            pack_first(0, cb_v_out, col - 2 * Kt);
        }
        tile_regs_release();
        cb_pop_front(cb_conv_acc, 1);
    }
    if (qk_range) {
        cb_push_back(cb_y_qk, cols.qk_cols);
    }
    if (v_range) {
        cb_push_back(cb_v_out, Vb);
    }
}

// ---------------------------------------------------------------- conv_block_fpu
// Same contract as conv_block: DEST = sum_j X_j (*)row W_j by FPU ELWMUL accumulation (DEST is zero
// after acquire), then silu in the same window. Exact for bf16 operands; products accumulate in fp32.
// CONV_BATCH columns per window: one init / reconfig / sync per batch, not per column. An R-BCAST unit's columns
// are one window group of one tile row, so its ragged last batch cannot straddle the wrap.
// Unit columns [first, last): a range starts at 0 or at qk_cols and ends at qk_cols or at the end, so cb_y_qk
// and cb_v are each filled by one range.
ALWI void conv_block_fpu(RowScope scope, UnitCols cols, uint32_t first, uint32_t last) {
    const bool qk_range = first < cols.qk_cols;
    const bool v_range = last > cols.qk_cols;
    if (qk_range) {
        cb_reserve_back(cb_y_qk, cols.qk_cols);
    }
    if (v_range) {
        cb_reserve_back(cb_v_out, Vb);
    }
    static_assert(QK_CORE || PREP_QK || pair_cols % CONV_BATCH == 0, "window groups tile the unit's columns");
    for (uint32_t col0 = first; col0 < last; col0 += CONV_BATCH) {
        const uint32_t n = last - col0 < CONV_BATCH ? last - col0 : CONV_BATCH;
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
        // Both Float32; a core with no q / k columns (PREP_SHARE_QK prep core) allocates no cb_y_qk.
        pack_reconfig_data_format(cols.qk_cols != 0 ? cb_y_qk : cb_v_out);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t col = cols.pair_col(col0 + i);
            if (col < 2 * Kt) {
                pack_at(i, cb_y_qk, col - cols.qk_lo);
            } else {
                pack_at(i, cb_v_out, col - 2 * Kt);
            }
        }
        tile_regs_release();
        cb_pop_front(cb_window, GDN_CONV_TAPS * n);
    }
    if (qk_range) {
        cb_push_back(cb_y_qk, cols.qk_cols);
    }
    if (v_range) {
        cb_push_back(cb_v_out, Vb);
    }
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

ALWI void kn_transpose_block(Buffers& bufs);

// ---------------------------------------------------------------- l2norm_block
// rinv_q = K^-1/2 rsqrt(sum q^2 + 1e-6), rinv_k = rsqrt(sum k^2 + 1e-6);
// kn -> cb_kq_rows[0, Kt), qn -> cb_kq_rows[Kt, 2Kt), kn^T -> cb_knT, then d (kn_transpose_block).
ALWI void l2norm_block(Buffers& bufs, RowScope scope) {
    // q then k: one loop body keeps a single expansion of the square chain and the reduce.
#pragma GCC unroll 1
    for (uint32_t part = 0; part < 2; ++part) {
        l2_inverse_norm(bufs, part * Kt, part == 0 ? Q_SCALE_BITS : ONE_BITS, scope);
    }

    cb_wait_front(cb_rinv, 2);
    cb_reserve_back(cb_kq_rows, 2 * Kt);

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
        tile_regs_commit();
        tile_regs_wait();
        pack_first(2, cb_kq_rows, kt);
        pack_at(0, cb_kq_rows, Kt + kt);
        tile_regs_release();
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_pop_front(cb_rinv, 2);
    cb_pop_front(cb_y_qk, 2 * Kt);
    kn_transpose_block(bufs);
}

// kn^T per tile from the kn row of cb_kq_rows -> cb_knT; token step: [kn; qn] x kn^T -> cb_d (tile 1 holds
// d = qn . kn of token row p at (p, p)). Then holds what the token loop reads.
ALWI void kn_transpose_block(Buffers& bufs) {
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
    cb_wait_front(cb_knT, Kt);

    if constexpr (!CHUNKED) {
        // One matmul instead of Kt SFPU products and a row reduction; d only scales u in the output (19-bit operands).
        DataflowBuffer knT{cb_knT};
        ckl::matmul_block<
            false,
            false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::InputPolicy::NoWaitNoPop>(
            bufs.kq_rows, knT, bufs.d, bufs.d, ckl::MatmulBlockShape::of(2, 1, 1, 1, Kt, 1));
        cb_wait_front(cb_d, D_PAGES);
    }
    if constexpr (!PREP) {
        cb_wait_front(cb_v, Vb);
    }
    cb_wait_front(cb_knT, Kt);
}

// ---------------------------------------------------------------- R-BCAST q/k blocks
// Injector: qn = K^-1/2 q (*)col rsqrt(sum q^2 + 1e-6) (or kn from k) of its Kt cb_y_qk tiles -> cb_qk_stage.
ALWI void qk_part_block(Buffers& bufs, bool is_q, RowScope scope) {
    l2_inverse_norm(bufs, 0, is_q ? Q_SCALE_BITS : ONE_BITS, scope);
    cb_wait_front(cb_rinv, 1);
    cb_reserve_back(cb_qk_stage, Kt);
    // rinv in DEST slot 0, up to DEST_AUTO_LIMIT - 1 column tiles after it per window.
    constexpr uint32_t BATCH = ckl::DEST_AUTO_LIMIT - 1;
    for (uint32_t kt0 = 0; kt0 < Kt; kt0 += BATCH) {
        const uint32_t n = Kt - kt0 < BATCH ? Kt - kt0 : BATCH;
        tile_regs_acquire();
        begin_loads(cb_y_qk);
        load(cb_rinv, 0, 0);
        for (uint32_t i = 0; i < n; ++i) {
            load(cb_y_qk, kt0 + i, 1 + i);
        }
        sfpu_bcast_col_init();
        for (uint32_t i = 0; i < n; ++i) {
            mul_bcast_col_scoped(1 + i, 0, scope);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(1, cb_qk_stage, kt0);
        for (uint32_t i = 1; i < n; ++i) {
            pack_at(1 + i, cb_qk_stage, kt0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_qk_stage, Kt);
    cb_pop_front(cb_rinv, 1);
    cb_pop_front(cb_y_qk, Kt);
}

// From kn / qn as landed in cb_kq_rows: kn^T -> cb_knT and d -> cb_d.
ALWI void qk_finish_block(Buffers& bufs) {
    cb_wait_front(cb_kq_rows_f32, 2 * Kt);  // the U2D view is popped with cb_kq_rows
    kn_transpose_block(bufs);
}

// ---------------------------------------------------------------- state read / update (shared)
// [rk; ro] = [kn; qn] x S_{in} -> cb_reads (2 x Vb tiles). S_in is S0 for the pair's first step.
template <uint32_t kq_cb>
ALWI void read_state(Buffers& bufs, bool first_in_pair) {
    const uint32_t src = first_in_pair ? cb_state_in : cb_state;
    const uint32_t src_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    DataflowBuffer& src_mm_buf = first_in_pair ? bufs.state_in_mm : bufs.state_mm;
    cb_wait_front(src, n_state);
    cb_wait_front(src_mm, n_state);
    DataflowBuffer kq{kq_cb};
    ckl::matmul_block<
        false,
        false,
        ckl::LastBlockTarget::Out,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndRetainOnLastBlock,
        ckl::InputPolicy::NoWaitNoPop>(
        kq,
        src_mm_buf,
        bufs.reads,
        bufs.reads,
        ckl::MatmulBlockShape::of(2, Vb / MM_SUBBLOCK_W, 1, MM_SUBBLOCK_W, Kt, 1));
}

// S_out = decay S_in (SFPU, fp32 scalar) + kn^T x u_cb (matmul accumulated onto DEST); S_out goes to
// cb_state_out after the pair's last step, else back into cb_state. Pops S_in.
template <uint32_t knT_cb>
ALWI void update_state(bool first_in_pair, bool last_in_pair, uint32_t u_cb, uint32_t decay_bits, uint32_t rot = 0) {
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
        // To cb_state_out the tiles run rotated by `rot` (packed in run order); the writer maps them back.
        const uint32_t run0 = dst == cb_state_out ? first + rot : first;
        auto run_tile = [&](uint32_t i) {
            const uint32_t t = run0 + i;
            return STATE_STAGGER != 0 && t >= n_state ? t - n_state : t;
        };
        for (uint32_t i = 0; i < count; ++i) {
            load(src, run_tile(i), i);
        }
        binop_with_scalar_tile_init();
        for (uint32_t i = 0; i < count; ++i) {
            mul_unary_tile(i, decay_bits);
        }
        wait_sfpu_before_fpu();
        reconfig_data_format(u_cb, knT_cb);
        matmul_init(knT_cb, u_cb);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = run_tile(i);
            matmul_tiles(knT_cb, u_cb, tile / Vb, tile % Vb, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        // A batch pushed to cb_state_out moves its write pointer: the next batch packs from index 0.
        const uint32_t pack_base = dst == cb_state_out ? 0 : first;
        for (uint32_t i = 0; i < count; ++i) {
            pack_first(i, dst, pack_base + i);
        }
        tile_regs_release();
        if (dst == cb_state_out) {
            cb_push_back(dst, count);  // the writer stores each batch while the next one is computed
        }
    }
    if (dst == cb_state) {
        cb_push_back(dst, n_state);
        cb_push_back(cb_state_mm, n_state);
    }
    if (!in_place) {
        cb_pop_front(src, n_state);
        cb_pop_front(src_mm, n_state);
    }
}

// ---------------------------------------------------------------- one token step
ALWI void osum_partial(Buffers& bufs, RowScope scope, bool squared);

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
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            tile_regs_acquire();
            begin_loads(cb_reads);
            load(cb_reads, vt, 0);       // rk
            load(cb_v, vt, 1);           // v
            load(cb_reads, Vb + vt, 2);  // ro
            load(cb_tok_coef, COEF_SEL, 3);
            binop_with_scalar_tile_init();
            binop_scalar_scoped<MUL_UNARY>(0, alpha_bits, scope);
            binop_scalar_scoped<MUL_UNARY>(2, alpha_bits, scope);
            // w in the v slot: off the scope's rows u keeps v's zeros (rk there may hold other sequences' rows).
            sub_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::SUB>(1, 0, 1, scope);
            sfpu_bcast_col_init();
            mul_bcast_col_scoped(1, 3, scope);
            mul_bcast_col_scoped(2, 3, scope);
            binop_with_scalar_tile_init();
            binop_scalar_scoped<MUL_UNARY>(1, beta_bits, scope);  // u
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(1, 0);
            binop_with_scalar_tile_init();
            binop_scalar_scoped<MUL_UNARY>(0, d_bits, scope);
            add_binary_tile_init();
            binary_scoped<ckernel::BinaryOp::ADD>(2, 0, 2, scope);  // o
            if (square_o) {
                mul_binary_tile_init();
                binary_scoped<ckernel::BinaryOp::MUL>(2, 2, 0, scope);
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
    update_state<cb_knT>(first_in_pair, last_in_pair, cb_u, alpha_bits, rot);
    cb_pop_front(cb_u, Vb);
    cb_pop_front(cb_u_f32, Vb);
    cb_pop_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
}

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
ALWI void head_prep(uint32_t row_last) {
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
    pack_at(2, cb_chunk_vec, 0);  // pack indices count from the reservation
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

// ---------------------------------------------------------------- rmsnorm_gate_block
// out = ((o (*)col rsqrt(mean o^2 + eps)) (*)row w) * silu(z), with
// rsqrt(mean o^2 + eps) = sqrt(V) * rsqrt(sum o^2 + V eps); sum over the whole V (all pieces).
// o_cb holds the unit's Vb columns of o for HEADS value heads (a prep core's r heads, else one head or one piece).
// R-VSPLIT: this piece's partial sum o^2 of cb_o_acc -> cb_osum_part (the writer all-gathers it over the pieces).
// squared: the token step already pushed o^2 to cb_sq_o.
ALWI void osum_partial(Buffers& bufs, RowScope scope, bool squared) {
    if (!squared) {
        square_block<cb_o_acc, cb_sq_o>(0, Vb, scope);
    }
    DataflowBuffer osum_part{cb_osum_part};
    row_sum<cb_sq_o, cb_osum_part>(bufs, bufs.sq_o, osum_part, Vb, ckl::NoOp{});
}

// PARTIAL_DONE: the token step already produced this piece's partial sum (osum_partial); squared: it pushed o^2 to
// cb_sq_o.
template <uint32_t o_cb, uint32_t HEADS, bool PARTIAL_DONE = false>
ALWI void rmsnorm_gate_block(Buffers& bufs, uint32_t v_eps_bits, RowScope scope, bool squared = false) {
    constexpr uint32_t W = Vb / HEADS;  // columns of one head in o_cb
    const RsqrtPost rsqrt_post{v_eps_bits, SQRT_V_BITS, scope};
    if constexpr (PIECES == 1 || PREP) {
        if (!squared) {
            square_block<o_cb, cb_sq_o>(0, Vb, scope);
        }
#pragma GCC unroll 1
        for (uint32_t head = 0; head < HEADS; ++head) {
            row_sum<cb_sq_o, cb_rinv_o>(bufs, bufs.sq_o, bufs.rinv_o, W, rsqrt_post);
        }
    } else {
        if constexpr (!PARTIAL_DONE) {
            osum_partial(bufs, scope, squared);
        }
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
    cb_wait_front(cb_rinv_o, HEADS);
    cb_reserve_back(cb_out_rows, Vb);
    if constexpr (PREP) {
        // The gate factor norm_weight (*)row silu(z) was made before o landed (zw_block).
        cb_wait_front(cb_zw, Vb);
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            tile_regs_acquire();
            begin_loads(o_cb);
            load(o_cb, vt, 0);
            load(cb_zw, vt, 2);
            begin_loads(cb_rinv_o);
            load(cb_rinv_o, vt / W, 1);
            sfpu_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_first(0, cb_out_rows, vt);
            tile_regs_release();
        }
        cb_push_back(cb_out_rows, Vb);
        cb_pop_front(o_cb, Vb);
        cb_pop_front(cb_zw, Vb);
        cb_pop_front(cb_rinv_o, HEADS);
        return;
    }
    cb_wait_front(cb_z, Vb);
    for (uint32_t vt = 0; vt < Vb; ++vt) {
        tile_regs_acquire();
        begin_loads(o_cb);
        load(o_cb, vt, 0);
        begin_loads(cb_rinv_o);
        load(cb_rinv_o, vt / W, 1);
        begin_loads(cb_norm_w);
        load(cb_norm_w, vt % W, 2);
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
    cb_pop_front(o_cb, Vb);
    cb_pop_front(cb_z, Vb);
    cb_pop_front(cb_rinv_o, HEADS);
}

}  // namespace

// R-PREP prep core: zw = norm_weight (*)row silu(z) of a tile row's Vb columns -> cb_zw (fp32), while the state cores
// still run the row, so the RMS norm after o lands is two products per tile.
ALWI void zw_block() {
    constexpr uint32_t W = PREP ? Vb / RATIO : 1;  // columns of one head (norm_weight holds W tiles)
    cb_wait_front(cb_z, Vb);
    cb_reserve_back(cb_zw, Vb);
    for (uint32_t vt = 0; vt < Vb; ++vt) {
        tile_regs_acquire();
        begin_loads(cb_z);
        load(cb_z, vt, 0);
        load(cb_norm_w, vt % W, 1);
        silu_tile_init();
        silu_tile(0);
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_zw, vt);
        tile_regs_release();
    }
    cb_push_back(cb_zw, Vb);
    cb_pop_front(cb_z, Vb);
}

// R-PREP prep core: tile rows offset, offset + stride, ... of its (sequence, key head). The RMS norm of a row runs
// after the next row's prep, by when the state cores have usually returned the row's o.
ALWI void prep_main(
    Buffers& bufs, uint32_t first_unit, uint32_t T, uint32_t tr_offset, uint32_t tr_stride, uint32_t v_eps_bits) {
    const UnitCols cols{0, PREP_SHARED_QK ? 0 : 2 * Kt};
    const uint32_t conv_w_pages = (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * cols.count();
    const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    const RowScope full{RowScope::FULL_TILE};
    cb_wait_front(cb_conv_w, conv_w_pages);
    cb_wait_front(cb_gate_w, 2);
    cb_wait_front(cb_norm_w, Vb / RATIO);
    // Pass k preps row k and normalizes row k - 1 (one call site each: the math thread's text budget is tight).
    const uint32_t rows = first_tr + tr_offset <= last_tr ? (last_tr - first_tr - tr_offset) / tr_stride + 1 : 0;
#pragma GCC unroll 1
    for (uint32_t k = 0; k <= rows && rows != 0; ++k) {
        if (k < rows) {
            const uint32_t base_row = (first_tr + tr_offset + k * tr_stride) * TILE_ROWS;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
            gates_block(full);
            if constexpr (CONV_ON_FPU) {
                conv_block_fpu(full, cols, 0, cols.count());
            } else {
                conv_block(full, cols, 0, cols.count());
            }
            if constexpr (PREP_SHARED_QK) {
                kn_transpose_block(bufs);  // on kn / qn as landed by the reader
            } else {
                l2norm_block(bufs, full);
            }
            gram_block(bufs);
            cb_wait_front(cb_tri_mask, 2);  // writer-built once (mask, identity), held for the kernel
            cb_wait_front(cb_v_prep, Vb);
#pragma GCC unroll 1
            for (uint32_t head = 0; head < RATIO; ++head) {
                head_prep(hi - 1 - base_row);
                head_block(head);
            }
            cb_pop_front(cb_v_prep, Vb);
            zw_block();
            cb_pop_front(cb_gram, 2);
            cb_pop_front(cb_kq_rows, 2 * Kt);
            cb_pop_front(cb_knT, Kt);
        }
        if (k > 0) {
            rmsnorm_gate_block<cb_o_in, RATIO>(bufs, v_eps_bits, full);
        }
    }
    cb_pop_front(cb_norm_w, Vb / RATIO);
}

// R-PREP state core: the chunk step's state part per tile row on its inbox slot; o goes to the row's prep core.
ALWI void state_main(Buffers& bufs, uint32_t first_unit, uint32_t T) {
    const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    uint32_t token = 0;
    for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
        const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
        cb_reserve_back(cb_o_acc, Vb);
        cb_wait_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_wait_front(cb_in_knt, Kt);
        const uint32_t decay_bits = read_tile_value(cb_in_vec, 0, elem_index(hi - 1 - base_row, 0));  // exp(c_last)
        chunk_state_prepped(bufs, token == 0, token + (hi - lo) == T, decay_bits);
        token += hi - lo;
        cb_push_back(cb_o_acc, Vb);
        cb_pop_front(cb_in_kq, 2 * Kt);
        cb_pop_front(cb_in_knt, Kt);
        cb_pop_front(cb_in_mat, GDN_IN_MAT_PAGES);
        cb_pop_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_pop_front(cb_in_v, Vb);  // last: the reader grants a row's slots on a free cb_in_v slot
    }
}

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t v_eps_bits = get_arg_val<uint32_t>(3);  // V * epsilon
    const uint32_t qk_role = get_arg_val<uint32_t>(4);
    const uint32_t tr_offset = get_arg_val<uint32_t>(5);  // R-PREP prep core: tile rows offset, + stride, ...
    const uint32_t tr_stride = get_arg_val<uint32_t>(6);
    const uint32_t unit_stride = get_arg_val<uint32_t>(7);  // units first_unit, + stride, ... (R-QKCORE: Hv)

    // R-PREP allocates each role only its own buffers: start up on buffers the role has.
    if constexpr (PREP) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_knT, cb_gram);
    } else if constexpr (PREP_QK) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_y_qk, cb_sq, cb_qk_stage);
    } else if constexpr (STATE) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in_kq, cb_state_mm, cb_reads);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_state_mm, cb_reads);
    }
    Buffers bufs;
    if constexpr (PREP) {
        prep_main(bufs, first_unit, T, tr_offset, tr_stride, v_eps_bits);
    } else if constexpr (STATE) {
        state_main(bufs, first_unit, T);
    } else if constexpr (QK_CORE || PREP_QK) {
        // R-QKCORE: num_units = the sequences B (token rows 0..B-1), one tile row. R-PREP q / k core: whole tiles of
        // tile rows offset, offset + stride, ... of its sequence. qn or kn of the Kt columns -> cb_qk_stage (the writer
        // multicasts / writes it to the prep cores).
        const bool is_q = qk_role == GDN_QK_Q_INJECTOR;
        const UnitCols cols{is_q ? 0u : Kt, Kt};
        const RowScope scope = PREP_QK ? RowScope{RowScope::FULL_TILE} : RowScope::of(0, num_units);
        uint32_t rows = 1;
        if constexpr (PREP_QK) {
            const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
            const uint32_t first_tr = seq_lo / TILE_ROWS;
            const uint32_t last_tr = (seq_lo + T - 1) / TILE_ROWS;
            rows = first_tr + tr_offset <= last_tr ? (last_tr - first_tr - tr_offset) / tr_stride + 1 : 0;
        }
        cb_wait_front(cb_conv_w, (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * Kt);
#pragma GCC unroll 1
        for (uint32_t row = 0; row < rows; ++row) {
            if constexpr (CONV_ON_FPU) {
                conv_block_fpu(scope, cols, 0, Kt);
            } else {
                conv_block(scope, cols, 0, Kt);
            }
            qk_part_block(bufs, is_q, scope);
        }
        cb_pop_front(cb_conv_w, (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * Kt);
    } else {
        constexpr bool qk_shared = QK_SHARED;  // every core of an R-BCAST program has a shared role
        // Compile-time outside R-BCAST (every q/k column), so those builds fold the column mapping away.
        const UnitCols cols =
            !QK_SHARED ? UnitCols{0, 2 * Kt}
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
                gates_block(scope);
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
                        if (pass == 0 && cols.qk_cols != 0) {
                            qk_part_block(bufs, qk_role == GDN_QK_Q_INJECTOR, scope);
                        }
                    }
                    if (!hold_qk || unit_idx == 0) {
                        qk_finish_block(bufs);
                    }
                } else {
                    if constexpr (CONV_ON_FPU) {
                        conv_block_fpu(scope, cols, 0, cols.count());
                    } else {
                        conv_block(scope, cols, 0, cols.count());
                    }
                    l2norm_block(bufs, scope);
                }
                cb_reserve_back(cb_o_acc, Vb);
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
                if (CHUNKED || PIECES == 1) {
                    cb_push_back(cb_o_acc, Vb);  // else pushed by the token step
                }
                if (!hold_qk || last_unit) {
                    cb_pop_front(cb_kq_rows, 2 * Kt);
                    if constexpr (qk_shared) {
                        cb_pop_front(cb_kq_rows_f32, 2 * Kt);
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
                rmsnorm_gate_block<cb_o_acc, 1, !CHUNKED>(
                    bufs, v_eps_bits, scope, QK_SHARED && hi - lo == 1);  // token_step's square_o
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
}
