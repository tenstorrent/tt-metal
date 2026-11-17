// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// gated_delta_net compute (regimes R-STEP, R-VSPLIT and R-CHUNK, op_design.md): the blocks shared by the per-role
// compute kernels gated_delta_net_compute_{local,prep,state,qk}.cpp (one binary per role, ROLE CT arg).
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
//   - token_vector_band / decay_tiles: one sfpi pass per tile band (decode token vector) or per DEST batch (state
//     decay), because each eltwise call pays an SFPU start / done and an init that dominate one-band work.
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
#include "api/compute/eltwise_binary.h"
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
#include "api/debug/kernel_profiler.hpp"

// Per-block device profiler zones, compiled in only with the host knob PROFILE_ZONES (GDN_PROFILE_ZONES), on the
// unpack and pack threads: they hold the CB waits, so their zones split wait from work. A math-thread zone is
// occupancy only, and that thread's binary (the largest) would push some builds over the kernel-config text budget.
// Capture window (GDN_ZONE_SKIP > 0, env GDN_ZONE_SKIP): a RISC does not record its first GDN_ZONE_SKIP zone
// executions, so runs with rising skips cover a launch whose zones do not fit the ~250 markers of one RISC.
#if GDN_PROFILE_ZONES && GDN_ZONE_SKIP && !defined(GDN_ZONE_WIN)
#define GDN_ZONE_WIN
namespace gdn_zone {
inline uint32_t seen = 0;
template <uint32_t id>
struct win {
    bool on;
    inline __attribute__((always_inline)) win() :
        on(seen++ >= GDN_ZONE_SKIP &&
           kernel_profiler::bufferHasRoom(2 * kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE - 1)) {
        if (on) {
            kernel_profiler::stackSize += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            kernel_profiler::mark_time_at_index_inlined(kernel_profiler::wIndex, id);
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    inline __attribute__((always_inline)) ~win() {
        if (on) {
            kernel_profiler::mark_time_at_index_inlined(
                kernel_profiler::wIndex, kernel_profiler::get_const_id(id, kernel_profiler::ZONE_END));
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            kernel_profiler::stackSize -= kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
};
}  // namespace gdn_zone
#endif
#if GDN_PROFILE_ZONES && GDN_ZONE_SKIP
#define GDN_SCOPE(name)                                                        \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    gdn_zone::win<hash> zone;
#else
#define GDN_SCOPE(name) DeviceZoneScopedN(name)
#endif
#if GDN_PROFILE_ZONES && !defined(TRISC_MATH)
#define GDN_ZONE(name) GDN_SCOPE(name)
#else
#define GDN_ZONE(name)
#endif
// A point in time (an empty zone), at GDN_PROFILE_ZONES level 2.
#if GDN_PROFILE_ZONES >= 2
#define GDN_MARK(name)  \
    do {                \
        GDN_ZONE(name); \
    } while (0)
#else
#define GDN_MARK(name)
#endif

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
constexpr uint32_t cb_rinv_in = GDN_CB_RINV_IN;
constexpr uint32_t cb_chunk_vec_mm = GDN_CB_CHUNK_VEC_MM;

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
constexpr uint32_t ROLE = get_compile_time_arg_val(11);              // GDN_ROLE_*
constexpr uint32_t RATIO = get_compile_time_arg_val(12);             // R-PREP: value heads per prep group
constexpr bool PREP = ROLE == GDN_ROLE_PREP;                         // Vb is then the group's RATIO * Vt v columns
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
// Host QK_RAW_HANDOFF: a q / k core hands over its conv output y (cb_qk_stage, packed with cb_y_qk) and then its
// inverse norm (cb_rinv); the state cores scale by the inverse norms (cb_rinv_in) in the token step.
constexpr bool QK_RAW = GDN_QK_RAW != 0;
constexpr bool STAGE_Y = QK_CORE && QK_RAW;
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

// GDN_SILU_BF16 (prep role): the whole tile through the bf16 arm of the SFPU silu (see the host knob).
#ifndef GDN_SILU_BF16
#define GDN_SILU_BF16 0
#endif
ALWI void silu_tile_bf16(uint32_t dst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_silu, (false, 8), dst, VectorMode::RC));
}

// GDN_SILU_MID: x / (1 + exp(-x)) with the bf16 arm's exp, two reciprocal iterations, no rounding to bf16.
#ifndef GDN_SILU_MID
#define GDN_SILU_MID 0
#endif
#ifdef TRISC_MATH
namespace gdn_sfpu {
template <int ITERATIONS>
inline void calculate_silu_mid() {
    using namespace ckernel::sfpu;
    sfpi::vFloat c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat e = _sfpu_exp_21f_bf16_<true>(-x, sfpi::vConstFloatPrgm1, c0, c1, c2);
        sfpi::dst_reg[0] = x * sfpu_reciprocal_iter<2>(1.0f + e);
        sfpi::dst_reg++;
    }
}
}  // namespace gdn_sfpu
#endif
ALWI void silu_tile_mid(uint32_t dst) {
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_([] { gdn_sfpu::calculate_silu_mid<8>(); }, dst, VectorMode::RC);
#endif
}

ALWI void silu_conv(uint32_t dst, RowScope scope, bool qk) {
    if constexpr ((PREP || PREP_QK) && ((GDN_SILU_BF16 & 6) != 0 || GDN_SILU_MID != 0)) {
        if ((GDN_SILU_BF16 & (qk ? 4 : 2)) != 0) {
            silu_tile_bf16(dst);
        } else if (GDN_SILU_MID != 0) {
            silu_tile_mid(dst);
        } else {
            silu_scoped(dst, scope);
        }
    } else {
        silu_scoped(dst, scope);
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

// Decode token vector of one column tile in one SFPU pass over the token's band (custom block op: it replaces ~14
// scalar, binary and broadcast SFPU calls, most with their own init; 0.6 -> 0.15 us per tile at B=8). DEST in: 0 rk,
// 1 v, 2 ro, 3 row selector (1 on row p, 0 on the band's other rows, every column: the writer fills the band). Out:
// 1 u' = ik beta sel (v - ik alpha rk), 2 o = iq alpha sel ro + iq d u', 0 o^2 when SQUARE. Its multiply-adds round
// once (SFPMAD), the per-op chain's twice.
template <bool SQUARE>
ALWI void token_vector_band(
    uint32_t alpha_bits, uint32_t beta_bits, uint32_t d_bits, uint32_t ik_bits, uint32_t iq_bits, RowScope scope) {
    binop_with_scalar_tile_init();  // ADDR_MOD_7
#ifdef TRISC_MATH
    sfpu_band(0, scope, [=] {
        using sfpi::dst_reg;
        using sfpi::vFloat;
        constexpr uint32_t TS = 32;  // DEST tile size in sfpi rows
        const vFloat ik = ckernel::sfpu::Converter::as_float(ik_bits);
        const vFloat iq = ckernel::sfpu::Converter::as_float(iq_bits);
        const vFloat alpha = ckernel::sfpu::Converter::as_float(alpha_bits);
        const vFloat ak = alpha * ik;
        const vFloat aq = alpha * iq;
        const vFloat bk = ckernel::sfpu::Converter::as_float(beta_bits) * ik;
        const vFloat dq = ckernel::sfpu::Converter::as_float(d_bits) * iq;
#pragma GCC unroll 2
        for (int d = 0; d < 2; ++d) {
            const vFloat sel = dst_reg[3 * TS];
            const vFloat u = (dst_reg[TS] - dst_reg[0] * ak) * sel * bk;
            const vFloat o = dst_reg[2 * TS] * sel * aq + u * dq;
            dst_reg[TS] = u;
            dst_reg[2 * TS] = o;
            if constexpr (SQUARE) {
                dst_reg[0] = o * o;
            }
            dst_reg++;
        }
    });
#endif
}

// DEST[0 .. N) *= decay (fp32), whole tiles, in one SFPU pass (custom block op: one start / done and the N
// multiplies of a row interleaved, instead of one mul_unary_tile per tile). Needs binop_with_scalar_tile_init.
template <uint32_t N>
ALWI void decay_tiles_body(uint32_t decay_bits) {
#ifdef TRISC_MATH
    using sfpi::dst_reg;
    using sfpi::vFloat;
    constexpr uint32_t TS = 32;
    const vFloat decay = ckernel::sfpu::Converter::as_float(decay_bits);
#pragma GCC unroll 1
    for (uint32_t row = 0; row < TS; ++row) {
#pragma GCC unroll 8
        for (uint32_t t = 0; t < N; ++t) {
            dst_reg[t * TS] = dst_reg[t * TS] * decay;
        }
        dst_reg++;
    }
#endif
}

// One pass when every batch is full (n_state a multiple of the batch), so a build holds one of the two variants.
template <uint32_t N>
ALWI void decay_tiles(uint32_t count, uint32_t decay_bits) {
    if constexpr (GDN_DECAY_ONE_PASS && n_state % N == 0) {
#ifdef TRISC_MATH
        _llk_math_eltwise_sfpu_start_(0);
        decay_tiles_body<N>(decay_bits);
        _llk_math_eltwise_sfpu_done_();
#endif
    } else {
        for (uint32_t t = 0; t < count; ++t) {
            mul_unary_tile(t, decay_bits);
        }
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
    load(cb_gate_w, 1, 1);   // dt_bias (row 0)
    load(cb_gate_w, 0, 2);   // A_log (row 0)
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
        if constexpr (STAGE_Y) {
            cb_reserve_back(cb_qk_stage, cols.qk_cols);
        }
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
            if constexpr (STAGE_Y) {
                pack_at(0, cb_qk_stage, col - cols.qk_lo);
            }
        } else {
            pack_first(0, cb_v_out, col - 2 * Kt);
        }
        tile_regs_release();
        cb_pop_front(cb_conv_acc, 1);
    }
    if (qk_range) {
        cb_push_back(cb_y_qk, cols.qk_cols);
        if constexpr (STAGE_Y) {
            cb_push_back(cb_qk_stage, cols.qk_cols);
        }
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
        if constexpr (STAGE_Y) {
            cb_reserve_back(cb_qk_stage, cols.qk_cols);
        }
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
            silu_conv(i, scope, col0 + i < cols.qk_cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        // Both Float32; a core with no q / k columns (PREP_SHARE_QK prep core) allocates no cb_y_qk.
        pack_reconfig_data_format(cols.qk_cols != 0 ? cb_y_qk : cb_v_out);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t col = cols.pair_col(col0 + i);
            if (col < 2 * Kt) {
                pack_at(i, cb_y_qk, col - cols.qk_lo);
                if constexpr (STAGE_Y) {
                    pack_at(i, cb_qk_stage, col - cols.qk_lo);
                }
            } else {
                pack_at(i, cb_v_out, col - 2 * Kt);
            }
        }
        tile_regs_release();
        cb_pop_front(cb_window, GDN_CONV_TAPS * n);
    }
    if (qk_range) {
        cb_push_back(cb_y_qk, cols.qk_cols);
        if constexpr (STAGE_Y) {
            cb_push_back(cb_qk_stage, cols.qk_cols);
        }
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
    GDN_MARK("m_sq");
    row_sum<cb_sq, cb_rinv>(bufs, bufs.sq, bufs.rinv, Kt, RsqrtPost{L2_EPS_BITS, post_mul_bits, scope});
    GDN_MARK("m_rowsum");
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
// Host STATE_HOLD (R-PREP state core, a piece of two update batches): the upper batch never leaves DEST. It has one
// DEST half to itself, where the decay and the increment run in place and only the packs to L1 (the next reads'
// operand) remain: no unpack of those tiles per step. Every other window of the step takes the lower half: its commit
// does not flip (commit_stay), and its release clears the half after the packs without flipping (release_stay). A
// window there starts on operands the previous one pushed (after its release), so none runs ahead into the clear.
constexpr bool HOLD = STATE && GDN_STATE_HOLD != 0 && n_state == 2 * UPDATE_BATCH && Vb <= ckl::DEST_AUTO_LIMIT;

ALWI void commit_stay() { MATH((ckernel::math::set_math_semaphores())); }

ALWI void release_stay() {
#ifdef TRISC_PACK
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    TT_ZEROACC(ckernel::p_zeroacc::CLR_HALF, DST_ACCUM_MODE, 0, ckernel::ADDR_MOD_1, ckernel::dest_offset_id % 2);
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
#endif
}

// The held batch's release: no clear, flip.
ALWI void release_keep() { PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>())); }

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
    for (uint32_t b = 0; b < n_state; b += UPDATE_BATCH) {
        const uint32_t first = HOLD ? n_state - UPDATE_BATCH - b : b;  // HOLD: the held batch, then the lower one
        const bool held = HOLD && first != 0;
        const uint32_t count = (n_state - first) < UPDATE_BATCH ? (n_state - first) : UPDATE_BATCH;
        tile_regs_acquire();
        begin_loads(src);
        // To cb_state_out the tiles run rotated by `rot` (packed in run order); the writer maps them back.
        const uint32_t run0 = dst == cb_state_out ? first + rot : first;
        auto run_tile = [&](uint32_t i) {
            const uint32_t t = run0 + i;
            return STATE_STAGGER != 0 && t >= n_state ? t - n_state : t;
        };
        for (uint32_t i = 0; i < count && !(held && !first_in_pair); ++i) {
            load(src, run_tile(i), i);
        }
        binop_with_scalar_tile_init();
        decay_tiles<UPDATE_BATCH>(count, decay_bits);
        wait_sfpu_before_fpu();
        reconfig_data_format(u_cb, knT_cb);
        matmul_init(knT_cb, u_cb);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t tile = run_tile(i);
            matmul_tiles(knT_cb, u_cb, tile / Vb, tile % Vb, i);
        }
        if (HOLD && !held) {
            commit_stay();
        } else {
            tile_regs_commit();
        }
        tile_regs_wait();
        // A batch pushed to cb_state_out moves its write pointer: the next batch packs from index 0.
        const uint32_t pack_base = dst == cb_state_out && !HOLD ? 0 : first;
        for (uint32_t i = 0; i < count; ++i) {
            pack_first(i, dst, pack_base + i);
        }
        if (held) {
            release_keep();
        } else if (HOLD) {
            release_stay();
        } else {
            tile_regs_release();
        }
        if (dst == cb_state_out && !HOLD) {
            cb_push_back(dst, count);  // the writer stores each batch while the next one is computed
        }
    }
    if (dst == cb_state_out && HOLD) {
        cb_push_back(dst, n_state);  // the batches ran out of order
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

// ---------------------------------------------------------------- rmsnorm_gate_block
// out = ((o (*)col rsqrt(mean o^2 + eps)) (*)row w) * silu(z), with
// rsqrt(mean o^2 + eps) = sqrt(V) * rsqrt(sum o^2 + V eps); sum over the whole V (all pieces).
// o_cb holds the unit's Vb columns of o for HEADS value heads (a prep core's r heads, else one head or one piece).
// R-VSPLIT: this piece's partial sum o^2 of cb_o_acc -> cb_osum_part (the writer all-gathers it over the pieces).
// squared: the token step already pushed o^2 to cb_sq_o.
template <uint32_t o_cb = cb_o_acc>
ALWI void osum_partial(Buffers& bufs, RowScope scope, bool squared) {
    if (!squared) {
        square_block<o_cb, cb_sq_o>(0, Vb, scope);
    }
    DataflowBuffer osum_part{cb_osum_part};
    row_sum<cb_sq_o, cb_osum_part>(bufs, bufs.sq_o, osum_part, Vb, ckl::NoOp{});
}

// PARTIAL_DONE: the token step already produced this piece's partial sum (osum_partial); squared: it pushed o^2 to
// cb_sq_o.
template <uint32_t o_cb, uint32_t HEADS, bool PARTIAL_DONE = false, uint32_t N = Vb>
ALWI void rmsnorm_gate_block(Buffers& bufs, uint32_t v_eps_bits, RowScope scope, bool squared = false) {
    constexpr uint32_t W = N / HEADS;  // columns of one head in o_cb (N: its tiles, the R-PREP norm unit's)
    const RsqrtPost rsqrt_post{v_eps_bits, SQRT_V_BITS, scope};
    if constexpr (PIECES == 1 || PREP) {
        if (!squared) {
            square_block<o_cb, cb_sq_o>(0, N, scope);
        }
#pragma GCC unroll 1
        for (uint32_t head = 0; head < HEADS; ++head) {
            row_sum<cb_sq_o, cb_rinv_o>(bufs, bufs.sq_o, bufs.rinv_o, W, rsqrt_post);
        }
    } else {
        if constexpr (!PARTIAL_DONE) {
            osum_partial<o_cb>(bufs, scope, squared);
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
    cb_reserve_back(cb_out_rows, N);
    if constexpr (PREP) {
        // The gate factor norm_weight (*)row silu(z) was made before o landed (zw_block).
        cb_wait_front(cb_zw, N);
        for (uint32_t vt = 0; vt < N; ++vt) {
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
        cb_push_back(cb_out_rows, N);
        cb_pop_front(o_cb, N);
        cb_pop_front(cb_zw, N);
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
        if constexpr (STATE && (GDN_SILU_BF16 & 1) != 0) {
            silu_tile_bf16(3);  // as the prep cores' gate factor (PREP_SILU_BF16)
        } else {
            silu_scoped(3, scope);
        }
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
