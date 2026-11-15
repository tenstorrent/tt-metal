// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute (regime R-STEP, op_design.md).
//
// Per pair, per tile row: gates_block -> conv_block -> l2norm_block -> recurrence_block (one
// token per step: read matmul, token vector, state update) -> rmsnorm_gate_block.
//
// Helpers: matmul_block (read [kn; qn] x S), reduce (row sums, Accurate fp32) and eltwise_chain
// (SFPU squares). Raw LLK where the design names a custom block op, with the reasons:
//   - every row / column broadcast must stay fp32 (alpha, the L2 scale, S): the only broadcast in
//     eltwise_chain is BinaryFpu (srcB, 19-bit), so these use sfpu_*_bcast_{row,col} in DEST;
//   - the state update accumulates the increment matmul onto an SFPU-decayed fp32 state already in
//     DEST; matmul_block owns DEST acquire/pack and cannot seed DEST with it.
// DEST discipline in the raw windows: every unpack-to-DEST load of a window is issued before its
// first SFPU op (an unpack-to-DEST load may run ahead of SFPU work on the math thread), and the
// math thread waits for the SFPU before the FPU accumulates onto SFPU results.

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
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
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
constexpr uint32_t n_state = Kt * Vt;
constexpr uint32_t pair_cols = 2 * Kt + Vt;
// fp32 half-sync DEST: state tiles per update batch, one slot kept for the alpha column.
constexpr uint32_t UPDATE_BATCH = (ckl::DEST_AUTO_LIMIT - 1) < n_state ? (ckl::DEST_AUTO_LIMIT - 1) : n_state;
static_assert(ckl::DEST_AUTO_LIMIT >= 4, "raw windows use 4 DEST slots");
static_assert(Vt % MM_SUBBLOCK_W == 0, "read-matmul subblock must divide Vt");

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
// alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b), on the whole gate tile.
ALWI void gates_block() {
    cb_wait_front(cb_gate_in, 2);
    tile_regs_acquire();
    begin_loads(cb_gate_in);
    load(cb_gate_in, 1, 0);  // a
    load(cb_gate_in, 0, 3);  // b
    begin_loads(cb_gate_w);
    load(cb_gate_w, 1, 1);  // dt_bias (row 0)
    load(cb_gate_w, 0, 2);  // A_log (row 0)
    sfpu_bcast_row_init();
    sfpu_add_bcast_row(0, 1);
    softplus_tile_init();
    softplus_tile(0, ONE_BITS, ONE_BITS, SOFTPLUS_THRESHOLD_BITS);
    // Pass 0: exp(A_log), then g = -exp(A_log) * softplus; pass 1: alpha = exp(g). One exp call site.
#pragma GCC unroll 1
    for (uint32_t pass = 0; pass < 2; ++pass) {
        exp_tile_init<false>();
        exp_tile<false>(pass == 0 ? 2 : 0);
        if (pass == 0) {
            sfpu_bcast_row_init();
            sfpu_mul_bcast_row(0, 2);
            negative_tile_init();
            negative_tile(0);
        }
    }
    sigmoid_tile_init<false>();
    sigmoid_tile(3);
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
    begin_loads(cb_conv_w);
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
ALWI void conv_block() {
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
        silu_tile(0);
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

// Squares of `count` tiles of an fp32 buffer starting at `base` (the input is kept) -> sq_buf.
template <uint32_t in_cb, uint32_t sq_cb>
ALWI void square_block(DataflowBuffer& in_buf, DataflowBuffer& sq_buf, uint32_t base, uint32_t count) {
    if constexpr (in_cb == cb_y_qk) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(count),
            ckl::CopyTile<ckl::input(
                cb_y_qk,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileAddressing::Offset)>{in_buf, base},
            ckl::Square<>{},
            ckl::PackTile<ckl::output(sq_cb)>{sq_buf});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(count),
            ckl::CopyTile<ckl::input(
                cb_o_acc, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::InputTileMapping::Block)>{in_buf},
            ckl::Square<>{},
            ckl::PackTile<ckl::output(sq_cb)>{sq_buf});
    }
}

// Post-reduce op on the column-0 row sum: rsqrt(sum + add) * post_mul.
struct RsqrtPost {
    uint32_t add_bits;
    uint32_t post_mul_bits;
    ALWI void operator()(uint32_t dst) const {
        binop_with_scalar_tile_init();
        add_unary_tile(dst, add_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, post_mul_bits);
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
ALWI void l2_inverse_norm(Buffers& bufs, uint32_t base, uint32_t post_mul_bits) {
    square_block<cb_y_qk, cb_sq>(bufs.y_qk, bufs.sq, base, Kt);
    row_sum<cb_sq, cb_rinv>(bufs, bufs.sq, bufs.rinv, Kt, RsqrtPost{L2_EPS_BITS, post_mul_bits});
}

// ---------------------------------------------------------------- l2norm_block
// rinv_q = K^-1/2 rsqrt(sum q^2 + 1e-6), rinv_k = rsqrt(sum k^2 + 1e-6);
// kn -> cb_kq_rows[0, Kt), qn -> cb_kq_rows[Kt, 2Kt), d = rowsum(qn * kn) -> cb_d, kn^T -> cb_knT.
ALWI void l2norm_block(Buffers& bufs) {
    // q then k: one loop body keeps a single expansion of the square chain and the reduce.
#pragma GCC unroll 1
    for (uint32_t part = 0; part < 2; ++part) {
        l2_inverse_norm(bufs, part * Kt, part == 0 ? Q_SCALE_BITS : ONE_BITS);
    }

    cb_wait_front(cb_rinv, 2);
    cb_reserve_back(cb_kq_rows, 2 * Kt);
    cb_reserve_back(cb_sq, Kt);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        tile_regs_acquire();
        begin_loads(cb_y_qk);
        load(cb_y_qk, kt, 0);       // q
        load(cb_y_qk, Kt + kt, 2);  // k
        begin_loads(cb_rinv);
        load(cb_rinv, 0, 1);
        load(cb_rinv, 1, 3);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);  // qn
        sfpu_mul_bcast_col(2, 3);  // kn
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 1);  // qn * kn
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
ALWI void token_step(Buffers& bufs, bool first_in_pair, bool last_in_pair, bool first_in_row) {
    const uint32_t src = first_in_pair ? cb_state_in : cb_state;
    const uint32_t src_mm = first_in_pair ? cb_state_in_mm : cb_state_mm;
    const uint32_t dst = last_in_pair ? cb_state_out : cb_state;
    DataflowBuffer& src_mm_buf = first_in_pair ? bufs.state_in_mm : bufs.state_mm;

    // read (5a): [rk; ro] = [kn; qn] x S_{t-1}
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

    // token vector (5b): u = beta_sel (*)col (v - alpha (*)col rk); o = alpha_sel (*)col ro + d (*)col u
    cb_wait_front(cb_tok_coef, GDN_TOK_COEF_PAGES);
    cb_wait_front(cb_reads, 2 * Vt);
    cb_reserve_back(cb_u, Vt);
    cb_reserve_back(cb_u_f32, Vt);
    for (uint32_t vt = 0; vt < Vt; ++vt) {
        tile_regs_acquire();
        begin_loads(cb_reads);
        load(cb_reads, vt, 0);
        begin_loads(cb_tok_coef);
        load(cb_tok_coef, COEF_ALPHA, 1);
        load(cb_tok_coef, COEF_BETA_SEL, 3);
        begin_loads(cb_v);
        load(cb_v, vt, 2);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(2, 3);
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
        begin_loads(cb_tok_coef);
        load(cb_tok_coef, COEF_ALPHA_SEL, 1);
        begin_loads(cb_u_f32);
        load(cb_u_f32, vt, 2);
        begin_loads(cb_d);
        load(cb_d, 0, 3);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        sfpu_mul_bcast_col(2, 3);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
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

    // update (5c): S_t = alpha S_{t-1} (SFPU, fp32) + kn^T u (matmul accumulated onto DEST)
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
        begin_loads(cb_tok_coef);
        load(cb_tok_coef, COEF_ALPHA, UPDATE_BATCH);
        begin_loads(src);
        for (uint32_t i = 0; i < count; ++i) {
            load(src, first + i, i);
        }
        sfpu_bcast_col_init();
        for (uint32_t i = 0; i < count; ++i) {
            sfpu_mul_bcast_col(i, UPDATE_BATCH);
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
ALWI void rmsnorm_gate_block(Buffers& bufs, uint32_t v_eps_bits) {
    square_block<cb_o_acc, cb_sq_o>(bufs.o_acc, bufs.sq_o, 0, Vt);
    row_sum<cb_sq_o, cb_rinv_o>(bufs, bufs.sq_o, bufs.rinv_o, Vt, RsqrtPost{v_eps_bits, SQRT_V_BITS});
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
        begin_loads(cb_z);
        load(cb_z, vt, 3);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(0, 2);
        silu_tile_init();
        silu_tile(3);
        mul_binary_tile_init();
        mul_binary_tile(0, 3, 0);
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
        cb_wait_front(cb_conv_w, GDN_CONV_TAPS * pair_cols);
        cb_wait_front(cb_gate_w, 2);
        cb_wait_front(cb_norm_w, Vt);

        uint32_t token = 0;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            gates_block();
            conv_block();
            l2norm_block(bufs);

            cb_reserve_back(cb_o_acc, Vt);
            for (uint32_t row = lo; row < hi; ++row, ++token) {
                token_step(bufs, token == 0, token == T - 1, row == lo);
            }
            cb_push_back(cb_o_acc, Vt);
            cb_pop_front(cb_kq_rows, 2 * Kt);
            cb_pop_front(cb_knT, Kt);
            cb_pop_front(cb_d, 1);
            cb_pop_front(cb_v, Vt);

            rmsnorm_gate_block(bufs, v_eps_bits);
        }
        cb_pop_front(cb_conv_w, GDN_CONV_TAPS * pair_cols);
        cb_pop_front(cb_gate_w, 2);
        cb_pop_front(cb_norm_w, Vt);
    }
}
