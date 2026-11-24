// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC parametrization on one core. Per token-tile (up to 32 tokens along tile rows):
//   comb = Sinkhorn( exp( mixes @ SEL_comb + base_comb ) )   via row/col-sum matmuls RB/COL_SUM
//   pre  = sigmoid( mixes @ SEL_pre  + base_pre  ) + eps
//   post = 2 * sigmoid( mixes @ SEL_post + base_post )
// SEL_* extract+left-align each group from the 24-wide mixes and fold in the scalar a; the
// Sinkhorn normalizations are same-shape `m / (m @ K)` divides (K = RB row-sum / COL_SUM col-sum),
// so the whole op is matmul + SFPU tiles. Validated bit-for-bit against the torch reference.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t DFB_MIXES = dfb::mixes;    // mixes, streamed
constexpr uint32_t DFB_CONSTS = dfb::consts;  // 8 resident tiles
constexpr uint32_t DFB_PRE = dfb::pre;        // out
constexpr uint32_t DFB_POST = dfb::post;      // out
constexpr uint32_t DFB_COMB = dfb::comb;      // out
constexpr uint32_t DFB_MA = dfb::ma;          // m ping
constexpr uint32_t DFB_MB = dfb::mb;          // m pong
constexpr uint32_t DFB_RS = dfb::rs;          // row/col-sum broadcast
constexpr uint32_t DFB_RECIP = dfb::recip;    // reciprocal
constexpr uint32_t DFB_TMP = dfb::tmp;        // projection scratch

// const-tile indices within DFB_CONSTS; RB / COL_SUM are the row-sum / col-sum tiles
constexpr uint32_t SEL_PRE = 0, SEL_POST = 1, SEL_COMB = 2;
constexpr uint32_t BASE_PRE = 3, BASE_POST = 4, BASE_COMB = 5;
constexpr uint32_t RB = 6, COL_SUM = 7;

constexpr uint32_t TWO_BITS = 0x40000000u;  // 2.0f

// DFB flow-control (wait/reserve/push/pop) goes through DataflowBuffer objects; the
// compute LLK ops (matmul_tiles/pack_tile/add_tiles/...) still take the raw DFB id, so both
// the id and the object are kept in scope. b_tile is a tile index within DFB_CONSTS, not a DFB.
// The helpers below wrap the acquire/commit/pack/release LLK sequence every tile op repeats;
// there is no shared eltwise-helper layer to call into yet, so they are local.

// out <- a @ consts[b_tile].  Reads a[0] and consts[b_tile]; pops neither.
FORCE_INLINE void mm(uint32_t a, uint32_t b_tile, uint32_t out) {
    DataflowBuffer dfb_a(a), dfb_out(out);
    dfb_a.wait_front(1);
    reconfig_data_format(a, DFB_CONSTS);
    matmul_init(a, DFB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(a, DFB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(1);
    pack_tile(0, out);
    dfb_out.push_back(1);
    tile_regs_release();
}

// out <- a + consts[b_tile].  Pops a.
FORCE_INLINE void add_bias(uint32_t a, uint32_t b_tile, uint32_t out) {
    DataflowBuffer dfb_a(a), dfb_out(out);
    dfb_a.wait_front(1);
    reconfig_data_format(a, DFB_CONSTS);
    add_init(a, DFB_CONSTS);
    tile_regs_acquire();
    add_tiles(a, DFB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(1);
    pack_tile(0, out);
    dfb_out.push_back(1);
    tile_regs_release();
    dfb_a.pop_front(1);
}

// out <- a * b (elementwise).  Pops a and b.
FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t out) {
    DataflowBuffer dfb_a(a), dfb_b(b), dfb_out(out);
    dfb_a.wait_front(1);
    dfb_b.wait_front(1);
    reconfig_data_format(a, b);
    mul_init(a, b);
    tile_regs_acquire();
    mul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(1);
    pack_tile(0, out);
    dfb_out.push_back(1);
    tile_regs_release();
    dfb_a.pop_front(1);
    dfb_b.pop_front(1);
}

// SFPU op ids for unary()
constexpr uint32_t OP_EXP = 0, OP_SIGMOID = 1, OP_RECIP = 2, OP_ADD_EPS = 3, OP_MUL2 = 4, OP_RECIP_EPS = 5, OP_COPY = 6;

// out <- op(a).  Pops a.  scalar_bits used by the *_EPS variants.
FORCE_INLINE void unary(uint32_t a, uint32_t op, uint32_t scalar_bits, uint32_t out) {
    DataflowBuffer dfb_a(a), dfb_out(out);
    dfb_a.wait_front(1);
    reconfig_data_format_srca(a);
    copy_init(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    if (op == OP_EXP) {
        // Cap logits before exp so an out-of-range learned logit cannot overflow to inf
        // (which becomes inf/inf = NaN in the softmax divide). The softmax ratio is unchanged
        // for any realistic logit (a_res init ~ 0.01 => |logit| << 1) -- equivalent to the
        // canonical row-max subtraction within range, with no cross-lane reduce.
        unary_min_tile_init();
        unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f)
        exp_tile_init();
        exp_tile(0);
    } else if (op == OP_SIGMOID) {
        sigmoid_tile_init();
        sigmoid_tile(0);
    } else if (op == OP_RECIP) {
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_RECIP_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_ADD_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
    } else if (op == OP_MUL2) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scalar_bits);
    }  // OP_COPY: copy_tile only, no SFPU

    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(1);
    pack_tile(0, out);
    dfb_out.push_back(1);
    tile_regs_release();
    dfb_a.pop_front(1);
}

// m_out <- m_in / (m_in @ K [+ eps]).  m_in is popped (consumed by the final mul).
FORCE_INLINE void normalize(uint32_t m_in, uint32_t k_tile, bool use_eps, uint32_t eps_bits, uint32_t m_out) {
    mm(m_in, k_tile, DFB_RS);                                               // RS = m_in @ K
    unary(DFB_RS, use_eps ? OP_RECIP_EPS : OP_RECIP, eps_bits, DFB_RECIP);  // RECIP = 1/(RS[+eps])
    mul(m_in, DFB_RECIP, m_out);                                            // m_out = m_in * RECIP
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(DFB_MIXES, DFB_CONSTS, DFB_COMB);  // must precede any other compute work
    const uint32_t num_token_tiles = get_arg(args::num_token_tiles);
    constexpr uint32_t iters = get_arg(args::iters);
    constexpr uint32_t eps_bits = get_arg(args::eps_bits);

    DataflowBuffer dfb_consts(dfb::consts), dfb_mixes(dfb::mixes);
    dfb_consts.wait_front(8);  // constants resident for the whole op

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        dfb_mixes.wait_front(1);  // reused by comb/pre/post; popped at the end

        // ---- comb = Sinkhorn(exp(mixes @ SEL_comb + base_comb)) ----
        mm(DFB_MIXES, SEL_COMB, DFB_TMP);
        add_bias(DFB_TMP, BASE_COMB, DFB_MA);
        unary(DFB_MA, OP_EXP, 0, DFB_MB);                    // m = exp(logits)  -> MB
        normalize(DFB_MB, RB, false, 0, DFB_MA);             // row softmax (no eps) -> MA
        unary(DFB_MA, OP_ADD_EPS, eps_bits, DFB_MB);         // m = softmax + eps -> MB
        normalize(DFB_MB, COL_SUM, true, eps_bits, DFB_MA);  // first column norm -> MA
        for (uint32_t i = 1; i < iters; ++i) {
            normalize(DFB_MA, RB, true, eps_bits, DFB_MB);       // row -> MB
            normalize(DFB_MB, COL_SUM, true, eps_bits, DFB_MA);  // col -> MA
        }
        unary(DFB_MA, OP_COPY, 0, DFB_COMB);  // comb out = m

        // ---- pre = sigmoid(mixes @ SEL_pre + base_pre) + eps ----
        mm(DFB_MIXES, SEL_PRE, DFB_TMP);
        add_bias(DFB_TMP, BASE_PRE, DFB_MA);
        unary(DFB_MA, OP_SIGMOID, 0, DFB_MB);
        unary(DFB_MB, OP_ADD_EPS, eps_bits, DFB_PRE);

        // ---- post = 2 * sigmoid(mixes @ SEL_post + base_post) ----
        mm(DFB_MIXES, SEL_POST, DFB_TMP);
        add_bias(DFB_TMP, BASE_POST, DFB_MA);
        unary(DFB_MA, OP_SIGMOID, 0, DFB_MB);
        unary(DFB_MB, OP_MUL2, TWO_BITS, DFB_POST);

        dfb_mixes.pop_front(1);
    }
}
