// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_fwd — compute (TRISC).
//
// Three stages in one binary, selected by the runtime role counts (a core may hold all three):
//   P  gate_columns -> decay_mask (L) -> key_prep -> ut_matrix (N) -> ut_inverse (Tinv, Neumann
//      doubling) -> key_products (nkcd, Q, intra, P^T, Gamma) -> value_products (v_corr per V block)
//   S  per scan unit (bh, v_block): the state block stays resident in cb_state across ALL chunk
//      steps; per chunk emit h_i, v_new = v_corr + nkcd@S, S <- Gamma*S + P^T@v_new
//   E  per item: o = Q@h_i + intra@v_new (one DEST accumulation), v_new -> input dtype
//
// ---------------------------------------------------------------------------------------------
// HELPER-LIBRARY NOTE (deviations, with reasons)
// ---------------------------------------------------------------------------------------------
// Elementwise steps use the kernel_lib eltwise helpers (`eltwise_chain`, `copy`, `sub`,
// MulUnary / FillScalar elements).  Three block operations are built on the raw compute API:
//
//  * every block MATMUL (`mm_block`): `matmul_block_helpers.hpp` does not exist in this tree
//    (`ttnn/cpp/ttnn/kernel_lib/` has no matmul helper).  `mm_block` walks a whole
//    [Mt,Kd]x[Kd,Nt] block in DEST-sized subblocks — one init per block when no epilogue is
//    fused — on `api/compute/matmul.h`, the layer such a helper would wrap.
//  * the fused DEST epilogues of those matmuls (exp, negate, SFPU multiply by a CB block, DEST
//    preload for accumulation, the Gamma*S SFPU carry): the eltwise chain owns its own
//    tile_regs window and cannot run on a DEST that a matmul has just accumulated.  The Gamma*S
//    carry is `mul_binary_tile` (SFPU, fp32 DEST) per the design's precision contract — an FPU
//    broadcast would pass the carried state through a ~tf32 source register every chunk.
//  * the P^T materialization: the chain has no transpose element; `transpose_block` is the block
//    op (`api/compute/transpose.h`).
//
// Every CB transfers ONE uniform quantum (see l1_ledger.md); in-place updates (`X <- f(X)`) reserve
// the new block behind the still-fronted old one in an ACCUM_DEPTH = 2 CB.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_binary_sfpu.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"

#include "cgdr_common.hpp"

#pragma GCC optimize("Os")

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t NONE = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------------------------
// mm_block — OUT[Mt,Nt] (+)= A[Mt,Kd] @ (tb ? B[Nt,Kd]^T : B[Kd,Nt])  [+ A2[Mt,Kd2] @ B2[Kd2,Nt]]
// with an optional DEST prologue / epilogue, packed to one or two CBs (caller reserved).
// All block operands are addressed by tile index inside their (already fronted) CB window.
// ---------------------------------------------------------------------------------------------
struct Mm {
    uint32_t a_cb, a0, b_cb, b0, Mt, Kd, Nt;
    uint32_t out_cb, out0;
    uint32_t tb = 0;
    uint32_t a2_cb = NONE, a20 = 0, b2_cb = NONE, b20 = 0, Kd2 = 0;  // second product, same DEST
    uint32_t pre_cb = NONE, pre0 = 0;                                // preload OUT's old value
    uint32_t gam_cb = NONE, gam0 = 0;                                // ... scaled by this tile (SFPU)
    uint32_t do_exp = 0, do_neg = 0;
    uint32_t mul_cb = NONE, mul0 = 0;  // epilogue: *= mul[r, c]   (SFPU, fp32 DEST)
    uint32_t mul2_cb = NONE, mul20 = 0;
    uint32_t out2_cb = NONE, out20 = 0;  // also pack the result here
};

static __attribute__((noipa)) uint32_t largest_divisor_le(uint32_t n, uint32_t cap) {
    uint32_t d = cap < n ? cap : n;
    if (d == 0) {
        d = 1;
    }
    while (n % d) {
        --d;
    }
    return d;
}

static __attribute__((noipa)) void mm_init(
    uint32_t a_cb, uint32_t b_cb, uint32_t tb, uint32_t ct, uint32_t rt, uint32_t kd) {
    // matmul maps in0 -> SrcB and in1 -> SrcA, hence the swapped reconfig operands.
    reconfig_data_format(b_cb, a_cb);
    matmul_block_init(a_cb, b_cb, tb, ct, rt, kd);
}

static __attribute__((noipa)) void copy_block_to_dest(
    uint32_t cb, uint32_t base, uint32_t Nt, uint32_t r0, uint32_t c0, uint32_t rt, uint32_t ct, uint32_t dst0) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    for (uint32_t i = 0; i < rt; ++i) {
        for (uint32_t j = 0; j < ct; ++j) {
            copy_tile(cb, base + (r0 + i) * Nt + c0 + j, dst0 + i * ct + j);
        }
    }
}

static __attribute__((noipa)) void mul_by_block(
    uint32_t cb, uint32_t base, uint32_t Nt, uint32_t r0, uint32_t c0, uint32_t rt, uint32_t ct) {
    const uint32_t n = rt * ct;
    copy_block_to_dest(cb, base, Nt, r0, c0, rt, ct, n);
    mul_binary_tile_init();
    for (uint32_t s = 0; s < n; ++s) {
        mul_binary_tile(s, n + s, s);
    }
}

// The block operand descriptor travels as scalar arguments: a constant `Mm` temporary per call site
// would be materialized in .rodata, which lives in the TRISC local data memory (a few KB).
static __attribute__((noipa)) void mm_impl(
    uint32_t a_cb,
    uint32_t a0,
    uint32_t b_cb,
    uint32_t b0,
    uint32_t Mt,
    uint32_t Kd,
    uint32_t Nt,
    uint32_t out_cb,
    uint32_t out0,
    uint32_t tb,
    uint32_t a2_cb,
    uint32_t a20,
    uint32_t b2_cb,
    uint32_t b20,
    uint32_t Kd2,
    uint32_t pre_cb,
    uint32_t pre0,
    uint32_t gam_cb,
    uint32_t gam0,
    uint32_t do_exp,
    uint32_t do_neg,
    uint32_t mul_cb,
    uint32_t mul0,
    uint32_t mul2_cb,
    uint32_t mul20,
    uint32_t out2_cb,
    uint32_t out20) {
    const Mm m{a_cb, a0,     b_cb, b0,     Mt,   Kd,     Nt,     out_cb, out0, tb,      a2_cb, a20,     b2_cb, b20,
               Kd2,  pre_cb, pre0, gam_cb, gam0, do_exp, do_neg, mul_cb, mul0, mul2_cb, mul20, out2_cb, out20};
    const bool prologue = (m.pre_cb != NONE);
    const bool epilogue = m.do_exp || m.do_neg || (m.mul_cb != NONE);
    const uint32_t per_tile = (m.mul_cb != NONE) ? 2 : 1;
    const uint32_t reserve = (m.gam_cb != NONE) ? 1 : 0;
    const uint32_t cap = (DEST_LIMIT - reserve) / per_tile;

    uint32_t ct = m.tb ? 1 : largest_divisor_le(m.Nt, cap);
    if (m.a2_cb != NONE) {
        ct = 1;  // keep the second product's in1 walk identical to the first
    }
    const uint32_t rt = largest_divisor_le(m.Mt, cap / ct);
    const uint32_t n = rt * ct;
    const bool reinit = prologue || epilogue || (m.a2_cb != NONE);

    if (!reinit) {
        mm_init(m.a_cb, m.b_cb, m.tb, ct, rt, m.Kd);
    }
    for (uint32_t c0 = 0; c0 < m.Nt; c0 += ct) {
        for (uint32_t r0 = 0; r0 < m.Mt; r0 += rt) {
            tile_regs_acquire();
            if (prologue) {
                copy_block_to_dest(m.pre_cb, m.pre0, m.Nt, r0, c0, rt, ct, 0);
                if (m.gam_cb != NONE) {
                    reconfig_data_format_srca(m.gam_cb);
                    copy_init(m.gam_cb);
                    copy_tile(m.gam_cb, m.gam0, n);
                    mul_binary_tile_init();
                    for (uint32_t s = 0; s < n; ++s) {
                        mul_binary_tile(s, n, s);
                    }
                }
            }
            if (reinit) {
                mm_init(m.a_cb, m.b_cb, m.tb, ct, rt, m.Kd);
            }
            for (uint32_t k = 0; k < m.Kd; ++k) {
                const uint32_t bi = m.tb ? (m.b0 + c0 * m.Kd + k) : (m.b0 + k * m.Nt + c0);
                matmul_block(m.a_cb, m.b_cb, m.a0 + r0 * m.Kd + k, bi, 0, m.tb, ct, rt, m.Kd);
            }
            if (m.a2_cb != NONE) {
                mm_init(m.a2_cb, m.b2_cb, 0, ct, rt, m.Kd2);
                for (uint32_t k = 0; k < m.Kd2; ++k) {
                    matmul_block(m.a2_cb, m.b2_cb, m.a20 + r0 * m.Kd2 + k, m.b20 + k * m.Nt + c0, 0, 0, ct, rt, m.Kd2);
                }
            }
            if (m.do_exp) {
                exp_tile_init();
                for (uint32_t s = 0; s < n; ++s) {
                    exp_tile(s);
                }
            }
            if (m.do_neg) {
                negative_tile_init();
                for (uint32_t s = 0; s < n; ++s) {
                    negative_tile(s);
                }
            }
            if (m.mul_cb != NONE) {
                mul_by_block(m.mul_cb, m.mul0, m.Nt, r0, c0, rt, ct);
            }
            if (m.mul2_cb != NONE) {
                mul_by_block(m.mul2_cb, m.mul20, m.Nt, r0, c0, rt, ct);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(m.out_cb);
            for (uint32_t i = 0; i < rt; ++i) {
                for (uint32_t j = 0; j < ct; ++j) {
                    pack_tile<true>(i * ct + j, m.out_cb, m.out0 + (r0 + i) * m.Nt + c0 + j);
                }
            }
            if (m.out2_cb != NONE) {
                pack_reconfig_data_format(m.out2_cb);
                for (uint32_t i = 0; i < rt; ++i) {
                    for (uint32_t j = 0; j < ct; ++j) {
                        pack_tile<true>(i * ct + j, m.out2_cb, m.out20 + (r0 + i) * m.Nt + c0 + j);
                    }
                }
            }
            tile_regs_release();
        }
    }
}

FORCE_INLINE void mm_block(const Mm& m) {
    mm_impl(
        m.a_cb,
        m.a0,
        m.b_cb,
        m.b0,
        m.Mt,
        m.Kd,
        m.Nt,
        m.out_cb,
        m.out0,
        m.tb,
        m.a2_cb,
        m.a20,
        m.b2_cb,
        m.b20,
        m.Kd2,
        m.pre_cb,
        m.pre0,
        m.gam_cb,
        m.gam0,
        m.do_exp,
        m.do_neg,
        m.mul_cb,
        m.mul0,
        m.mul2_cb,
        m.mul20,
        m.out2_cb,
        m.out20);
}

// Transpose an [Rt, Nt] tile block (cbi from i0) into an [Nt, Rt] block (cbo from o0).
static __attribute__((noipa)) void transpose_blk(
    uint32_t cbi, uint32_t i0, uint32_t cbo, uint32_t o0, uint32_t Rt, uint32_t Nt) {
    reconfig_data_format_srca(cbi);
    transpose_init(cbi);
    pack_reconfig_data_format(cbo);
    const uint32_t n = Rt * Nt;
    for (uint32_t i = 0; i < n; i += DEST_LIMIT) {
        const uint32_t m = (n - i) < DEST_LIMIT ? (n - i) : DEST_LIMIT;
        tile_regs_acquire();
        transpose_block(cbi, i0 + i, 0, m);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t lin = i + j;
            pack_tile<true>(j, cbo, o0 + (lin % Nt) * Rt + (lin / Nt));
        }
        tile_regs_release();
    }
}

// CB block lifecycle for compute -> compute CBs.
FORCE_INLINE void publish(uint32_t cb, uint32_t n) {
    cb_push_back(cb, n);
    cb_wait_front(cb, n);
}
// In-place replacement: the new block was packed behind the fronted old one.
FORCE_INLINE void replace(uint32_t cb, uint32_t n) {
    cb_push_back(cb, n);
    cb_pop_front(cb, n);
    cb_wait_front(cb, n);
}

// ---- helper-backed elementwise block operations --------------------------------------------
// Resident operands are caller-managed (WaitPolicy::None / PopPolicy::None) and addressed by
// tile offset; outputs are packed at an explicit offset into a caller-reserved window.
constexpr InputSpec resident(uint32_t cb, InputTileMapping m = InputTileMapping::Block) {
    return input(cb, WaitPolicy::None, PopPolicy::None, m, DataFormatReconfig::Enabled, TileAddressing::Offset);
}
constexpr OutputSpec into(uint32_t cb) {
    return output(cb, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Offset);
}

// OUT[Mt,Nt] = A[Mt,Nt] * col(V)   (V: one tile per row of tiles, column 0 broadcast)
template <uint32_t cb_a, uint32_t cb_v, uint32_t cb_o>
FORCE_INLINE void mul_col(uint32_t a0, uint32_t v0, uint32_t o0, uint32_t Mt, uint32_t Nt) {
    eltwise_chain(
        IterationShape::grid(Mt, Nt),
        BinaryFpu<BinaryFpuOp::Mul, resident(cb_a), input(resident(cb_v, InputTileMapping::Col), BroadcastDim::Col)>{
            a0, v0},
        PackTile<into(cb_o)>{o0});
}

// In place: X[Mt,Nt] <- X * col(V), streamed through the CB (X is its own output).
template <uint32_t cb_x, uint32_t cb_v>
FORCE_INLINE void mul_col_inplace(uint32_t v0, uint32_t Mt, uint32_t Nt) {
    eltwise_chain(
        IterationShape::grid(Mt, Nt),
        BinaryFpu<BinaryFpuOp::Mul, input(cb_x), input(resident(cb_v, InputTileMapping::Col), BroadcastDim::Col)>{
            0, v0},
        PackTile<output(cb_x)>{});
}

template <uint32_t cb_i, uint32_t cb_o>
FORCE_INLINE void copy_blk(uint32_t i0, uint32_t o0, uint32_t n) {
    eltwise_chain(IterationShape::tiles(n), CopyTile<resident(cb_i)>{i0}, PackTile<into(cb_o)>{o0});
}

}  // namespace

void kernel_main() {
    const uint32_t num_items = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t scale_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_q_in, cb_const, cb_scratch_egress);

    cb_wait_front(cb_const, NCONST);

    // =========================================================================================
    // Stage P — one (bh, chunk) item per iteration.
    // =========================================================================================
    for (uint32_t r = 0; r < num_items; ++r) {
        cb_wait_front(cb_gate_in, 2 * Ct);
        cb_wait_front(cb_q_in, CtKt);
        cb_wait_front(cb_k_in, CtKt);

        // ---- gate_columns_block ---------------------------------------------------------
        // decay = LT @ g -> g_cumsum ; gamma = exp(LT @ g) ; w = exp(SU @ g) ;
        // Gamma_full = exp(ONES @ (g @ E_ROW0)) (every element).  exp() runs on the fp32 DEST
        // accumulation, so decay never passes through an FPU source register on its way to exp.
        cb_reserve_back(cb_out_egress, QO);
        mm_block(
            {.a_cb = cb_const,
             .a0 = CST_LT,
             .b_cb = cb_gate_in,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = 1,
             .out_cb = cb_out_egress,
             .out0 = 0});
        cb_push_back(cb_out_egress, QO);

        cb_reserve_back(cb_vec, VEC_PAGES);
        mm_block(
            {.a_cb = cb_const,
             .a0 = CST_LT,
             .b_cb = cb_gate_in,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = 1,
             .out_cb = cb_vec,
             .out0 = V_GAMMA,
             .do_exp = 1});
        mm_block(
            {.a_cb = cb_const,
             .a0 = CST_SU,
             .b_cb = cb_gate_in,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = 1,
             .out_cb = cb_vec,
             .out0 = V_W,
             .do_exp = 1});
        cb_reserve_back(cb_cc_b, CtCt);  // g replicated to every column (temporary)
        mm_block(
            {.a_cb = cb_gate_in,
             .a0 = 0,
             .b_cb = cb_const,
             .b0 = CST_EROW0,
             .Mt = Ct,
             .Kd = 1,
             .Nt = 1,
             .out_cb = cb_cc_b,
             .out0 = 0});
        publish(cb_cc_b, CtCt);
        mm_block(
            {.a_cb = cb_const,
             .a0 = CST_ONES,
             .b_cb = cb_cc_b,
             .b0 = 0,
             .Mt = 1,
             .Kd = Ct,
             .Nt = 1,
             .out_cb = cb_vec,
             .out0 = V_GFULL,
             .do_exp = 1});
        cb_pop_front(cb_cc_b, CtCt);
        publish(cb_vec, VEC_PAGES);

        // ---- decay_mask_block: L = exp((LT @ diag(g)) @ SL) * LT --------------------------
        cb_reserve_back(cb_cc_a, CtCt);
        mul_col<cb_const, cb_gate_in, cb_cc_a>(CST_EYE, 0, 0, Ct, Ct);  // diag(g)
        publish(cb_cc_a, CtCt);
        cb_reserve_back(cb_cc_b, CtCt);
        mm_block(
            {.a_cb = cb_const,
             .a0 = CST_LT,
             .b_cb = cb_cc_a,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = Ct,
             .out_cb = cb_cc_b,
             .out0 = 0});
        publish(cb_cc_b, CtCt);
        cb_pop_front(cb_cc_a, CtCt);
        cb_reserve_back(cb_L, CtCt);
        mm_block(
            {.a_cb = cb_cc_b,
             .a0 = 0,
             .b_cb = cb_const,
             .b0 = CST_SL,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = Ct,
             .out_cb = cb_L,
             .out0 = 0,
             .do_exp = 1,
             .mul_cb = cb_const,
             .mul0 = CST_LT});
        publish(cb_L, CtCt);
        cb_pop_front(cb_cc_b, CtCt);

        // ---- key_prep_block: q~ = scale * q ; k_beta = k * beta ----------------------------
        cb_reserve_back(cb_qs, CtKt);
        eltwise_chain(
            IterationShape::tiles(CtKt),
            CopyTile<resident(cb_q_in)>{0},
            MulUnary<>{scale_bits},
            PackTile<into(cb_qs)>{0});
        publish(cb_qs, CtKt);
        cb_pop_front(cb_q_in, CtKt);
        cb_reserve_back(cb_kb, CtKt);
        mul_col<cb_k_in, cb_gate_in, cb_kb>(0, Ct, 0, Ct, Kt);
        publish(cb_kb, CtKt);

        // ---- ut_matrix_block: N = (k_beta @ k^T) * L * SL ----------------------------------
        cb_reserve_back(cb_cc_a, CtCt);
        mm_block(
            {.a_cb = cb_kb,
             .a0 = 0,
             .b_cb = cb_k_in,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Kt,
             .Nt = Ct,
             .out_cb = cb_cc_a,
             .out0 = 0,
             .tb = 1,
             .mul_cb = cb_L,
             .mul0 = 0,
             .mul2_cb = cb_const,
             .mul20 = CST_SL});
        publish(cb_cc_a, CtCt);

        // ---- ut_inverse_block: Tinv = (I - N) prod_{j>=1} (I + N^(2^j)) --------------------
        cb_reserve_back(cb_T, CtCt);
        eltwise_chain(
            IterationShape::tiles(CtCt),
            BinaryFpu<BinaryFpuOp::Sub, resident(cb_const), resident(cb_cc_a)>{CST_EYE, 0},
            PackTile<into(cb_T)>{0});
        publish(cb_T, CtCt);
        if constexpr (NEUMANN_STEPS > 1) {
            cb_reserve_back(cb_pow, CtCt);
            mm_block(
                {.a_cb = cb_cc_a,
                 .a0 = 0,
                 .b_cb = cb_cc_a,
                 .b0 = 0,
                 .Mt = Ct,
                 .Kd = Ct,
                 .Nt = Ct,
                 .out_cb = cb_pow,
                 .out0 = 0});
            publish(cb_pow, CtCt);
        }
        cb_pop_front(cb_cc_a, CtCt);
        for (uint32_t j = 1; j < NEUMANN_STEPS; ++j) {
            cb_reserve_back(cb_T, CtCt);  // T <- T + T @ Pw
            mm_block(
                {.a_cb = cb_T,
                 .a0 = 0,
                 .b_cb = cb_pow,
                 .b0 = 0,
                 .Mt = Ct,
                 .Kd = Ct,
                 .Nt = Ct,
                 .out_cb = cb_T,
                 .out0 = 0,
                 .pre_cb = cb_T,
                 .pre0 = 0});
            replace(cb_T, CtCt);
            if (j + 1 < NEUMANN_STEPS) {
                cb_reserve_back(cb_pow, CtCt);  // Pw <- Pw @ Pw
                mm_block(
                    {.a_cb = cb_pow,
                     .a0 = 0,
                     .b_cb = cb_pow,
                     .b0 = 0,
                     .Mt = Ct,
                     .Kd = Ct,
                     .Nt = Ct,
                     .out_cb = cb_pow,
                     .out0 = 0});
                replace(cb_pow, CtCt);
            }
        }
        if constexpr (NEUMANN_STEPS > 1) {
            cb_pop_front(cb_pow, CtCt);
        }
        cb_reserve_back(cb_out_egress, QO);  // Tinv -> A
        copy_blk<cb_T, cb_out_egress>(0, 0, CtCt);
        cb_push_back(cb_out_egress, QO);

        // ---- key_products_block ----------------------------------------------------------
        mul_col_inplace<cb_kb, cb_vec>(V_GAMMA, Ct, Kt);  // U = k_beta * gamma
        cb_wait_front(cb_kb, CtKt);

        cb_reserve_back(cb_scratch_egress, QF);  // nkcd = -(Tinv @ U)
        mm_block(
            {.a_cb = cb_T,
             .a0 = 0,
             .b_cb = cb_kb,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Ct,
             .Nt = Kt,
             .out_cb = cb_scratch_egress,
             .out0 = 0,
             .do_neg = 1});
        cb_push_back(cb_scratch_egress, QF);

        cb_reserve_back(cb_scratch_egress, QF);  // Q = q~ * gamma
        mul_col<cb_qs, cb_vec, cb_scratch_egress>(0, V_GAMMA, 0, Ct, Kt);
        cb_push_back(cb_scratch_egress, QF);

        cb_reserve_back(cb_scratch_egress, QF);  // intra = (q~ @ k^T) * L
        mm_block(
            {.a_cb = cb_qs,
             .a0 = 0,
             .b_cb = cb_k_in,
             .b0 = 0,
             .Mt = Ct,
             .Kd = Kt,
             .Nt = Ct,
             .out_cb = cb_scratch_egress,
             .out0 = 0,
             .tb = 1,
             .mul_cb = cb_L,
             .mul0 = 0});
        cb_push_back(cb_scratch_egress, QF);

        cb_reserve_back(cb_kw, CtKt);  // P^T = (k * w)^T
        mul_col<cb_k_in, cb_vec, cb_kw>(0, V_W, 0, Ct, Kt);
        publish(cb_kw, CtKt);
        cb_reserve_back(cb_scratch_egress, QF);
        transpose_blk(cb_kw, 0, cb_scratch_egress, 0, Ct, Kt);
        cb_push_back(cb_scratch_egress, QF);
        cb_pop_front(cb_kw, CtKt);

        cb_reserve_back(cb_scratch_egress, QF);  // Gamma_full
        copy_blk<cb_vec, cb_scratch_egress>(V_GFULL, 0, 1);
        cb_push_back(cb_scratch_egress, QF);

        cb_pop_front(cb_k_in, CtKt);
        cb_pop_front(cb_qs, CtKt);
        cb_pop_front(cb_kb, CtKt);

        // ---- value_products_block, per V block: v_corr = Tinv @ (v * beta) ---------------
        for (uint32_t vb = 0; vb < NVI; ++vb) {
            cb_wait_front(cb_vblock_in, QV);
            cb_reserve_back(cb_vmat, CtVi);
            mul_col<cb_vblock_in, cb_gate_in, cb_vmat>(0, Ct, 0, Ct, Vi);
            publish(cb_vmat, CtVi);
            cb_pop_front(cb_vblock_in, QV);
            cb_reserve_back(cb_scratch_egress, QF);
            mm_block(
                {.a_cb = cb_T,
                 .a0 = 0,
                 .b_cb = cb_vmat,
                 .b0 = 0,
                 .Mt = Ct,
                 .Kd = Ct,
                 .Nt = Vi,
                 .out_cb = cb_scratch_egress,
                 .out0 = 0});
            cb_push_back(cb_scratch_egress, QF);
            cb_pop_front(cb_vmat, CtVi);
        }

        cb_pop_front(cb_T, CtCt);
        cb_pop_front(cb_L, CtCt);
        cb_pop_front(cb_vec, VEC_PAGES);
        cb_pop_front(cb_gate_in, 2 * Ct);
    }

    // =========================================================================================
    // Stage S — scan units.  cb_state is resident across every chunk step.
    // =========================================================================================
    for (uint32_t uu = 0; uu < num_units; ++uu) {
        cb_reserve_back(cb_state, KtVs);
        if constexpr (HAS_H0) {
            cb_wait_front(cb_vblock_in, QV);
            copy_blk<cb_vblock_in, cb_state>(0, 0, KtVs);
            cb_pop_front(cb_vblock_in, QV);
        } else {
            eltwise_chain(IterationShape::tiles(KtVs), FillScalar<Dst::D0>{0.0f}, PackTile<into(cb_state)>{0});
        }
        publish(cb_state, KtVs);

        for (uint32_t i = 0; i < NC; ++i) {
            cb_wait_front(cb_kmat_in, CtKt);
            cb_wait_front(cb_scan_pt, CtKt);
            cb_wait_front(cb_scan_vcorr, CtVs);
            cb_wait_front(cb_scan_gamma, 1);

            if (!(HAS_H0 && i == 0)) {  // h_i = S (the state ENTERING chunk i)
                cb_reserve_back(cb_out_egress, QO);
                copy_blk<cb_state, cb_out_egress>(0, 0, KtVs);
                cb_push_back(cb_out_egress, QO);
            }

            // v_new = v_corr + nkcd @ S  -> cb_scan_vnew (next matmul) and scratch (stage E)
            cb_reserve_back(cb_scan_vnew, CtVs);
            cb_reserve_back(cb_scratch_egress, QF);
            mm_block(
                {.a_cb = cb_kmat_in,
                 .a0 = 0,
                 .b_cb = cb_state,
                 .b0 = 0,
                 .Mt = Ct,
                 .Kd = Kt,
                 .Nt = Vs,
                 .out_cb = cb_scan_vnew,
                 .out0 = 0,
                 .pre_cb = cb_scan_vcorr,
                 .pre0 = 0,
                 .out2_cb = cb_scratch_egress,
                 .out20 = 0});
            cb_push_back(cb_scratch_egress, QF);
            publish(cb_scan_vnew, CtVs);

            // S <- Gamma * S + P^T @ v_new   (in place)
            cb_reserve_back(cb_state, KtVs);
            mm_block(
                {.a_cb = cb_scan_pt,
                 .a0 = 0,
                 .b_cb = cb_scan_vnew,
                 .b0 = 0,
                 .Mt = Kt,
                 .Kd = Ct,
                 .Nt = Vs,
                 .out_cb = cb_state,
                 .out0 = 0,
                 .pre_cb = cb_state,
                 .pre0 = 0,
                 .gam_cb = cb_scan_gamma,
                 .gam0 = 0});
            replace(cb_state, KtVs);

            cb_pop_front(cb_scan_vnew, CtVs);
            cb_pop_front(cb_kmat_in, CtKt);
            cb_pop_front(cb_scan_pt, CtKt);
            cb_pop_front(cb_scan_vcorr, CtVs);
            cb_pop_front(cb_scan_gamma, 1);
        }

        cb_reserve_back(cb_out_egress, QO);  // final_state
        copy_blk<cb_state, cb_out_egress>(0, 0, KtVs);
        cb_push_back(cb_out_egress, QO);
        cb_pop_front(cb_state, KtVs);
    }

    // =========================================================================================
    // Stage E — o = Q @ h_i + intra @ v_new, per V block.
    // =========================================================================================
    for (uint32_t r = 0; r < num_items; ++r) {
        cb_wait_front(cb_kmat_in, CtKt);
        cb_wait_front(cb_intra_in, CtCt);
        for (uint32_t vb = 0; vb < NVI; ++vb) {
            cb_wait_front(cb_vblock_in, QV);
            cb_wait_front(cb_vnew_in, CtVi);
            cb_reserve_back(cb_out_egress, QO);
            mm_block(
                {.a_cb = cb_kmat_in,
                 .a0 = 0,
                 .b_cb = cb_vblock_in,
                 .b0 = 0,
                 .Mt = Ct,
                 .Kd = Kt,
                 .Nt = Vi,
                 .out_cb = cb_out_egress,
                 .out0 = 0,
                 .a2_cb = cb_intra_in,
                 .a20 = 0,
                 .b2_cb = cb_vnew_in,
                 .b20 = 0,
                 .Kd2 = Ct});
            cb_push_back(cb_out_egress, QO);
            cb_reserve_back(cb_out_egress, QO);
            copy_blk<cb_vnew_in, cb_out_egress>(0, 0, CtVi);
            cb_push_back(cb_out_egress, QO);
            cb_pop_front(cb_vblock_in, QV);
            cb_pop_front(cb_vnew_in, CtVi);
        }
        cb_pop_front(cb_kmat_in, CtKt);
        cb_pop_front(cb_intra_in, CtCt);
    }
}
