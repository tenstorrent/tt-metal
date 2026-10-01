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
//  * every block MATMUL (`mm`): `matmul_block_helpers.hpp` does not exist in this tree
//    (`ttnn/cpp/ttnn/kernel_lib/` has no matmul helper).  `mm` walks a whole
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
// mm — OUT[Mt,Nt] (+)= A[Mt,Kd] @ (TB ? B[Nt,Kd]^T : B[Kd,Nt])   with an optional fused DEST
// prologue / epilogue, packed into a caller-reserved window of `out_cb` (and optionally a second).
// Every operand is addressed by tile index inside its already-fronted CB window.
//
// The two auxiliary operands x / y take a role chosen by `flags`:
//   MM_PRE   x = OUT's previous value, preloaded into DEST (the matmul accumulates onto it)
//   MM_GAM   y = one tile the preload is scaled by (SFPU, fp32 DEST) before accumulating
//   MM_MUL   x = epilogue multiplier block [Mt,Nt] (SFPU)      MM_MUL2  y = a second one
//   MM_OUT2  y = second output CB (same tile layout)
//   MM_DUAL  + x[Mt,kd2] @ y[kd2,Nt] accumulated into the same DEST (E's o = Q@h + intra@v_new)
//   MM_EXP / MM_NEG   exp / negate on the fp32 accumulation, before any MM_MUL
// Operand descriptors travel as scalar arguments on purpose: a per-call-site constant struct is
// materialized in .data, which lives in the TRISC local data memory (< 2 KB).
// ---------------------------------------------------------------------------------------------
constexpr uint32_t MM_TB = 1u << 0;
constexpr uint32_t MM_EXP = 1u << 1;
constexpr uint32_t MM_NEG = 1u << 2;
constexpr uint32_t MM_PRE = 1u << 3;
constexpr uint32_t MM_GAM = 1u << 4;
constexpr uint32_t MM_MUL = 1u << 5;
constexpr uint32_t MM_MUL2 = 1u << 6;
constexpr uint32_t MM_OUT2 = 1u << 7;
constexpr uint32_t MM_DUAL = 1u << 8;

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

// DEST[s] *= cb[base + (r0+i)*Nt + c0+j] for the n = rt*ct subblock tiles (scratch slots n..2n-1).
static __attribute__((noipa)) void mul_by_block(
    uint32_t cb, uint32_t base, uint32_t Nt, uint32_t r0, uint32_t c0, uint32_t rt, uint32_t ct) {
    const uint32_t n = rt * ct;
    copy_block_to_dest(cb, base, Nt, r0, c0, rt, ct, n);
    mul_binary_tile_init();
    for (uint32_t s = 0; s < n; ++s) {
        mul_binary_tile(s, n + s, s);
    }
}

static __attribute__((noipa)) void pack_block(
    uint32_t cb, uint32_t base, uint32_t Nt, uint32_t r0, uint32_t c0, uint32_t rt, uint32_t ct) {
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < rt; ++i) {
        for (uint32_t j = 0; j < ct; ++j) {
            pack_tile<true>(i * ct + j, cb, base + (r0 + i) * Nt + c0 + j);
        }
    }
}

static __attribute__((noipa)) void mm(
    uint32_t flags,
    uint32_t a_cb,
    uint32_t a0,
    uint32_t b_cb,
    uint32_t b0,
    uint32_t Mt,
    uint32_t Kd,
    uint32_t Nt,
    uint32_t out_cb,
    uint32_t out0,
    uint32_t x_cb = NONE,
    uint32_t x0 = 0,
    uint32_t y_cb = NONE,
    uint32_t y0 = 0,
    uint32_t kd2 = 0) {
    const uint32_t tb = (flags & MM_TB) ? 1 : 0;
    const uint32_t per_tile = (flags & (MM_MUL | MM_MUL2)) ? 2 : 1;
    const uint32_t reserve = (flags & MM_GAM) ? 1 : 0;
    const uint32_t cap = (DEST_LIMIT - reserve) / per_tile;

    // in1-transpose reads Kd consecutive B tiles per output column, and the DUAL product walks a
    // different in1 grid, so both keep one output column per subblock.
    const uint32_t ct = (tb || (flags & MM_DUAL)) ? 1 : largest_divisor_le(Nt, cap);
    const uint32_t rt = largest_divisor_le(Mt, cap / ct);
    const uint32_t n = rt * ct;
    // Any fused DEST op re-programs unpack / math, so the matmul init is re-issued per subblock;
    // a plain block matmul pays exactly one init.
    const bool reinit = (flags & (MM_PRE | MM_EXP | MM_NEG | MM_MUL | MM_MUL2 | MM_DUAL)) != 0;

    if (!reinit) {
        mm_init(a_cb, b_cb, tb, ct, rt, Kd);
    }
    for (uint32_t c0 = 0; c0 < Nt; c0 += ct) {
        for (uint32_t r0 = 0; r0 < Mt; r0 += rt) {
            tile_regs_acquire();
            if (flags & MM_PRE) {
                copy_block_to_dest(x_cb, x0, Nt, r0, c0, rt, ct, 0);
                if (flags & MM_GAM) {
                    reconfig_data_format_srca(y_cb);
                    copy_init(y_cb);
                    copy_tile(y_cb, y0, n);
                    mul_binary_tile_init();
                    for (uint32_t s = 0; s < n; ++s) {
                        mul_binary_tile(s, n, s);
                    }
                }
            }
            if (reinit) {
                mm_init(a_cb, b_cb, tb, ct, rt, Kd);
            }
            for (uint32_t k = 0; k < Kd; ++k) {
                const uint32_t bi = tb ? (b0 + c0 * Kd + k) : (b0 + k * Nt + c0);
                matmul_block(a_cb, b_cb, a0 + r0 * Kd + k, bi, 0, tb, ct, rt, Kd);
            }
            if (flags & MM_DUAL) {
                mm_init(x_cb, y_cb, 0, ct, rt, kd2);
                for (uint32_t k = 0; k < kd2; ++k) {
                    matmul_block(x_cb, y_cb, x0 + r0 * kd2 + k, y0 + k * Nt + c0, 0, 0, ct, rt, kd2);
                }
            }
            if (flags & MM_EXP) {
                exp_tile_init();
                for (uint32_t s = 0; s < n; ++s) {
                    exp_tile(s);
                }
            }
            if (flags & MM_NEG) {
                negative_tile_init();
                for (uint32_t s = 0; s < n; ++s) {
                    negative_tile(s);
                }
            }
            if (flags & MM_MUL) {
                mul_by_block(x_cb, x0, Nt, r0, c0, rt, ct);
            }
            if (flags & MM_MUL2) {
                mul_by_block(y_cb, y0, Nt, r0, c0, rt, ct);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_block(out_cb, out0, Nt, r0, c0, rt, ct);
            if (flags & MM_OUT2) {
                pack_block(y_cb, y0, Nt, r0, c0, rt, ct);
            }
            tile_regs_release();
        }
    }
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
        mm(0, cb_const, CST_LT, cb_gate_in, 0, Ct, Ct, 1, cb_out_egress, 0);
        cb_push_back(cb_out_egress, QO);

        cb_reserve_back(cb_vec, VEC_PAGES);
        mm(MM_EXP, cb_const, CST_LT, cb_gate_in, 0, Ct, Ct, 1, cb_vec, V_GAMMA);
        mm(MM_EXP, cb_const, CST_SU, cb_gate_in, 0, Ct, Ct, 1, cb_vec, V_W);
        cb_reserve_back(cb_cc_b, CtCt);  // g replicated to every column (temporary)
        mm(0, cb_gate_in, 0, cb_const, CST_EROW0, Ct, 1, 1, cb_cc_b, 0);
        publish(cb_cc_b, CtCt);
        mm(MM_EXP, cb_const, CST_ONES, cb_cc_b, 0, 1, Ct, 1, cb_vec, V_GFULL);
        cb_pop_front(cb_cc_b, CtCt);
        publish(cb_vec, VEC_PAGES);

        // ---- decay_mask_block: L = exp((LT @ diag(g)) @ SL) * LT --------------------------
        cb_reserve_back(cb_cc_a, CtCt);
        mul_col<cb_const, cb_gate_in, cb_cc_a>(CST_EYE, 0, 0, Ct, Ct);  // diag(g)
        publish(cb_cc_a, CtCt);
        cb_reserve_back(cb_cc_b, CtCt);
        mm(0, cb_const, CST_LT, cb_cc_a, 0, Ct, Ct, Ct, cb_cc_b, 0);
        publish(cb_cc_b, CtCt);
        cb_pop_front(cb_cc_a, CtCt);
        cb_reserve_back(cb_L, CtCt);
        mm(MM_EXP | MM_MUL, cb_cc_b, 0, cb_const, CST_SL, Ct, Ct, Ct, cb_L, 0, cb_const, CST_LT);
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
        mm(MM_TB | MM_MUL | MM_MUL2, cb_kb, 0, cb_k_in, 0, Ct, Kt, Ct, cb_cc_a, 0, cb_L, 0, cb_const, CST_SL);
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
            mm(0, cb_cc_a, 0, cb_cc_a, 0, Ct, Ct, Ct, cb_pow, 0);
            publish(cb_pow, CtCt);
        }
        cb_pop_front(cb_cc_a, CtCt);
        for (uint32_t j = 1; j < NEUMANN_STEPS; ++j) {
            cb_reserve_back(cb_T, CtCt);  // T <- T + T @ Pw
            mm(MM_PRE, cb_T, 0, cb_pow, 0, Ct, Ct, Ct, cb_T, 0, cb_T, 0);
            replace(cb_T, CtCt);
            if (j + 1 < NEUMANN_STEPS) {
                cb_reserve_back(cb_pow, CtCt);  // Pw <- Pw @ Pw
                mm(0, cb_pow, 0, cb_pow, 0, Ct, Ct, Ct, cb_pow, 0);
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
        mm(MM_NEG, cb_T, 0, cb_kb, 0, Ct, Ct, Kt, cb_scratch_egress, 0);
        cb_push_back(cb_scratch_egress, QF);

        cb_reserve_back(cb_scratch_egress, QF);  // Q = q~ * gamma
        mul_col<cb_qs, cb_vec, cb_scratch_egress>(0, V_GAMMA, 0, Ct, Kt);
        cb_push_back(cb_scratch_egress, QF);

        cb_reserve_back(cb_scratch_egress, QF);  // intra = (q~ @ k^T) * L
        mm(MM_TB | MM_MUL, cb_qs, 0, cb_k_in, 0, Ct, Kt, Ct, cb_scratch_egress, 0, cb_L, 0);
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
            mm(0, cb_T, 0, cb_vmat, 0, Ct, Ct, Vi, cb_scratch_egress, 0);
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
            mm(MM_PRE | MM_OUT2,
               cb_kmat_in,
               0,
               cb_state,
               0,
               Ct,
               Kt,
               Vs,
               cb_scan_vnew,
               0,
               cb_scan_vcorr,
               0,
               cb_scratch_egress,
               0);
            cb_push_back(cb_scratch_egress, QF);
            publish(cb_scan_vnew, CtVs);

            // S <- Gamma * S + P^T @ v_new   (in place)
            cb_reserve_back(cb_state, KtVs);
            mm(MM_PRE | MM_GAM, cb_scan_pt, 0, cb_scan_vnew, 0, Kt, Ct, Vs, cb_state, 0, cb_state, 0, cb_scan_gamma, 0);
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
            mm(MM_DUAL,
               cb_kmat_in,
               0,
               cb_vblock_in,
               0,
               Ct,
               Kt,
               Vi,
               cb_out_egress,
               0,
               cb_intra_in,
               0,
               cb_vnew_in,
               0,
               Ct);
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
