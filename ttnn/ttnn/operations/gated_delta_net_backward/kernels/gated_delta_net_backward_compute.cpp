// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gated_delta_net_backward — compute (TRISC).
//
// Three stages in one binary, selected by the runtime item counts:
//   P : chunk_decay -> decay_mask(L) -> ut_matrix(A) -> ut_inverse(Tinv, by
//       Neumann doubling) -> chunk_prep_products (kcd, Q, P, U) -> intra_attn
//       -> scan_seed (u, c)
//   S : state_forward_step / state_reverse_step.  The state block stays
//       resident in cb_sa / cb_sb across ALL chunk steps — it is never packed
//       out to DRAM between steps.
//   G : the six gradients, V-accumulated inside the core.
//
// ---------------------------------------------------------------------------
// HELPER-LIBRARY NOTE (deviations, with reasons)
// ---------------------------------------------------------------------------
// `ttnn/cpp/ttnn/kernel_lib/` on this branch contains 16 files and has NO
// matmul, eltwise-chain or broadcast helper (`matmul_block_helpers.hpp`,
// `eltwise_convenience.hpp`, `eltwise_chain.hpp`, `mcast_pipe.hpp` do not
// exist).  Every block matmul / elementwise / broadcast operation below is
// therefore built directly on the layer those helpers wrap
// (`api/compute/matmul.h`, `eltwise_binary.h`, `bcast.h`) as a *block*
// operation: `mmx()` walks a whole [Mt,Kd] x [Kd,Nt] block in DEST-sized
// subblocks with ONE init per block, never one tile at a time.
//
// The reduce helper (`reduce_helpers_compute.hpp`) DOES exist and is the
// design's nominated choice for the row sums.  It is deliberately not used:
//
//   1. Every reduction here is a contraction over an axis (K, V or C) whose
//      operand is already resident as a block, and every surrounding phase is
//      a matmul.  `X @ colones` (colones holds 1.0 in column 0, so the product
//      is an exact SUM) keeps the pipeline in matmul state at all seven reduce
//      sites instead of paying a reduce_init/uninit + data-format reconfig
//      round trip at each, on an op that already has ~25 phase boundaries per
//      block, and needs no scaler CB.
//   2. Four of the seven sites accumulate across the V-block loop, and the
//      helper's `Accumulate` path is single-tile only
//      (`reduce_helpers_compute.inl:224` waits/pops `onetile`) while this op's
//      row-reduce outputs are `Ct` tiles — those four would need a separate
//      add pass regardless.
//
// Sign conventions that remove constant-tile and masking work:
//   * `CST_NSTRICT` holds -1 on the strictly-lower triangle, so
//     `A = ((k_beta@k^T) . L) . NSTRICT` is the negated UT matrix in one op,
//     and `dAn = (Tinv^T d_attn Tinv^T) . NSTRICT` is -dA, which makes both of
//     dA's consumers (`W = -dA.L`, and the R term) plain positive products.
//   * `L` is built with an additive -1e4 bias on the strict upper triangle, so
//     exp() masks it to zero without a separate tril multiply AND without ever
//     evaluating exp() of a large positive argument (decay reaches -350 at the
//     strongest gate setting; exp(+350) is inf and inf*0 is NaN).
//   * `ndkcd = +dv_new @ S^T = -d_kcd`; its two consumers carry the sign.
//   * `M = Mraw . L` is used directly (never `Mt = Mraw . tril`): L is already
//     zero above the diagonal, and the decay-gradient term regroups as
//     `R = M . (q~ k^T) + W . (k_beta k^T)`, which needs no un-masked Mt.
//
// No matmul takes the same CB for both operands (a single operand id would
// have to configure both unpackers); where the algebra wants that, one side is
// copied first.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/exp.h"

#include "gdn_common.hpp"

// This kernel is code-size bound, not instruction bound: ~25 distinct block
// phases per item, each expanding to a full LLK init + op sequence on three
// TRISC binaries that share ONE kernel-config ring buffer with the reader and
// writer.  At -O3 the three binaries alone are ~105 KB against a ~70 KB budget.
// -Os on the whole translation unit is what makes the program fit; the inner
// loops are LLK macro sequences whose cost is hardware issue, not scheduling.
#pragma GCC optimize("Os")

namespace {

constexpr uint32_t NO_ACC = 0xFFFFFFFFu;

static __attribute__((noipa)) void pick_sub(uint32_t Mt, uint32_t Nt, uint32_t& rt, uint32_t& ct) {
    ct = Nt;
    if (ct > DEST_LIMIT) {
        ct = DEST_LIMIT;
    }
    while (Nt % ct) {
        --ct;
    }
    rt = DEST_LIMIT / ct;
    if (rt == 0) {
        rt = 1;
    }
    if (rt > Mt) {
        rt = Mt;
    }
    while (Mt % rt) {
        --rt;
    }
}

// Block matmul.  OUT[Mt,Nt] = A[Mt,Kd] @ (tb ? B[Nt,Kd]^T : B[Kd,Nt]).
// `acc_cb != NO_ACC` pre-loads OUT's previous value into DEST so the matmul
// accumulates on top of it (the K-blocking reload pattern).
static __attribute__((noipa)) void mmx(
    uint32_t cba,
    uint32_t cbb,
    uint32_t cbo,
    uint32_t a0,
    uint32_t b0,
    uint32_t o0,
    uint32_t Mt,
    uint32_t Kd,
    uint32_t Nt,
    uint32_t tb,
    uint32_t acc_cb,
    uint32_t acc0) {
    uint32_t rt, ct;
    if (tb) {
        // in1-transpose reads Kd CONSECUTIVE B tiles per output column, so the
        // column extent of a subblock must be 1.
        ct = 1;
        rt = DEST_LIMIT;
        if (rt > Mt) {
            rt = Mt;
        }
        while (Mt % rt) {
            --rt;
        }
    } else {
        pick_sub(Mt, Nt, rt, ct);
    }
    // matmul maps in0 -> SrcB and in1 -> SrcA, so the reconfig operands are
    // swapped relative to every other op.  This is NOT optional: a preceding
    // copy/eltwise leaves the unpacker programmed for ITS operands, and with a
    // bfloat16 input CB anywhere in the kernel the matmul would then read
    // float32 scratch through a bfloat16 tile descriptor.
    reconfig_data_format(cbb, cba);
    matmul_block_init(cba, cbb, tb, ct, rt, Kd);
    for (uint32_t c0 = 0; c0 < Nt; c0 += ct) {
        for (uint32_t r0 = 0; r0 < Mt; r0 += rt) {
            tile_regs_acquire();
            if (acc_cb != NO_ACC) {
                reconfig_data_format(acc_cb, acc_cb);
                copy_init(acc_cb);
                for (uint32_t i = 0; i < rt; ++i) {
                    for (uint32_t j = 0; j < ct; ++j) {
                        copy_tile(acc_cb, acc0 + (r0 + i) * Nt + c0 + j, i * ct + j);
                    }
                }
                reconfig_data_format(cbb, cba);
                matmul_block_init(cba, cbb, tb, ct, rt, Kd);
            }
            for (uint32_t k = 0; k < Kd; ++k) {
                const uint32_t bi = tb ? (b0 + c0 * Kd + k) : (b0 + k * Nt + c0);
                matmul_block(cba, cbb, a0 + r0 * Kd + k, bi, 0, tb, ct, rt, Kd);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < rt; ++i) {
                for (uint32_t j = 0; j < ct; ++j) {
                    pack_tile<true>(i * ct + j, cbo, o0 + (r0 + i) * Nt + c0 + j);
                }
            }
            tile_regs_release();
        }
    }
}

// ---- CB block lifecycle -------------------------------------------------
static __attribute__((noipa)) void fresh_end(uint32_t cb, uint32_t n_push, uint32_t n_front) {
    cb_push_back(cb, n_push);
    cb_wait_front(cb, n_front);
}
// In-place replacement of a single-block CB backed by ACCUM_DEPTH * n pages.
static __attribute__((noipa)) void inplace_end(uint32_t cb, uint32_t n) {
    cb_push_back(cb, n);
    cb_pop_front(cb, n);
    cb_wait_front(cb, n);
}

// ---- block producers ----------------------------------------------------
// Every phase of this kernel is "reserve a block, run one op into it, finish
// it", so that triple lives in ONE place per op kind.  `front` is the number of
// tiles to wait for after the push; `front == 0` means an in-place replacement
// (push, pop the stale block, wait for the new one), which is what the
// ACCUM_DEPTH == 2 CBs are sized for.
static __attribute__((noipa)) void blk_end(uint32_t cb, uint32_t n, uint32_t front) {
    cb_push_back(cb, n);
    if (front) {
        cb_wait_front(cb, front);
    } else {
        cb_pop_front(cb, n);
        cb_wait_front(cb, n);
    }
}

static __attribute__((noipa)) void p_mm(
    uint32_t cba,
    uint32_t cbb,
    uint32_t cbo,
    uint32_t a0,
    uint32_t b0,
    uint32_t Mt,
    uint32_t Kd,
    uint32_t Nt,
    uint32_t tb,
    uint32_t front,
    uint32_t nblk) {
    cb_reserve_back(cbo, nblk);
    mmx(cba, cbb, cbo, a0, b0, 0, Mt, Kd, Nt, tb, NO_ACC, 0);
    blk_end(cbo, nblk, front);
}
#define mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt) p_mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, 0, (Mt) * (Nt), (Mt) * (Nt))
#define mmT(cba, cbb, cbo, a0, b0, Mt, Kd, Nt) p_mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, 1, (Mt) * (Nt), (Mt) * (Nt))
#define mm_ip(cba, cbb, cbo, a0, b0, Mt, Kd, Nt) p_mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, 0, 0, (Mt) * (Nt))
#define PMM(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, front) p_mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, 0, front, (Mt) * (Nt))
#define PMMB(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, front, nblk) p_mm(cba, cbb, cbo, a0, b0, Mt, Kd, Nt, 0, front, nblk)

// ACC += A @ (B or B^T).  `first` writes fresh instead of reloading.
static __attribute__((noipa)) void mm_accum(
    uint32_t cba,
    uint32_t cbb,
    uint32_t cbacc,
    uint32_t a0,
    uint32_t b0,
    uint32_t Mt,
    uint32_t Kd,
    uint32_t Nt,
    uint32_t tb,
    bool first) {
    const uint32_t n = Mt * Nt;
    cb_reserve_back(cbacc, n);
    mmx(cba, cbb, cbacc, a0, b0, 0, Mt, Kd, Nt, tb, first ? NO_ACC : cbacc, 0);
    cb_push_back(cbacc, n);
    if (!first) {
        cb_pop_front(cbacc, n);
    }
    cb_wait_front(cbacc, n);
}

// ---- elementwise blocks -------------------------------------------------
// ONE function for all three binary ops (op: 0 = mul, 1 = add, 2 = sub).  The
// op is a runtime argument on purpose: three template instantiations cost three
// copies of the LLK init sequence, and this kernel is code-size bound.
static __attribute__((noipa)) void ew_blk(
    uint32_t op, uint32_t cba, uint32_t cbb, uint32_t cbo, uint32_t a0, uint32_t b0, uint32_t o0, uint32_t n) {
    reconfig_data_format(cba, cbb);
    if (op == 0) {
        mul_init(cba, cbb);
    } else if (op == 1) {
        add_init(cba, cbb);
    } else {
        sub_init(cba, cbb);
    }
    for (uint32_t i = 0; i < n; i += DEST_LIMIT) {
        uint32_t m = n - i;
        if (m > DEST_LIMIT) {
            m = DEST_LIMIT;
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; ++j) {
            if (op == 0) {
                mul_tiles(cba, cbb, a0 + i + j, b0 + i + j, j);
            } else if (op == 1) {
                add_tiles(cba, cbb, a0 + i + j, b0 + i + j, j);
            } else {
                sub_tiles(cba, cbb, a0 + i + j, b0 + i + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile<true>(j, cbo, o0 + i + j);
        }
        tile_regs_release();
    }
}
#define MUL_BLK(...) ew_blk(0, __VA_ARGS__)
#define ADD_BLK(...) ew_blk(1, __VA_ARGS__)
#define SUB_BLK(...) ew_blk(2, __VA_ARGS__)

// OUT[Mt,Nt] = A[Mt,Nt] * vec (column 0 broadcast across columns).
// `vstride` is 1 for a per-row vector and 0 when one tile serves every row —
// the stride-0 case replaces SCALAR broadcast everywhere in this kernel, so
// COL is the only broadcast flavour the binary instantiates.
static __attribute__((noipa)) void mul_bcol(
    uint32_t cba,
    uint32_t cbv,
    uint32_t cbo,
    uint32_t a0,
    uint32_t v0,
    uint32_t o0,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t vstride) {
    reconfig_data_format(cba, cbv);
    mul_bcast_cols_init(cba, cbv);
    for (uint32_t r = 0; r < Mt; ++r) {
        for (uint32_t c = 0; c < Nt; c += DEST_LIMIT) {
            uint32_t m = Nt - c;
            if (m > DEST_LIMIT) {
                m = DEST_LIMIT;
            }
            tile_regs_acquire();
            for (uint32_t j = 0; j < m; ++j) {
                mul_tiles_bcast<BroadcastType::COL>(cba, cbv, a0 + r * Nt + c + j, v0 + r * vstride, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile<true>(j, cbo, o0 + r * Nt + c + j);
            }
            tile_regs_release();
        }
    }
}

// ---- unary blocks -------------------------------------------------------
// One function for copy and exp(copy); `do_exp` is a runtime flag for the same
// code-size reason as ew_blk's `op`.
static __attribute__((noipa)) void copy_blk(
    uint32_t cbi, uint32_t cbo, uint32_t i0, uint32_t o0, uint32_t n, uint32_t do_exp, uint32_t istride) {
    reconfig_data_format(cbi, cbi);
    copy_init(cbi);
    if (do_exp) {
        exp_tile_init();
    }
    for (uint32_t i = 0; i < n; i += DEST_LIMIT) {
        uint32_t m = n - i;
        if (m > DEST_LIMIT) {
            m = DEST_LIMIT;
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < m; ++j) {
            copy_tile(cbi, i0 + (i + j) * istride, j);
            if (do_exp) {
                exp_tile(j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile<true>(j, cbo, o0 + i + j);
        }
        tile_regs_release();
    }
}
#define COPY_BLK(cbi, cbo, i0, o0, n) copy_blk(cbi, cbo, i0, o0, n, 0, 1)
#define EXP_BLK(cbi, cbo, i0, o0, n) copy_blk(cbi, cbo, i0, o0, n, 1, 1)
// A block of zeros: the constant zero tile replicated with a stride-0 read
// (never a CPU fill).
#define ZERO_BLK(cbo, o0, n) copy_blk(cb_const, cbo, CST_ZERO, o0, n, 0, 0)

// Transpose an [Rt, Nt] tile block into an [Nt, Rt] block.
//
// `transpose_block` is the uniform block entry point of the transpose op group
// (`api/compute/transpose.h`), so the whole DEST-sized group of tiles issues
// behind ONE init and ONE DEST handshake instead of one per tile.  The tile
// GRID permutation is the caller's: input tile `r*Nt + c` packs to output tile
// `c*Rt + r`, which is why the pack loop re-derives (r, c) from the linear
// index rather than packing consecutively.
static __attribute__((noipa)) void tr_blk(
    uint32_t cbi, uint32_t cbo, uint32_t i0, uint32_t o0, uint32_t Rt, uint32_t Nt) {
    reconfig_data_format(cbi, cbi);
    transpose_init(cbi);
    const uint32_t n = Rt * Nt;
    for (uint32_t i = 0; i < n; i += DEST_LIMIT) {
        uint32_t m = n - i;
        if (m > DEST_LIMIT) {
            m = DEST_LIMIT;
        }
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

static __attribute__((noipa)) void p_ew(
    uint32_t op, uint32_t cba, uint32_t cbb, uint32_t cbo, uint32_t a0, uint32_t b0, uint32_t n, uint32_t front) {
    cb_reserve_back(cbo, n);
    ew_blk(op, cba, cbb, cbo, a0, b0, 0, n);
    blk_end(cbo, n, front);
}
#define PMUL(...) p_ew(0, __VA_ARGS__)
#define PADD(...) p_ew(1, __VA_ARGS__)
#define PSUB(...) p_ew(2, __VA_ARGS__)

static __attribute__((noipa)) void p_mbc(
    uint32_t cba,
    uint32_t cbv,
    uint32_t cbo,
    uint32_t a0,
    uint32_t v0,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t vstride,
    uint32_t front) {
    cb_reserve_back(cbo, Mt * Nt);
    mul_bcol(cba, cbv, cbo, a0, v0, 0, Mt, Nt, vstride);
    blk_end(cbo, Mt * Nt, front);
}

static __attribute__((noipa)) void p_copy(
    uint32_t cbi,
    uint32_t cbo,
    uint32_t i0,
    uint32_t n,
    uint32_t do_exp,
    uint32_t istride,
    uint32_t front,
    uint32_t nblk) {
    cb_reserve_back(cbo, nblk);
    copy_blk(cbi, cbo, i0, 0, n, do_exp, istride);
    blk_end(cbo, nblk, front);
}
// `nblk` is the block the CB transfers; `n` is how many of its tiles this phase
// actually writes (a Gamma tile is one tile of a Ct-tile column group).
#define PCOPY(cbi, cbo, i0, n, front) p_copy(cbi, cbo, i0, n, 0, 1, front, n)
#define PEXP(cbi, cbo, i0, n, front) p_copy(cbi, cbo, i0, n, 1, 1, front, n)
#define PEXP1(cbi, cbo, i0, front, nblk) p_copy(cbi, cbo, i0, 1, 1, 1, front, nblk)
#define PZERO(cbo, n, front) p_copy(cb_const, cbo, CST_ZERO, n, 0, 0, front, n)

static __attribute__((noipa)) void p_tr(
    uint32_t cbi, uint32_t cbo, uint32_t i0, uint32_t Rt, uint32_t Nt, uint32_t front) {
    cb_reserve_back(cbo, Rt * Nt);
    tr_blk(cbi, cbo, i0, 0, Rt, Nt);
    blk_end(cbo, Rt * Nt, front);
}

// ---- egress -------------------------------------------------------------
static __attribute__((noipa)) void egress_f32(uint32_t cbi, uint32_t i0, uint32_t n) {
    cb_reserve_back(cb_egr, MAXBLK);
    COPY_BLK(cbi, cb_egr, i0, 0, n);
    cb_push_back(cb_egr, MAXBLK);
}
// The prep vector block is three separate column slots of cb_veca gathered into
// one egress block.
static __attribute__((noipa)) void egress_vec4(uint32_t cbi, uint32_t a0, uint32_t b0, uint32_t c0, uint32_t d0) {
    cb_reserve_back(cb_egr, MAXBLK);
    for (uint32_t s = 0; s < 4; ++s) {
        const uint32_t src = (s == 0) ? a0 : ((s == 1) ? b0 : ((s == 2) ? c0 : d0));
        COPY_BLK(cbi, cb_egr, src, s * Ct, Ct);
    }
    cb_push_back(cb_egr, MAXBLK);
}
static __attribute__((noipa)) void egress_grad(uint32_t cbi, uint32_t i0, uint32_t n) {
    cb_reserve_back(cb_gegr, MAXBLK_G);
    pack_reconfig_data_format(cb_gegr);
    COPY_BLK(cbi, cb_gegr, i0, 0, n);
    pack_reconfig_data_format(cb_egr);
    cb_push_back(cb_gegr, MAXBLK_G);
}

// L[t,s] = exp(decay[t] - decay[s]) for s <= t, 0 above.  cb_ca and cb_cc are
// scratch; the result lands in cb_cbL and both scratch blocks are released.
static __attribute__((noipa)) void build_L() {
    // X[t,s] = decay[t]: the outer product decay (x) row-of-ones, so the whole
    // construction stays inside matmul + transpose + eltwise and needs no
    // ROW-broadcast or unary-broadcast LLK instantiation.
    PMM(cb_veca, cb_const, cb_ca, VA_DECAY, CST_ROWONES, Ct, 1, Ct, CtCt);
    // Y = X^T, i.e. Y[t,s] = decay[s].
    p_tr(cb_ca, cb_cd, 0, Ct, Ct, CtCt);
    // D = X - Y, then L = exp(D + BIAS).
    PSUB(cb_ca, cb_cd, cb_cc, 0, 0, CtCt, CtCt);
    PADD(cb_cc, cb_const, cb_cc, 0, CST_BIAS, CtCt, 0);
    PEXP(cb_cc, cb_cbL, 0, CtCt, CtCt);

    cb_pop_front(cb_ca, CtCt);
    cb_pop_front(cb_cd, CtCt);
    cb_pop_front(cb_cc, CtCt);
}

}  // namespace

void kernel_main() {
    const uint32_t num_items = get_arg_val<uint32_t>(1);
    const uint32_t num_owned = get_arg_val<uint32_t>(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_const, cb_const, cb_egr);

    cb_wait_front(cb_const, NCONST);
    cb_wait_front(cb_colones, NCOL);

    // =====================================================================
    // Stage P — prep.  Block = one (bh, chunk) item.
    // =====================================================================
    for (uint32_t p = 0; p < num_items; ++p) {
        cb_wait_front(cb_qin, CtKt);
        cb_wait_front(cb_kin, CtKt);
        cb_wait_front(cb_gatein, 2 * Ct);

        // q~ = q * scale (also the in-dtype -> f32 conversion); k -> f32
        p_mbc(cb_qin, cb_const, cb_ka, 0, CST_SCALE, Ct, Kt, 0, CtKt);
        PCOPY(cb_kin, cb_kb, 0, CtKt, CtKt);
        cb_pop_front(cb_qin, CtKt);
        cb_pop_front(cb_kin, CtKt);

        // ---- chunk_decay_block -----------------------------------------
        // decay = LT_ones @ g ; rmg = decay[C-1] - decay = SUT_ones @ g (the
        // exclusive reverse cumsum) ; w = exp(rmg) ; dc1 = decay + rmg.
        // The cumsums read `g` straight out of the gathered CB.  Routing it
        // through an intermediate float32 copy first truncates it to the ~19-bit
        // SrcA width TWICE (measured 1.7e-3 relative error on decay, versus
        // 8.4e-4 for a single pass), and decay feeds exp(), where that error
        // compounds along the chunk -- dg is the gradient it lands in.
        // g is copied out of the gathered CB with a DATACOPY, never read from it
        // as an FPU operand.  `copy_init`/`copy_tile` select the unpack-to-DEST
        // path automatically whenever the operand's DEST format is 32-bit
        // (`eltwise_unary/*.h` -> `llk_unpack_A<..., UnpackToDestEn>`), so a
        // float32 `g` lands in DEST at full width instead of going through the
        // ~tf32-wide SrcA register an FPU operand would use.  Nothing in the
        // program descriptor configures this; it follows from the CB format.
        PCOPY(cb_gatein, cb_vecc, 0, Ct, Ct);                       // g -> f32
        PMM(cb_const, cb_vecc, cb_veca, CST_LT, 0, Ct, Ct, 1, Ct);  // decay = LT_ones @ g
        // rmg = decay[C-1] - decay = SUT_ones @ g, the EXCLUSIVE reverse cumsum.
        // w = exp(rmg) is carried to stage G in the scratch rather than
        // recomputed there as exp(dc1 - decay): dc1 and decay are both
        // ~|sum(g)| and nearly equal, so subtracting them cancels
        // catastrophically at the ~tf32 width of the FPU's source registers,
        // and exp() magnifies whatever survives.
        mm_ip(cb_const, cb_vecc, cb_vecc, CST_SUT, 0, Ct, Ct, 1);  // g -> rmg
        PEXP(cb_veca, cb_veca, VA_DECAY, Ct, 2 * Ct);              // gamma = exp(decay)
        PEXP(cb_vecc, cb_veca, 0, Ct, 3 * Ct);
        PCOPY(cb_gatein, cb_veca, Ct, Ct, 4 * Ct);                 // beta
        PADD(cb_veca, cb_vecc, cb_veca, VA_DECAY, 0, Ct, 5 * Ct);  // dc1 = decay + rmg
        cb_pop_front(cb_vecc, Ct);
        cb_pop_front(cb_gatein, 2 * Ct);

        // ---- decay_mask_block ------------------------------------------
        build_L();

        // ---- ut_matrix_block -------------------------------------------
        p_mbc(cb_kb, cb_veca, cb_kc, 0, VA_BETA, Ct, Kt, 1, CtKt);
        mmT(cb_kc, cb_kb, cb_cc, 0, 0, Ct, Kt, Ct);
        PMUL(cb_cc, cb_cbL, cb_cc, 0, 0, CtCt, 0);
        PMUL(cb_cc, cb_const, cb_cc, 0, CST_NSTRICT, CtCt, 0);  // cb_cc = A

        // ---- ut_inverse_block (Neumann doubling) -----------------------
        PCOPY(cb_cc, cb_cd, 0, CtCt, CtCt);
        PADD(cb_const, cb_cc, cb_ca, CST_EYE, 0, CtCt, CtCt);
        cb_pop_front(cb_cc, CtCt);
        for (uint32_t j = 1; j < NEUMANN_STEPS; ++j) {
            PCOPY(cb_cd, cb_ce, 0, CtCt, CtCt);
            mm_ip(cb_cd, cb_ce, cb_cd, 0, 0, Ct, Ct, Ct);  // Pw <- Pw @ Pw
            cb_pop_front(cb_ce, CtCt);
            mm(cb_ca, cb_cd, cb_ce, 0, 0, Ct, Ct, Ct);  // Z = Tinv @ Pw
            PADD(cb_ca, cb_ce, cb_ca, 0, 0, CtCt, 0);
            cb_pop_front(cb_ce, CtCt);
        }
        cb_pop_front(cb_cd, CtCt);

        // ---- chunk_prep_products_block ---------------------------------
        p_mbc(cb_kc, cb_veca, cb_kc, 0, VA_GAMMA, Ct, Kt, 1, 0);
        mm(cb_ca, cb_kc, cb_kd, 0, 0, Ct, Ct, Kt);  // kcd = Tinv @ U

        // ---- intra_attn_block ------------------------------------------
        mmT(cb_ka, cb_kb, cb_cc, 0, 0, Ct, Kt, Ct);
        PMUL(cb_cc, cb_cbL, cb_cc, 0, 0, CtCt, 0);  // cb_cc = intra

        p_mbc(cb_ka, cb_veca, cb_ka, 0, VA_GAMMA, Ct, Kt, 1, 0);
        p_mbc(cb_kb, cb_veca, cb_kb, 0, VA_W, Ct, Kt, 1, 0);

        p_tr(cb_ka, cb_ktr, 0, Ct, Kt, CtKt);
        p_tr(cb_cc, cb_ce, 0, Ct, Ct, CtCt);

        // ---- store_prep_block ------------------------------------------
        egress_f32(cb_ca, 0, CtCt);  // Tinv
        egress_f32(cb_kd, 0, CtKt);  // kcd
        egress_f32(cb_kb, 0, CtKt);  // P
        egress_vec4(cb_veca, VA_DECAY, VA_BETA, VA_DC1, VA_W);  // decay, beta, dc1, w

        // ---- scan_seed_block, per V block ------------------------------
        for (uint32_t vb = 0; vb < NVB; ++vb) {
            cb_wait_front(cb_vin, MAXV);
            cb_wait_front(cb_doin, CtVb);
            PCOPY(cb_vin, cb_vc, 0, CtVb, CtVb);
            PCOPY(cb_doin, cb_vb, 0, CtVb, CtVb);
            p_mbc(cb_vc, cb_veca, cb_vc, 0, VA_BETA, Ct, Vb, 1, 0);
            mm(cb_ca, cb_vc, cb_va, 0, 0, Ct, Ct, Vb);  // v_corr = Tinv @ v_beta
            egress_f32(cb_va, 0, CtVb);
            mm_ip(cb_ce, cb_vb, cb_vc, 0, 0, Ct, Ct, Vb);  // u = intra^T @ do
            egress_f32(cb_vc, 0, CtVb);
            mm(cb_ktr, cb_vb, cb_sa, 0, 0, Kt, Ct, Vb);  // c = Q^T @ do
            egress_f32(cb_sa, 0, KtVb);

            cb_pop_front(cb_va, CtVb);
            cb_pop_front(cb_vb, CtVb);
            cb_pop_front(cb_vc, CtVb);
            cb_pop_front(cb_sa, KtVb);
            cb_pop_front(cb_vin, MAXV);
            cb_pop_front(cb_doin, CtVb);
        }

        cb_pop_front(cb_ka, CtKt);
        cb_pop_front(cb_kb, CtKt);
        cb_pop_front(cb_kc, CtKt);
        cb_pop_front(cb_kd, CtKt);
        cb_pop_front(cb_ktr, CtKt);
        cb_pop_front(cb_ca, CtCt);
        cb_pop_front(cb_cbL, CtCt);
        cb_pop_front(cb_cc, CtCt);
        cb_pop_front(cb_ce, CtCt);
        cb_pop_front(cb_veca, VECA_BLOCKS * Ct);
    }

    // =====================================================================
    // Stage S — the sequential scan.  The state block is resident across all
    // chunk steps; only S_i / v_new / dS / dv_new leave to DRAM.
    // =====================================================================
    for (uint32_t oi = 0; oi < num_owned; ++oi) {
        for (uint32_t vb = 0; vb < NVB; ++vb) {
            if constexpr (HAS_H0) {
                cb_wait_front(cb_vin, MAXV);
                PCOPY(cb_vin, cb_sa, 0, KtVb, KtVb);
            } else {
                PZERO(cb_sa, KtVb, KtVb);
            }
            if constexpr (HAS_H0) {
                cb_pop_front(cb_vin, MAXV);
            }

            for (uint32_t i = 0; i < NC; ++i) {
                constexpr uint32_t L_KCD = 0;
                constexpr uint32_t L_P = CtKt;
                constexpr uint32_t L_VCORR = 2 * CtKt;
                constexpr uint32_t L_DC1 = 2 * CtKt + CtVb;
                cb_wait_front(cb_load_vb, LVB);
                egress_f32(cb_sa, 0, KtVb);  // S_i
                PEXP1(cb_load_vb, cb_vecc, L_DC1, Ct, Ct);  // Gamma tile
                mm(cb_load_vb, cb_sa, cb_vb, L_KCD, 0, Ct, Kt, Vb);
                PSUB(cb_load_vb, cb_vb, cb_va, L_VCORR, 0, CtVb, CtVb);  // v_new
                egress_f32(cb_va, 0, CtVb);
                p_tr(cb_load_vb, cb_ktr, L_P, Ct, Kt, CtKt);  // P^T
                mm(cb_ktr, cb_va, cb_sb, 0, 0, Kt, Ct, Vb);
                p_mbc(cb_sa, cb_vecc, cb_sa, 0, 0, Kt, Vb, 0, 0);
                PADD(cb_sa, cb_sb, cb_sa, 0, 0, KtVb, 0);

                cb_pop_front(cb_va, CtVb);
                cb_pop_front(cb_vb, CtVb);
                cb_pop_front(cb_sb, KtVb);
                cb_pop_front(cb_ktr, CtKt);
                cb_pop_front(cb_vecc, Ct);
                cb_pop_front(cb_load_vb, LVB);
            }
            cb_pop_front(cb_sa, KtVb);

            if constexpr (HAS_DHT) {
                cb_wait_front(cb_vin, MAXV);
                PCOPY(cb_vin, cb_sb, 0, KtVb, KtVb);
            } else {
                PZERO(cb_sb, KtVb, KtVb);
            }
            if constexpr (HAS_DHT) {
                cb_pop_front(cb_vin, MAXV);
            }

            for (uint32_t i = 0; i < NC; ++i) {
                constexpr uint32_t L_P = 0;
                constexpr uint32_t L_U = CtKt;
                constexpr uint32_t L_KCD = CtKt + CtVb;
                constexpr uint32_t L_C = 2 * CtKt + CtVb;
                constexpr uint32_t L_DC1 = 2 * CtKt + CtVb + KtVb;
                cb_wait_front(cb_load_vb, LVB);
                egress_f32(cb_sb, 0, KtVb);  // dS_{i+1}
                PEXP1(cb_load_vb, cb_vecc, L_DC1, Ct, Ct);  // Gamma tile
                mm(cb_load_vb, cb_sb, cb_va, L_P, 0, Ct, Kt, Vb);
                PADD(cb_load_vb, cb_va, cb_vb, L_U, 0, CtVb, CtVb);  // dv_new
                egress_f32(cb_vb, 0, CtVb);
                p_tr(cb_load_vb, cb_ktr, L_KCD, Ct, Kt, CtKt);  // kcd^T
                mm(cb_ktr, cb_vb, cb_sa, 0, 0, Kt, Ct, Vb);
                p_mbc(cb_sb, cb_vecc, cb_sb, 0, 0, Kt, Vb, 0, 0);
                PADD(cb_sb, cb_load_vb, cb_sb, 0, L_C, KtVb, 0);
                PSUB(cb_sb, cb_sa, cb_sb, 0, 0, KtVb, 0);

                cb_pop_front(cb_va, CtVb);
                cb_pop_front(cb_vb, CtVb);
                cb_pop_front(cb_sa, KtVb);
                cb_pop_front(cb_ktr, CtKt);
                cb_pop_front(cb_vecc, Ct);
                cb_pop_front(cb_load_vb, LVB);
            }

            if constexpr (HAS_H0) {
                egress_grad(cb_sb, 0, KtVb);  // dh0 = dS_0
            }
            cb_pop_front(cb_sb, KtVb);
        }
    }

    // =====================================================================
    // Stage G — gradient assembly.  V-reduced terms accumulate in L1.
    // =====================================================================
    for (uint32_t p = 0; p < num_items; ++p) {
        constexpr uint32_t I_TINV = 0;
        constexpr uint32_t I_DECAY = CtCt;
        constexpr uint32_t I_BETA = CtCt + Ct;
        constexpr uint32_t I_DC1 = CtCt + 2 * Ct;
        constexpr uint32_t I_W = CtCt + 3 * Ct;
        cb_wait_front(cb_qin, CtKt);
        cb_wait_front(cb_kin, CtKt);
        cb_wait_front(cb_load_item, LITEM);

        // ---- rebuild the decay vector block ----------------------------
        PCOPY(cb_load_item, cb_veca, I_DECAY, Ct, Ct);
        PEXP(cb_veca, cb_veca, VA_DECAY, Ct, 2 * Ct);
        PCOPY(cb_load_item, cb_veca, I_W, Ct, 3 * Ct);  // w, carried in the scratch
        PCOPY(cb_load_item, cb_veca, I_BETA, Ct, 4 * Ct);
        PCOPY(cb_load_item, cb_veca, I_DC1, Ct, 5 * Ct);

        build_L();

        p_tr(cb_load_item, cb_ce, I_TINV, Ct, Ct, CtCt);

        PCOPY(cb_kin, cb_kb, 0, CtKt, CtKt);
        p_mbc(cb_qin, cb_const, cb_ke, 0, CST_SCALE, Ct, Kt, 0, CtKt);
        cb_pop_front(cb_qin, CtKt);
        cb_pop_front(cb_kin, CtKt);
        p_mbc(cb_kb, cb_veca, cb_ka, 0, VA_BETA, Ct, Kt, 1, CtKt);
        p_mbc(cb_ka, cb_veca, cb_ka, 0, VA_GAMMA, Ct, Kt, 1, 0);

        // ---- V-block loop ----------------------------------------------
        for (uint32_t vb = 0; vb < NVB; ++vb) {
            constexpr uint32_t L_VNEW = 0;
            constexpr uint32_t L_DVNEW = CtVb;
            constexpr uint32_t L_S = 2 * CtVb;
            constexpr uint32_t L_DS = 2 * CtVb + KtVb;
            const bool first = (vb == 0);
            cb_wait_front(cb_vin, MAXV);
            cb_wait_front(cb_doin, CtVb);
            cb_wait_front(cb_load_vb, LVB);

            PCOPY(cb_vin, cb_vb, 0, CtVb, CtVb);
            PCOPY(cb_doin, cb_va, 0, CtVb, CtVb);

            // dv_beta = Tinv^T @ dv_new
            mm(cb_ce, cb_load_vb, cb_vc, 0, L_DVNEW, Ct, Ct, Vb);
            // dv = dv_beta * beta   (the only V-local output)
            cb_reserve_back(cb_gegr, MAXBLK_G);
            pack_reconfig_data_format(cb_gegr);
            mul_bcol(cb_vc, cb_veca, cb_gegr, 0, VA_BETA, 0, Ct, Vb, 1);
            pack_reconfig_data_format(cb_egr);
            cb_push_back(cb_gegr, MAXBLK_G);
            // dbeta_v += rowsum_V(dv_beta * v)
            PMUL(cb_vc, cb_vb, cb_vc, 0, 0, CtVb, 0);
            mm_accum(cb_vc, cb_colones, cb_vecc, 0, 0, Ct, Vb, 1, 0, first);
            // v_beta = v * beta
            p_mbc(cb_vb, cb_veca, cb_vb, 0, VA_BETA, Ct, Vb, 1, 0);
            // d_attn += dv_new @ v_beta^T ; Mraw += do @ v_new^T
            mm_accum(cb_load_vb, cb_vb, cb_cd, L_DVNEW, 0, Ct, Vb, Ct, 1, first);
            mm_accum(cb_va, cb_load_vb, cb_cc, 0, L_VNEW, Ct, Vb, Ct, 1, first);
            // S into its own CB so no matmul takes one CB for both operands
            PCOPY(cb_load_vb, cb_sa, L_S, KtVb, KtVb);
            mm_accum(cb_va, cb_sa, cb_kc, 0, 0, Ct, Vb, Kt, 1, first);             // dQ
            mm_accum(cb_load_vb, cb_sa, cb_kf, L_DVNEW, 0, Ct, Vb, Kt, 1, first);  // ndkcd
            // dP += v_new @ dS^T
            PCOPY(cb_load_vb, cb_vc, L_VNEW, CtVb, 0);
            mm_accum(cb_vc, cb_load_vb, cb_kd, 0, L_DS, Ct, Vb, Kt, 1, first);
            // dGamma partial: rowsum_V(dS * S) as a [Kt,1] column
            PMUL(cb_sa, cb_load_vb, cb_sa, 0, L_DS, KtVb, 0);
            mm_accum(cb_sa, cb_colones, cb_sb, 0, 0, Kt, Vb, 1, 0, first);
            cb_pop_front(cb_sa, KtVb);

            cb_pop_front(cb_va, CtVb);
            cb_pop_front(cb_vb, CtVb);
            cb_pop_front(cb_vc, CtVb);
            cb_pop_front(cb_vin, MAXV);
            cb_pop_front(cb_doin, CtVb);
            cb_pop_front(cb_load_vb, LVB);
        }

        // ---- dbeta_v out of the rolling accumulator into the column list --
        PCOPY(cb_vecc, cb_vecb, 0, Ct, Ct);  // vecb[0] = dbeta_v
        cb_pop_front(cb_vecc, Ct);

        // ---- M = Mraw . L ----------------------------------------------
        PMUL(cb_cc, cb_cbL, cb_cc, 0, 0, CtCt, 0);

        // ---- d_attn -= ndkcd @ U^T ; ndU = Tinv^T @ ndkcd --------------
        mmT(cb_kf, cb_ka, cb_ca, 0, 0, Ct, Kt, Ct);
        PSUB(cb_cd, cb_ca, cb_cd, 0, 0, CtCt, 0);
        cb_pop_front(cb_ca, CtCt);
        mm(cb_ce, cb_kf, cb_ktr, 0, 0, Ct, Ct, Kt);  // cb_ktr = ndU
        cb_pop_front(cb_kf, CtKt);

        // ---- dAn = (Tinv^T d_attn Tinv^T) . NSTRICT ; W = dAn . L ------
        mm(cb_cd, cb_ce, cb_ca, 0, 0, Ct, Ct, Ct);
        mm_ip(cb_ce, cb_ca, cb_cd, 0, 0, Ct, Ct, Ct);
        cb_pop_front(cb_ca, CtCt);
        PMUL(cb_cd, cb_const, cb_cd, 0, CST_NSTRICT, CtCt, 0);
        PMUL(cb_cd, cb_cbL, cb_cd, 0, 0, CtCt, 0);  // cb_cd = W
        cb_pop_front(cb_ce, CtCt);

        // ---- dgamma*gamma = rowsum(dQ . q~)*gamma - rowsum(ndU . U) -----
        PMUL(cb_ktr, cb_ka, cb_kf, 0, 0, CtKt, CtKt);
        PMM(cb_kf, cb_colones, cb_vecb, 0, 0, Ct, Kt, 1, 2 * Ct);  // vecb[1] = s1
        PMUL(cb_kc, cb_ke, cb_kf, 0, 0, CtKt, 0);
        PMM(cb_kf, cb_colones, cb_vecc, 0, 0, Ct, Kt, 1, Ct);   // s2
        PMUL(cb_vecc, cb_veca, cb_vecc, 0, VA_GAMMA, Ct, 0);    // s2 * gamma
        PSUB(cb_vecc, cb_vecb, cb_vecb, 0, VB_S1, Ct, 3 * Ct);  // vecb[2] = dgamma*gamma
        cb_pop_front(cb_vecc, Ct);
        cb_pop_front(cb_ka, CtKt);

        p_mbc(cb_kb, cb_veca, cb_ka, 0, VA_BETA, Ct, Kt, 1, CtKt);

        // ---- R = M . (q~ k^T) + W . (k_beta k^T) -----------------------
        mmT(cb_ke, cb_kb, cb_ca, 0, 0, Ct, Kt, Ct);
        PMUL(cb_cc, cb_ca, cb_ce, 0, 0, CtCt, CtCt);
        cb_pop_front(cb_ca, CtCt);
        mmT(cb_ka, cb_kb, cb_ca, 0, 0, Ct, Kt, Ct);
        PMUL(cb_cd, cb_ca, cb_ca, 0, 0, CtCt, 0);
        PADD(cb_ce, cb_ca, cb_ce, 0, 0, CtCt, 0);  // cb_ce = R
        cb_pop_front(cb_ca, CtCt);

        // d_decay needs rowsum(R) - rowsum(R^T).  Subtracting the two SUMS is a
        // catastrophic cancellation when the gate is strongly decaying (the two
        // sums nearly cancel and only the matmul's ~19-bit SrcA truncation
        // survives).  Subtract ELEMENTWISE first, where both operands are O(1),
        // then reduce once: same answer, one reduce instead of two.
        p_tr(cb_ce, cb_ca, 0, Ct, Ct, CtCt);
        PSUB(cb_ce, cb_ca, cb_ce, 0, 0, CtCt, 0);
        cb_pop_front(cb_ca, CtCt);
        PMM(cb_ce, cb_colones, cb_vecb, 0, 0, Ct, Ct, 1, 4 * Ct);  // vecb[3] = rowsum(R - R^T)
        cb_pop_front(cb_ce, CtCt);

        // ---- dw*w ------------------------------------------------------
        PMUL(cb_kd, cb_kb, cb_kf, 0, 0, CtKt, 0);
        PMM(cb_kf, cb_colones, cb_vecc, 0, 0, Ct, Kt, 1, Ct);
        PMUL(cb_vecc, cb_veca, cb_vecb, 0, VA_W, Ct, 5 * Ct);  // vecb[5] = dw*w
        cb_pop_front(cb_vecc, Ct);

        // ---- dq = scale * (dQ*gamma + M @ k) ---------------------------
        p_mbc(cb_kc, cb_veca, cb_kc, 0, VA_GAMMA, Ct, Kt, 1, 0);
        mm_ip(cb_cc, cb_kb, cb_kf, 0, 0, Ct, Ct, Kt);
        PADD(cb_kc, cb_kf, cb_kc, 0, 0, CtKt, 0);
        cb_reserve_back(cb_gegr, MAXBLK_G);
        pack_reconfig_data_format(cb_gegr);
        mul_bcol(cb_kc, cb_const, cb_gegr, 0, CST_SCALE, 0, Ct, Kt, 0);
        pack_reconfig_data_format(cb_egr);
        cb_push_back(cb_gegr, MAXBLK_G);

        // ---- dk_beta = W @ k - ndU * gamma -----------------------------
        mm_ip(cb_cd, cb_kb, cb_kc, 0, 0, Ct, Ct, Kt);
        p_mbc(cb_ktr, cb_veca, cb_ktr, 0, VA_GAMMA, Ct, Kt, 1, 0);
        PSUB(cb_kc, cb_ktr, cb_kc, 0, 0, CtKt, 0);  // cb_kc = dk_beta
        cb_pop_front(cb_ktr, CtKt);

        // ---- dbeta = dbeta_v + rowsum_K(dk_beta * k) -------------------
        PMUL(cb_kc, cb_kb, cb_kf, 0, 0, CtKt, 0);
        PMM(cb_kf, cb_colones, cb_vecc, 0, 0, Ct, Kt, 1, Ct);
        PADD(cb_vecb, cb_vecc, cb_vecb, VB_DBETA_V, 0, Ct, 6 * Ct);  // vecb[6] = dbeta
        cb_pop_front(cb_vecc, Ct);
        egress_grad(cb_vecb, VB_DBETA, Ct);

        // ---- dk = M^T q~ + dP*w + W^T k_beta + dk_beta*beta ------------
        p_tr(cb_cc, cb_ca, 0, Ct, Ct, CtCt);
        mm_ip(cb_ca, cb_ke, cb_kf, 0, 0, Ct, Ct, Kt);
        cb_pop_front(cb_ca, CtCt);
        p_mbc(cb_kd, cb_veca, cb_kd, 0, VA_W, Ct, Kt, 1, 0);
        PADD(cb_kf, cb_kd, cb_kf, 0, 0, CtKt, 0);
        p_tr(cb_cd, cb_ca, 0, Ct, Ct, CtCt);
        mm_ip(cb_ca, cb_ka, cb_kd, 0, 0, Ct, Ct, Kt);
        cb_pop_front(cb_ca, CtCt);
        PADD(cb_kf, cb_kd, cb_kf, 0, 0, CtKt, 0);
        p_mbc(cb_kc, cb_veca, cb_kd, 0, VA_BETA, Ct, Kt, 1, 0);
        PADD(cb_kf, cb_kd, cb_kf, 0, 0, CtKt, 0);
        egress_grad(cb_kf, 0, CtKt);

        // ---- dGamma = sum(dS . S) --------------------------------------
        // cb_sb holds the accumulated [Kt,1] column; transpose it into cb_sa
        // (a second CB, so neither block can straddle a fifo wrap) and reduce.
        p_tr(cb_sb, cb_sa, 0, Kt, 1, Kt);
        PMMB(cb_sa, cb_colones, cb_vecc, 0, 0, 1, Kt, 1, Ct, Ct);  // dGamma (element [0,0] only)
        cb_pop_front(cb_sa, Kt);
        cb_pop_front(cb_sb, Kt);
        PEXP1(cb_veca, cb_vecb, VA_DC1, 7 * Ct, Ct);  // vecb[7] = Gamma
        cb_reserve_back(cb_vecc, Ct);
        MUL_BLK(cb_vecc, cb_vecb, cb_vecc, 0, VB_GAMMA, 0, 1);
        // Only tile 0 carries the scalar; the rest of the Ct-tile column must be
        // ZERO, because the LT_ones spread below sums the whole column.
        if constexpr (Ct > 1) {
            ZERO_BLK(cb_vecc, 1, Ct - 1);
        }
        inplace_end(cb_vecc, Ct);  // dGamma * Gamma, in element [0,0]
        // Spread it down the column with LT_ones: only x[0] is non-zero, so
        // (LT @ x)[t] == x[0] for every t.
        PMM(cb_const, cb_vecc, cb_vecb, CST_LT, 0, Ct, Ct, 1, 8 * Ct);  // vecb[8]
        cb_pop_front(cb_vecc, Ct);

        // ---- dg = UT @ (rowsum(R - R^T) + dgg) - NSTRICT @ dww + dGamma*Gamma
        PADD(cb_vecb, cb_vecb, cb_vecc, VB_RMC, VB_DGG, Ct, Ct);        // e
        PMM(cb_const, cb_vecc, cb_vecb, CST_UT, 0, Ct, Ct, 1, 9 * Ct);  // vecb[8] = UT @ e
        cb_pop_front(cb_vecc, Ct);
        PMM(cb_const, cb_vecb, cb_vecc, CST_NSTRICT, VB_DWW, Ct, Ct, 1, Ct);
        PSUB(cb_vecb, cb_vecc, cb_vecc, VB_UTE, 0, Ct, 0);
        cb_reserve_back(cb_gegr, MAXBLK_G);
        pack_reconfig_data_format(cb_gegr);
        ADD_BLK(cb_vecc, cb_vecb, cb_gegr, 0, VB_DGGAM, 0, Ct);
        pack_reconfig_data_format(cb_egr);
        cb_push_back(cb_gegr, MAXBLK_G);
        cb_pop_front(cb_vecc, Ct);

        // ---- release the item ------------------------------------------
        cb_pop_front(cb_vecb, VECB_BLOCKS * Ct);
        cb_pop_front(cb_ka, CtKt);
        cb_pop_front(cb_kb, CtKt);
        cb_pop_front(cb_kc, CtKt);
        cb_pop_front(cb_kd, CtKt);
        cb_pop_front(cb_ke, CtKt);
        cb_pop_front(cb_kf, CtKt);
        cb_pop_front(cb_cbL, CtCt);
        cb_pop_front(cb_cc, CtCt);
        cb_pop_front(cb_cd, CtCt);
        cb_pop_front(cb_veca, VECA_BLOCKS * Ct);
        cb_pop_front(cb_load_item, LITEM);
    }
}
