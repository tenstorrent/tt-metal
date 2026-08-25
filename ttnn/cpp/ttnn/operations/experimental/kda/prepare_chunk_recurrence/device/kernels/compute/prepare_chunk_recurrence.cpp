// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Chunk-parallel KDA prep. For cumulative vector gate G [C,K], factor
// exp(G_i-G_j) inside each q/k dot as exp(G_i)*exp(-G_j):
//   qd=q*exp(G), kl=k*exp(G), kr=k*exp(-G)
//   Akk=strictly_lower((beta*kl)@kr^T), Aqk=tril(qd@kr^T)
//   kd=beta*kl, k_dec_t=(kr*exp(G_last))^T, dl=exp(G_last).
// The existing blocked WY inverse helpers below are reused unchanged.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

inline void WAIT(DataflowBuffer& buffer, uint32_t n) { buffer.wait_front(n); }
inline void POP(DataflowBuffer& buffer, uint32_t n) { buffer.pop_front(n); }

constexpr uint32_t largest_divisor_at_most(uint32_t value, uint32_t limit) {
    for (uint32_t divisor = limit; divisor > 1; --divisor) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr>
inline void mm(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    constexpr uint32_t subblock_columns = largest_divisor_at_most(Nt, dst_tiles);
    constexpr uint32_t subblock_rows = largest_divisor_at_most(Mt, dst_tiles / subblock_columns);
    static_assert(subblock_rows * subblock_columns <= dst_tiles);
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
    pack_reconfig_data_format(o_id);  // mixed bf16/fp32 CBs: set packer to this output's format
    // Matmul maps in0=a_id->srcB and in1=b_id->srcA. Reconfigure unpack formats explicitly because init only
    // asserts formats; otherwise mixed fp32/bf16 CBs are read in the wrong format.
    reconfig_data_format(b_id, a_id);
    matmul_block_init(a_id, b_id, Tr, subblock_columns, subblock_rows, Kt);
    for (uint32_t mi = 0; mi < Mt; mi += subblock_rows) {
        for (uint32_t ni = 0; ni < Nt; ni += subblock_columns) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                // kt_dim describes operand geometry; each K slice is still issued explicitly. B^T is stored
                // [Nt,Kt], while the non-transposed B is stored [Kt,Nt].
                const uint32_t b_index = Tr ? ni * Kt + ki : ki * Nt + ni;
                matmul_block(a_id, b_id, mi * Kt + ki, b_index, 0, Tr, subblock_columns, subblock_rows, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < subblock_rows; row++) {
                for (uint32_t column = 0; column < subblock_columns; column++) {
                    pack_tile(row * subblock_columns + column, o_id, (mi + row) * Nt + ni + column);
                }
            }
            tile_regs_release();
        }
    }
    o.push_back(Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
inline void ew(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& o, uint32_t n, int op) {
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format(a_id, b_id);  // binary(a_id,b_id): a_id->srcA, b_id->srcB
    if (op == 0) {
        add_init(a_id, b_id);
    } else if (op == 1) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a_id, b_id, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a_id, b_id, i, i, 0);
        } else {
            mul_tiles(a_id, b_id, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(n);
}

// Square through SFPU destination-register multiply; avoids occupying the matrix FPU used by prep.
inline void square_sfpu(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in_id, i, 0);
        copy_tile(in_id, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(n);
}

inline void expc(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);  // unary: in_id->srcA
    copy_tile_to_dst_init_short(in_id);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in_id, i, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(n);
}

inline void halfc(DataflowBuffer& in, DataflowBuffer& o, uint32_t n) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in_id, i, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3f000000);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void bcast_cols_mul(DataflowBuffer& a, DataflowBuffer& col, DataflowBuffer& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t a_id = a.get_id();
    const uint32_t col_id = col.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Mt * Nt);
    pack_reconfig_data_format(o_id);
    reconfig_data_format(a_id, col_id);  // bcast(a_id,col_id): a_id->srcA, col_id->srcB
    mul_bcast_cols_init(a_id, col_id);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a_id, col_id, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o_id, mi * Nt + ni);
            tile_regs_release();
        }
    }
    o.push_back(Mt * Nt);
}

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
inline void cpy_t(DataflowBuffer& src, uint32_t src_tile, DataflowBuffer& o) {
    const uint32_t src_id = src.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(1);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(src_id);
    copy_tile_to_dst_init_short(src_id);
    tile_regs_acquire();
    copy_tile(src_id, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o_id, 0);
    tile_regs_release();
    o.push_back(1);
}

// Invert (I-N) for a strictly-lower 32x32 N using the exact nilpotent product
// (I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16). Eight tile matmuls replace
// the masked 16x16 Horner path's thirty full-tile matmuls; the shorter dependency chain is also
// expected to improve fp32 stability, but that must be validated empirically.
inline void invert_doubling(
    DataflowBuffer& src,
    uint32_t tile,
    DataflowBuffer& out,
    DataflowBuffer& state,
    DataflowBuffer& final_state,
    DataflowBuffer& state_two,
    DataflowBuffer& state_three,
    DataflowBuffer& eye) {
    DataflowBuffer* power = &state;
    DataflowBuffer* sum = &final_state;
    DataflowBuffer* next_power = &state_two;
    DataflowBuffer* product = &state_three;

    cpy_t(src, tile, *power);
    power->wait_front(1);
    ew(eye, *power, *sum, 1, 0);
    sum->wait_front(1);
    mm<1, 1, 1, false>(*power, *power, *next_power);
    next_power->wait_front(1);
    power->pop_front(1);
    power = next_power;
    next_power = &state;

    for (uint32_t step = 0; step < 4; ++step) {
        mm<1, 1, 1, false>(*power, *sum, *product);
        product->wait_front(1);
        if (step < 3) {
            mm<1, 1, 1, false>(*power, *power, *next_power);
            next_power->wait_front(1);
        }
        power->pop_front(1);
        ew(*sum, *product, *power, 1, 0);
        power->wait_front(1);
        sum->pop_front(1);
        product->pop_front(1);
        if (step < 3) {
            DataflowBuffer* old_sum = sum;
            sum = power;
            power = next_power;
            next_power = old_sum;
        } else {
            sum = power;
        }
    }
    cpy_t(*sum, 0, out);
    out.wait_front(1);
    sum->pop_front(1);
}

// out[1,Ct] row-form = transpose of col[Ct,1]; produces Ct tiles (each row0 = a 32-chunk of col).
inline void transpose_col(DataflowBuffer& in, DataflowBuffer& o, uint32_t Ct) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(Ct);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);  // unary: in_id->srcA
    transpose_init(in_id);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        transpose_tile(in_id, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(Ct);
}

// Exact fp32 row sum for q/k L2 normalization. The shared SFPU reducer avoids four full-tile
// matrix multiplies and preserves the input for the caller-managed scratch lifetime.
inline void rowsum_k(uint32_t Mt, uint32_t Kt) {
    compute_kernel_lib::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        dfb::scratch_one,
        dfb::ones,
        dfb::scratch_two,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(Mt, Kt));
}

// inv_rms: o[i] = rsqrt(in[i] + eps) [* scale]. in holds per-row sum-of-squares (rowsum_k output);
// out is the per-row inverse-L2 factor (optionally pre-scaled, for folding q's scale into the norm).
// eps/scale arrive as fp32-bit-cast uint32 compile args.
inline void inv_rms(
    DataflowBuffer& in, DataflowBuffer& o, uint32_t n, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    const uint32_t in_id = in.get_id();
    const uint32_t o_id = o.get_id();

    o.reserve_back(n);
    pack_reconfig_data_format(o_id);
    reconfig_data_format_srca(in_id);
    copy_tile_to_dst_init_short(in_id);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in_id, i, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);  // + eps
        rsqrt_tile_init();
        rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
        if (do_scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale_bits);  // * scale (q only)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_id, i);
        tile_regs_release();
    }
    o.push_back(n);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t SCALE_BITS, uint32_t EPS_BITS>
TT_KERNEL void compute(uint32_t work_count) {
    static_assert(Ct == 1, "chunk KDA currently requires chunk_size=32");

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer state(dfb::state);
    DataflowBuffer decay(dfb::decay);
    DataflowBuffer decay_exp(dfb::decay_exp);
    DataflowBuffer decay_factor(dfb::decay_factor);
    DataflowBuffer lower_mask(dfb::lower_mask);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer k_beta(dfb::k_beta);
    DataflowBuffer w(dfb::w);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer state_two(dfb::state_two);
    DataflowBuffer v_new(dfb::v_new);
    DataflowBuffer output_intermediate(dfb::output_intermediate);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch_one(dfb::scratch_one);
    DataflowBuffer scratch_two(dfb::scratch_two);
    DataflowBuffer scratch_three(dfb::scratch_three);
    DataflowBuffer state_three(dfb::state_three);

    compute_kernel_hw_startup(dfb::q, dfb::k, dfb::scratch_one);
    WAIT(eye, cc);
    WAIT(tril, cc);
    WAIT(ones, cc);

    for (uint32_t c = 0; c < work_count; c++) {
        WAIT(q, ck);
        WAIT(k, ck);
        WAIT(v, cv);
        WAIT(g, ck);
        WAIT(beta, Ct);

        square_sfpu(q, scratch_one, ck);
        rowsum_k(Ct, Kt);
        WAIT(scratch_two, Ct);
        inv_rms(scratch_two, scratch_three, Ct, EPS_BITS, SCALE_BITS, true);
        WAIT(scratch_three, Ct);
        POP(scratch_two, Ct);
        bcast_cols_mul(q, scratch_three, state_temporary, Ct, Kt);
        WAIT(state_temporary, ck);
        POP(scratch_three, Ct);
        POP(q, ck);

        square_sfpu(k, scratch_one, ck);
        rowsum_k(Ct, Kt);
        WAIT(scratch_two, Ct);
        inv_rms(scratch_two, scratch_three, Ct, EPS_BITS, SCALE_BITS, false);
        WAIT(scratch_three, Ct);
        POP(scratch_two, Ct);
        bcast_cols_mul(k, scratch_three, final_state, Ct, Kt);
        WAIT(final_state, ck);
        POP(scratch_three, Ct);
        POP(k, ck);

        // v_beta and k_beta.
        bcast_cols_mul(v, beta, v_beta, Ct, Vt);
        WAIT(v_beta, cv);
        POP(v, cv);
        bcast_cols_mul(final_state, beta, k_beta, Ct, Kt);
        WAIT(k_beta, ck);
        POP(beta, Ct);

        // G = cumsum(g). Anchor the separable pairwise factors at G_last/2 so neither
        // exp(G-anchor) nor exp(anchor-G) spans the full chunk range. Their products are
        // unchanged, while realistic KDA gates no longer overflow exp(-G).
        mm<Ct, Ct, Kt, false>(tril, g, decay);
        WAIT(decay, ck);
        expc(decay, decay_exp, ck);  // exp(G), for scan-facing q_decay/kd
        WAIT(decay_exp, ck);

        mm<Ct, Ct, Kt, false>(ones, g, scratch_one);  // replicated G_last
        WAIT(scratch_one, ck);
        POP(g, ck);
        halfc(scratch_one, scratch_two, ck);  // anchor = G_last/2
        WAIT(scratch_two, ck);
        ew(decay, scratch_two, output_intermediate, ck, 1);  // G-anchor
        WAIT(output_intermediate, ck);
        POP(decay, ck);
        expc(output_intermediate, decay, ck);  // exp(G-anchor); cumsum has already released this full-size CB.
        WAIT(decay, ck);

        decay_factor.reserve_back(ck);
        pack_reconfig_data_format(decay_factor.get_id());
        reconfig_data_format_srca(output_intermediate.get_id());
        copy_tile_to_dst_init_short(output_intermediate.get_id());
        for (uint32_t i = 0; i < ck; i++) {
            tile_regs_acquire();
            copy_tile(output_intermediate.get_id(), i, 0);
            negative_tile_init();
            negative_tile(0);
            exp_tile_init();
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, decay_factor.get_id(), i);
            tile_regs_release();
        }
        decay_factor.push_back(ck);  // exp(anchor-G)
        WAIT(decay_factor, ck);
        POP(output_intermediate, ck);

        expc(scratch_two, state_update, ck);  // exp(G_last/2), also exp(G_last-anchor)
        WAIT(state_update, ck);

        // Preserve exact scan-facing factors, and use anchored factors only for pairwise products.
        ew(state_temporary, decay_exp, q_decay, ck, 2);
        WAIT(q_decay, ck);
        ew(state_temporary, decay, state_three, ck, 2);
        WAIT(state_three, ck);  // q*exp(G-anchor)
        POP(state_temporary, ck);
        ew(k_beta, decay_exp, w, ck, 2);
        WAIT(w, ck);
        ew(k_beta, decay, state_two, ck, 2);
        WAIT(state_two, ck);  // beta*k*exp(G-anchor)
        POP(k_beta, ck);
        POP(decay, ck);
        POP(decay_exp, ck);
        expc(scratch_one, decay, ck);  // exp(G_last), for state decay dl
        WAIT(decay, ck);
        POP(scratch_one, ck);
        POP(scratch_two, ck);
        ew(final_state, decay_factor, scratch_one, ck, 2);
        WAIT(scratch_one, ck);  // k*exp(anchor-G)
        POP(final_state, ck);
        POP(decay_factor, ck);

        // Materialize both anchored pairwise products, then release state_two/state_three before
        // the doubling inverse reuses those CBs as private scratch. Only the masked Aqk is published to
        // writer-facing intra; publishing the raw matrix creates a second consumer race.
        mm<Ct, Kt, Ct, true>(state_two, scratch_one, lower_mask);
        WAIT(lower_mask, cc);  // raw beta*k_i*k_j*exp(G_i-G_j)
        POP(state_two, ck);
        mm<Ct, Kt, Ct, true>(state_three, scratch_one, scratch_two);
        WAIT(scratch_two, cc);  // raw q_i*k_j*exp(G_i-G_j)
        POP(state_three, ck);
        ew(scratch_two, tril, intra, cc, 2);
        WAIT(intra, cc);
        POP(scratch_two, cc);

        // T_inv = (I + strictly_lower(Akk))^-1.
        ew(lower_mask, tril, scratch_two, cc, 2);  // lower(A), including diagonal
        WAIT(scratch_two, cc);
        POP(lower_mask, cc);
        ew(scratch_two, eye, scratch_three, cc, 2);  // diag(A)
        WAIT(scratch_three, cc);
        ew(scratch_three, scratch_two, lower_mask, cc, 1);  // -strictly_lower(A)
        WAIT(lower_mask, cc);
        POP(scratch_three, cc);
        POP(scratch_two, cc);
        invert_doubling(lower_mask, 0, t_inv, state, final_state, state_two, state_three, eye);
        WAIT(t_inv, cc);
        POP(lower_mask, cc);

        // k_dec_t = (kr * exp(G_last))^T.
        ew(scratch_one, state_update, scratch_two, ck, 2);
        WAIT(scratch_two, ck);
        POP(scratch_one, ck);
        transpose_col(scratch_two, k_decay_transposed, Kt);
        POP(scratch_two, ck);

        // dl [K,1] is the transpose of any replicated exp(G_last) row.
        transpose_col(decay, v_new, Kt);
        POP(state_update, ck);
        POP(decay, ck);
        // v_beta, kd, q_decay, intra, k_dec_t, dl, T_inv stay pushed for the writer.
    }
}
