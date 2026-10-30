// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The per-token recurrence of the GDN decode op (fused_recurrent_gated_delta_rule), one head, K = Kt*32, V = Vt*32,
// as compute-kernel functions over circular buffers whose ids the program factory assigns (every id is a parameter).
// The caller owns the token loop and the ring of state CBs; decode_token does one token:
//
//   e      = exp(g)                        constants pass, SFPU exact exp of column 0 of gcol32 -> ecol32
//   c      = q~ . k^                       constants pass, FPU: Kt products qk16 x qcol32 -> craw16[0,0], then the FPU
//                                          scalar broadcast; e, beta, c go to ebc16 (column 0 rows 0-3 / 4-7 / 8-11)
//   r      = [k^; q~] . S_old              read pass, FPU: 16x32 qk16 x 32x32 S tiles -> r16 (row 0 = r_k, row 2 = r_q)
//   u      = beta (v - e r_k)              vector pass, SFPU fp32 -> u16 (row 0 = u, rows 1-15 exactly 0)
//   o      = e r_q + c u                   vector pass, SFPU fp32 -> out16 (row 0 = o, rows 1-15 0)
//   S_new  = e S_old + k^T u               update pass: exact seed (copy_tile, UnpackToDestFp32), SFPU decay, FPU
//                                          rank-1
//
// Tile shapes live on the CBs (set_tile_dims / TileDescriptor): qk16, v16, bcol16, r16, craw16, u16, ebc16, out16 are
// 16x32, kT16 is 32x16, everything else 32x32; all fp32. CBs read by copy_tile (gcol32, v16, bcol16, S, ecol32, r16,
// ebc16) are UnpackToDestFp32 so the copies are exact. A utd CB cannot be a matmul operand (the jit-build gives it the
// Float32 unpack format and the matmul then reads garbage from SrcA; measured in test_gdn_decode_math.cpp), so every
// state CB is TWO ids over the same L1 memory (one CircularBufferConfig with two buffer indices): `utd` for the exact
// seed copies, `mm` for the read-pass matmul in1. Both ids are reserved / pushed / waited / popped in lockstep.
// DEST holds 4 fp32 tiles per acquire (half-sync); the same code runs under full sync. The packer is re-initialised
// (pack_init) whenever the output tile shape changes, the unpacker MOPs by copy_init / matmul_init per acquire, and the
// unpacker's per-source tile geometry (which only compute_kernel_hw_startup and reconfig_full_operand program, not
// matmul_init: #49266) by reconfig_full_operand before every FPU op whose operand shapes differ from the previous one.
// Every pass waits for its inputs and never pops them; decode_token pops the CBs internal to the token (ecol32,
// ebc16, r16, u16) and, with pop_inputs, the per-token inputs and S_cur (never ones32).
//
// The SFPU helpers are K1's fused_recurrent_gdn_sfpu.hpp (namespace gdn_decode_sfpu); nothing else touches the SFPU.

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "fused_recurrent_gdn_sfpu.hpp"

namespace gdn_decode {

// fp32 DEST tiles per acquire with half-sync.
inline constexpr uint32_t kDstTiles = 4;
// S tiles per acquire in the update pass; slot kUpdateGroup holds ecol32.
inline constexpr uint32_t kUpdateGroup = kDstTiles - 1;

// CB ids of one token (the factory's assignment; see ../fused_recurrent_gdn_cb.hpp).
struct TokenCbs {
    uint32_t qk16;       // Kt x 16x32: row 0 = k^, row 2 = q~ (reader)
    uint32_t kT16;       // Kt x 32x16: column 0 = k^ (reader)
    uint32_t qcol32;     // Kt x 32x32: column 0 = q~ (reader)
    uint32_t v16;        // Vt x 16x32: row 0 = v (reader, utd)
    uint32_t gcol32;     // 32x32: column 0 = g in all 32 rows (reader, utd)
    uint32_t bcol16;     // 16x32: column 0 = beta in rows 0-15 (reader, utd)
    uint32_t ones32;     // 32x32 of 1.0 (reader, once)
    uint32_t ecol32;     // 32x32: column 0 = e (compute, utd; only column 0 of faces 0 and 2 is ever read)
    uint32_t r16;        // Vt x 16x32: row 0 = r_k, row 2 = r_q (compute, utd)
    uint32_t craw16;     // 16x32: [0,0] = c (compute)
    uint32_t ccol32;     // 32x32 of c: allocated by the factory but not produced or read by these passes (c goes from
                         // DEST straight into ebc16); kept so the factory's CB plan needs no change
    uint32_t u16;        // Vt x 16x32: row 0 = u, rows 1-15 exactly 0 (compute)
    uint32_t out16;      // Vt x 16x32: row 0 = o, rows 1-15 0 (compute -> writer)
    uint32_t state_out;  // Kt*Vt x 32x32 (compute -> writer, when emit_state)
    uint32_t ebc16;      // 16x32: column 0 rows 0-3 = e, 4-7 = beta, 8-11 = c, else 0 (compute, utd)
};

// One state CB (Kt*Vt x 32x32) under its two ids: `utd` (UnpackToDestFp32, read by copy_tile for the exact seed)
// and `mm` (plain, the read-pass matmul in1). utd == mm is allowed when the CB is never a matmul operand (a test's
// output CB); it is then handled once.
struct StateCbs {
    uint32_t utd;
    uint32_t mm;
};

// Constants pass (three acquires): ecol32 <- exp(gcol32) on the column-0 vectors (column 0 of all 32 rows; the other
// columns are never read); craw16 <- sum_ki qk16[ki] x qcol32[ki] (c at [0,0]); then one acquire with the FPU scalar
// broadcast of craw16[0,0] onto ones32 (c in every element of slot 0), copies of ecol32 and bcol16, and build_ebc16
// -> ebc16 (column 0 rows 0-3 = e, 4-7 = beta, 8-11 = c). Waits gcol32 (1), qk16 (Kt), qcol32 (Kt), ones32 (1),
// bcol16 (1); pushes ecol32 (1) and ebc16 (1); craw16 is produced, consumed and popped here.
template <uint32_t Kt>
void constants_pass(
    uint32_t cb_gcol32,
    uint32_t cb_qk16,
    uint32_t cb_qcol32,
    uint32_t cb_ones32,
    uint32_t cb_bcol16,
    uint32_t cb_ecol32,
    uint32_t cb_craw16,
    uint32_t cb_ebc16) {
    CircularBuffer gcol(cb_gcol32), qk(cb_qk16), qcol(cb_qcol32), ones(cb_ones32), bcol(cb_bcol16);
    CircularBuffer ecol(cb_ecol32), craw(cb_craw16), ebc(cb_ebc16);
    gcol.wait_front(1);
    qk.wait_front(Kt);
    qcol.wait_front(Kt);
    ones.wait_front(1);
    bcol.wait_front(1);

    // 1. e = exp(g), exact, on column 0.
    ecol.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_gcol32);
    copy_tile(cb_gcol32, 0, 0);
    gdn_decode_sfpu::exp_col0_exact(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_init(cb_ecol32);
    pack_tile<true>(0, cb_ecol32, 0);
    tile_regs_release();
    ecol.push_back(1);

    // 2. c = k^ . q~ at [0,0] (row 0 of qk16 is k^, column 0 of qcol32 is q~); [2,0] = q~ . q~, unused.
    craw.reserve_back(1);
    tile_regs_acquire();
    reconfig_full_operand<SrcOrder::Reverse>(cb_qk16, cb_qcol32);
    matmul_init(cb_qk16, cb_qcol32, 0);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_tiles(cb_qk16, cb_qcol32, ki, ki, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_init(cb_craw16);
    pack_tile<true>(0, cb_craw16, 0);
    tile_regs_release();
    craw.push_back(1);

    // 3. slot 0 = c everywhere (FPU scalar broadcast of craw16[0,0] onto the ones tile), slot 1 = ecol32, slot 2 =
    //    bcol16; ebc16 built in slot 3 from column 0 rows 0-3 of each.
    craw.wait_front(1);
    ecol.wait_front(1);
    ebc.reserve_back(1);
    tile_regs_acquire();
    reconfig_full_operand(cb_ones32, cb_craw16);
    mul_bcast_scalar_init(cb_ones32, cb_craw16);
    mul_tiles_bcast_scalar(cb_ones32, cb_craw16, 0, 0, 0);
    copy_init(cb_ecol32);  // 32x32
    copy_tile(cb_ecol32, 0, 1);
    copy_init(cb_bcol16);  // 16x32
    copy_tile(cb_bcol16, 0, 2);
    gdn_decode_sfpu::build_ebc16(1, 2, 0, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_init(cb_ebc16);
    pack_tile<true>(3, cb_ebc16, 0);
    tile_regs_release();
    ebc.push_back(1);
    craw.pop_front(1);
}

// Read pass: r16[j] <- sum_ki qk16[ki] x S_cur[ki*Vt + j], up to kDstTiles output tiles per acquire. cb_s_cur is the
// plain (`mm`) id of the state. Waits qk16 (Kt) and S_cur (Kt*Vt); pushes r16 (Vt).
template <uint32_t Kt, uint32_t Vt>
void read_pass(uint32_t cb_qk16, uint32_t cb_s_cur, uint32_t cb_r16) {
    CircularBuffer qk(cb_qk16), s(cb_s_cur), r(cb_r16);
    qk.wait_front(Kt);
    s.wait_front(Kt * Vt);
    r.reserve_back(Vt);
    reconfig_full_operand<SrcOrder::Reverse>(cb_qk16, cb_s_cur);
    matmul_init(cb_qk16, cb_s_cur, 0);
    for (uint32_t j0 = 0; j0 < Vt; j0 += kDstTiles) {
        const uint32_t nn = (Vt - j0 < kDstTiles) ? (Vt - j0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t jj = 0; jj < nn; jj++) {
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(cb_qk16, cb_s_cur, ki, ki * Vt + j0 + jj, jj);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_init(cb_r16);
        for (uint32_t jj = 0; jj < nn; jj++) {
            pack_tile<true>(jj, cb_r16, j0 + jj);
        }
        tile_regs_release();
    }
    r.push_back(Vt);
}

// Vector pass, one acquire per V tile j. The live data of the 16x32 tiles is in rows 0 and 2 and rows 4-15 are zero
// paddings, so the SFPU ops run on rows 0..3 only (one 4-row band of faces 0 and 1); the constants come from ebc16's
// column 0 (band 0 = e, 1 = beta, 2 = c):
//   slot 0 = r16[j], 1 = v16[j], 2 = ebc16
//   0 <- 0 * e                  e r           (row 0 = e r_k, row 2 = e r_q)
//   1 <- 1 - 0                  v - e r
//   1 <- 1 * beta               beta (v - e r_k) in row 0; rows 1-15 then zeroed  -> u
//   0 <- row 2 of 0 in row 0    e r_q, rows 1-15 zero
//   3 <- 1; 3 <- 3 * c          c u  (u copied inside DEST: a slot cannot be reloaded after its pack)
//   0 <- 0 + 3                  o
//   pack 1 -> u16[j], 0 -> out16[j]
// Waits r16 (Vt), v16 (Vt), ebc16 (1); pushes u16 (Vt) and out16 (Vt).
template <uint32_t Vt>
void vector_pass(uint32_t cb_r16, uint32_t cb_v16, uint32_t cb_ebc16, uint32_t cb_u16, uint32_t cb_out16) {
    CircularBuffer r(cb_r16), v(cb_v16), ebc(cb_ebc16), u(cb_u16), out(cb_out16);
    r.wait_front(Vt);
    v.wait_front(Vt);
    ebc.wait_front(1);
    u.reserve_back(Vt);
    out.reserve_back(Vt);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_init(cb_r16);  // 16x32 tiles throughout
        copy_tile(cb_r16, j, 0);
        copy_tile(cb_v16, j, 1);
        copy_tile(cb_ebc16, 0, 2);
        gdn_decode_sfpu::mul_col_bcast_band0(0, 2, 0);
        gdn_decode_sfpu::sub_dst<4>(1, 0);
        gdn_decode_sfpu::mul_col_bcast_band0(1, 2, 1);
        gdn_decode_sfpu::zero_rows_from_1(1);
        gdn_decode_sfpu::move_row2_to_row0(0);
        gdn_decode_sfpu::copy_dst<4>(3, 1);
        gdn_decode_sfpu::mul_col_bcast_band0(3, 2, 2);
        gdn_decode_sfpu::add_dst<4>(0, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_init(cb_u16);
        pack_tile<true>(1, cb_u16, j);
        pack_tile<true>(0, cb_out16, j);
        tile_regs_release();
    }
    u.push_back(Vt);
    out.push_back(Vt);
}

// Fused read + vector pass (the optional form of section 8): per V tile j one acquire does the Kt products
// qk16 x S_cur[., j] into slot 0 (r_j, exactly what the read pass packs) and then the vector pass's SFPU chain on it,
// so r16 is never packed or copied. The round trip it removes is exact (fp32 pack of an fp32 DEST, UnpackToDestFp32
// copy), so the bits are identical to read_pass + vector_pass. cb_s_cur is the plain (`mm`) id. Waits qk16 (Kt),
// S_cur (Kt*Vt), v16 (Vt), ebc16 (1); pushes u16 (Vt) and out16 (Vt).
template <uint32_t Kt, uint32_t Vt>
void read_vector_pass(
    uint32_t cb_qk16, uint32_t cb_s_cur, uint32_t cb_v16, uint32_t cb_ebc16, uint32_t cb_u16, uint32_t cb_out16) {
    CircularBuffer qk(cb_qk16), s(cb_s_cur), v(cb_v16), ebc(cb_ebc16), u(cb_u16), out(cb_out16);
    qk.wait_front(Kt);
    s.wait_front(Kt * Vt);
    v.wait_front(Vt);
    ebc.wait_front(1);
    u.reserve_back(Vt);
    out.reserve_back(Vt);
    reconfig_full_operand<SrcOrder::Reverse>(cb_qk16, cb_s_cur);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        matmul_init(cb_qk16, cb_s_cur, 0);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(cb_qk16, cb_s_cur, ki, ki * Vt + j, 0);
        }
        copy_init(cb_v16);  // 16x32 tiles
        copy_tile(cb_v16, j, 1);
        copy_tile(cb_ebc16, 0, 2);
        gdn_decode_sfpu::mul_col_bcast_band0(0, 2, 0);
        gdn_decode_sfpu::sub_dst<4>(1, 0);
        gdn_decode_sfpu::mul_col_bcast_band0(1, 2, 1);
        gdn_decode_sfpu::zero_rows_from_1(1);
        gdn_decode_sfpu::move_row2_to_row0(0);
        gdn_decode_sfpu::copy_dst<4>(3, 1);
        gdn_decode_sfpu::mul_col_bcast_band0(3, 2, 2);
        gdn_decode_sfpu::add_dst<4>(0, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_init(cb_u16);
        pack_tile<true>(1, cb_u16, j);
        pack_tile<true>(0, cb_out16, j);
        tile_regs_release();
    }
    u.push_back(Vt);
    out.push_back(Vt);
}

// Update pass: S_next[i, j] <- e * S_cur[i, j] + kT16[i] x u16[j], groups of up to kUpdateGroup S tiles per acquire
// (exact seed by copy_tile from the utd S_cur, decay on the SFPU against ecol32 in slot kUpdateGroup, rank-1 accumulate
// by matmul_tiles onto the seeded slots). With emit_state every slot is also packed to state_out. cb_s_cur is the utd
// id of the current state; s_next is the next state's pair (packed through s_next.utd, both ids reserved and pushed).
// The outputs are reserved, packed (indices relative to the reservation; the (i, j) order is sequential) and pushed
// per acquire group, so a writer can stream the state tiles while the pass runs. Waits kT16 (Kt), u16 (Vt), S_cur
// (Kt*Vt), ecol32 (1); pushes S_next (Kt*Vt in total) and, with emit_state, state_out (Kt*Vt in total).
template <uint32_t Kt, uint32_t Vt>
void update_pass(
    uint32_t cb_kT16,
    uint32_t cb_u16,
    uint32_t cb_s_cur,
    uint32_t cb_ecol32,
    StateCbs s_next,
    uint32_t cb_state_out,
    bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t cb_s_next = s_next.utd;
    const bool two_ids = s_next.mm != s_next.utd;
    CircularBuffer kT(cb_kT16), u(cb_u16), s(cb_s_cur), ecol(cb_ecol32), sn(cb_s_next), sn_mm(s_next.mm);
    CircularBuffer st(cb_state_out);
    kT.wait_front(Kt);
    u.wait_front(Vt);
    s.wait_front(kv);
    ecol.wait_front(1);
    // Source geometry for the rank-1 matmul (32x16 x 16x32); the copies in the loop only change formats.
    reconfig_full_operand<SrcOrder::Reverse>(cb_kT16, cb_u16);
    for (uint32_t i = 0; i < Kt; i++) {
        for (uint32_t j0 = 0; j0 < Vt; j0 += kUpdateGroup) {
            const uint32_t nn = (Vt - j0 < kUpdateGroup) ? (Vt - j0) : kUpdateGroup;
            sn.reserve_back(nn);
            if (two_ids) {
                sn_mm.reserve_back(nn);
            }
            if (emit_state) {
                st.reserve_back(nn);
            }
            tile_regs_acquire();
            copy_init(cb_s_cur);
            for (uint32_t jj = 0; jj < nn; jj++) {
                copy_tile(cb_s_cur, i * Vt + j0 + jj, jj);
            }
            copy_tile(cb_ecol32, 0, kUpdateGroup);
            for (uint32_t jj = 0; jj < nn; jj++) {
                gdn_decode_sfpu::mul_col_bcast<32>(jj, kUpdateGroup);
            }
            matmul_init(cb_kT16, cb_u16, 0);
            for (uint32_t jj = 0; jj < nn; jj++) {
                matmul_tiles(cb_kT16, cb_u16, i, j0 + jj, jj);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_init(cb_s_next);
            for (uint32_t jj = 0; jj < nn; jj++) {
                pack_tile<true>(jj, cb_s_next, jj);
                if (emit_state) {
                    pack_tile<true>(jj, cb_state_out, jj);
                }
            }
            tile_regs_release();
            sn.push_back(nn);
            if (two_ids) {
                sn_mm.push_back(nn);
            }
            if (emit_state) {
                st.push_back(nn);
            }
        }
    }
}

struct NoHook {
    ALWI void operator()(uint32_t /*pass_done*/) const {}
};

// One token: constants -> read -> vector -> update, then pop the token-internal CBs. s_cur is the state the token
// reads (the reader's S at t = 0, the ring after), s_next the ring CB it writes (left in its CBs for the next token);
// emit_state also packs S_new to state_out. With pop_inputs (the op) the per-token inputs qk16, kT16, qcol32, v16,
// gcol32, bcol16 and both ids of s_cur are popped at the end as well; ones32 never is. The caller must have run
// compute_kernel_hw_startup once (both source orders give the same bits). FuseReadVector selects read_vector_pass
// (one acquire per V tile for the products and the vector chain; r16 unused) instead of read_pass + vector_pass; the
// bits are the same. `hook(p)` is called after pass p (0 constants, 1 read, 2 vector, 3 update; fused: 1 after the
// fused pass and no 2); the gtest stamps the MATH clock there, the op passes nothing.
template <uint32_t Kt, uint32_t Vt, bool FuseReadVector = false, typename Hook = NoHook>
void decode_token(
    const TokenCbs& cb,
    StateCbs s_cur,
    StateCbs s_next,
    bool emit_state,
    bool pop_inputs = true,
    Hook&& hook = Hook{}) {
    constexpr uint32_t kv = Kt * Vt;
    constants_pass<Kt>(cb.gcol32, cb.qk16, cb.qcol32, cb.ones32, cb.bcol16, cb.ecol32, cb.craw16, cb.ebc16);
    hook(0);
    if constexpr (FuseReadVector) {
        read_vector_pass<Kt, Vt>(cb.qk16, s_cur.mm, cb.v16, cb.ebc16, cb.u16, cb.out16);
        hook(1);
    } else {
        read_pass<Kt, Vt>(cb.qk16, s_cur.mm, cb.r16);
        hook(1);
        vector_pass<Vt>(cb.r16, cb.v16, cb.ebc16, cb.u16, cb.out16);
        hook(2);
    }
    update_pass<Kt, Vt>(cb.kT16, cb.u16, s_cur.utd, cb.ecol32, s_next, cb.state_out, emit_state);
    hook(3);
    CircularBuffer(cb.ecol32).pop_front(1);
    CircularBuffer(cb.ebc16).pop_front(1);
    if constexpr (!FuseReadVector) {
        CircularBuffer(cb.r16).pop_front(Vt);
    }
    CircularBuffer(cb.u16).pop_front(Vt);
    if (pop_inputs) {
        CircularBuffer(cb.qk16).pop_front(Kt);
        CircularBuffer(cb.kT16).pop_front(Kt);
        CircularBuffer(cb.qcol32).pop_front(Kt);
        CircularBuffer(cb.v16).pop_front(Vt);
        CircularBuffer(cb.gcol32).pop_front(1);
        CircularBuffer(cb.bcol16).pop_front(1);
        CircularBuffer(s_cur.utd).pop_front(kv);
        if (s_cur.mm != s_cur.utd) {
            CircularBuffer(s_cur.mm).pop_front(kv);
        }
    }
}

}  // namespace gdn_decode
