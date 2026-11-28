// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DEST-level compositions of the GDN decode compute kernel (fused_recurrent_gated_delta_rule, two-pass form), written
// on the compute API only: every SFPU primitive the passes use (column broadcast on a 4-row band, the parked scalar,
// the rows forms of the binary and copy ops, fill rows, row and band copies, the exact exp) is a compute-API op with
// its own tests. Each composition is called between tile_regs_acquire() and tile_regs_commit() on DEST slot indices
// (0..3 with an fp32 DEST in half-sync) and is a no-op off the MATH thread, as the ops it is made of.
//
// DEST geometry the compositions rely on (the API's contract, tested in tests/tt_metal/tt_metal/llk/): a 16x32 tile
// occupies faces 0 and 1 of its slot; the vector-mode R forms touch those two faces and the ITERATIONS = 2 forms
// touch rows 0..3 of them, which is the band the token vectors live in (row 0 of v16 / r16 / u16, rows 0 and 2 of
// qk16). Every op here writes whole rows, so a packed tile never carries a partially written row.
//
// Inits: the column broadcast's init (mask, replay program) and the binary ops' init run right before their ops,
// inside the acquire, because the broadcast init, the parked-scalar load (LREG14, and LREG6, the broadcast's mask)
// and the exact exp's init (LREG12..14) overwrite each other's SFPU state. The fill and copy ops only need the
// zero-increment DEST address mode every family here leaves in place, so their two inits run once per acquire
// (vector_chain_init) ahead of zero_rows_from_1, move_row2_to_row0 and copy_rows4, instead of once per op: four
// inits per V tile were about a tenth of the vector pass.

//
// One piece is still private sfpi code: the in-kernel norm (square_acc_rows4, norm_scalars) needs an 8-lane row sum
// the compute API does not have yet (a future sfpu_row_sum); it keeps the LLK's bracket functions until then.

#pragma once

#include <cstdint>

#include "api/compute/copy_dest_rows.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#endif

namespace gdn_decode_sfpu {

#ifdef TRISC_MATH
namespace detail {
inline constexpr uint32_t kSlotVecs = 32;  // sfpi DST vectors per DEST slot (the 32x32 fp32 stride)
inline constexpr uint32_t kFaceVecs = 8;   // per 16x16 face
// The four sfpi vectors that hold rows 0..3 of a 16x32 tile: band 0 of faces 0 and 1, both column parities.
inline constexpr uint32_t kBand0Vecs[4] = {0, 1, 8, 9};

// Brackets an sfpi body the way the LLK's unary / binary SFPU params functions do: DST base at slot 0 (the bodies
// address slots absolutely), then the address reset.
template <typename Body>
inline void run(Body&& body) {
    _llk_math_eltwise_sfpu_start_(0);
    body();
    _llk_math_eltwise_sfpu_done_();
}

// acc's rows 0..3 (the vectors of kBand0Vecs) <- (first ? 0 : acc) + x * x over slot src's rows 0..3: the squares of
// a row tile's live rows (row 0 = k, row 2 = q), fp32 products of exact copies, summed over the K tiles.
inline void square_acc_rows4_body(uint32_t acc, uint32_t src, bool first) {
    const uint32_t ab = acc * kSlotVecs, sb = src * kSlotVecs;
#pragma GCC unroll 4
    for (uint32_t i = 0; i < 4; i++) {
        const sfpi::vFloat x = sfpi::dst_reg[sb + kBand0Vecs[i]];
        if (first) {
            sfpi::dst_reg[ab + kBand0Vecs[i]] = x * x;
        } else {
            const sfpi::vFloat a = sfpi::dst_reg[ab + kBand0Vecs[i]];
            sfpi::dst_reg[ab + kBand0Vecs[i]] = a + x * x;
        }
    }
}
// Every lane of each 8-lane subvector <- the subvector's sum (rotate by 1, 2, 4 and add).
inline sfpi::vFloat subvec_sum(sfpi::vFloat v) {
    v = v + sfpi::subvec_shflror1(v);
    sfpi::vFloat t = sfpi::subvec_shflror1(v);
    t = sfpi::subvec_shflror1(t);
    v = v + t;
    t = sfpi::subvec_shflror1(v);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    t = sfpi::subvec_shflror1(t);
    return v + t;
}
// The norm tile N in slot out, a 16x32 tile: column 0 row 0 = ik = 1 / sqrt(k.k + eps), row 2 = s_q = scale / sqrt(q.q
// + eps), everything else of faces 0 and 1 zero. k.k and q.q come from the squares accumulated in slot acc's rows 0..3
// (row 0 = k, row 2 = q): the four band-0 vectors (faces 0 and 1, both column parities) summed lane-wise, then each
// row's 8 lanes summed, so lane 0 of a row's subvector holds the row's sum (fp32 throughout). The rsqrt is the
// LLK's exact-mode body (rsqrt_tile_init programs its constants). Without Norm: row 0 = 1, row 2 = scale.
template <bool Norm>
inline void norm_scalars_body(uint32_t acc, uint32_t out, uint32_t scale_bits, uint32_t eps_bits) {
    const sfpi::vFloat scale = __builtin_bit_cast(float, scale_bits);
    sfpi::vFloat v;
    if constexpr (Norm) {
        const uint32_t ab = acc * kSlotVecs;
        const sfpi::vFloat a0 = sfpi::dst_reg[ab + kBand0Vecs[0]];
        const sfpi::vFloat a1 = sfpi::dst_reg[ab + kBand0Vecs[1]];
        const sfpi::vFloat a2 = sfpi::dst_reg[ab + kBand0Vecs[2]];
        const sfpi::vFloat a3 = sfpi::dst_reg[ab + kBand0Vecs[3]];
        v = (a0 + a1) + (a2 + a3);
        v = subvec_sum(v);
        const sfpi::vFloat eps = __builtin_bit_cast(float, eps_bits);
        v = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(v + eps);
    } else {
        v = 1.0f;
    }
    // Lane l holds row l / 8, column 2 * (l % 8); vConstTileId = 2 * l. Row 2 (lanes 16..23) takes the scale; only
    // the column-0 lanes of rows 0 and 2 (tile ids 0 and 32) survive.
    v_if(sfpi::vConstTileId >= 32 && sfpi::vConstTileId < 48) { v = v * scale; }
    v_endif;
    v_if((sfpi::vConstTileId & static_cast<int>(0xE | 16)) != 0) { v = 0.0f; }
    v_endif;
    const uint32_t ob = out * kSlotVecs;
    sfpi::dst_reg[ob + 0] = v;
#pragma GCC unroll 15
    for (uint32_t i = 1; i < 2 * kFaceVecs; i++) {
        sfpi::dst_reg[ob + i] = 0.0f;
    }
}
}  // namespace detail
#endif  // TRISC_MATH

// The in-kernel norm, two calls per token in the constants pass (the one private sfpi piece left, see the top):
//   square_acc_rows4(acc, src, first): rows 0..3 of acc <- (first ? 0 : acc) + src * src (src: an exact copy of a
//     qk16 tile, row 0 = k, row 2 = q), once per K tile;
//   norm_scalars<Norm>(acc, out, scale_bits): the norm tile N in slot out (16x32: column 0 row 0 = 1/|k|, row 2 =
//     scale/|q|, else 0) from the accumulated squares, eps = 1e-6 inside the root as the golden's l2_norm; without
//     Norm, 1 and scale. Programs the SFPU's sqrt constants (LREG12, 13); park a scalar with sfpu_bcast_scalar_load
//     after it.
inline void square_acc_rows4(uint32_t acc, uint32_t src, bool first) {
    MATH((detail::run([&] { detail::square_acc_rows4_body(acc, src, first); })));
}
template <bool Norm>
inline void norm_scalars(uint32_t acc, uint32_t out, uint32_t scale_bits) {
    constexpr uint32_t kEps = 0x358637BDu;  // 1e-6f
    if constexpr (Norm) {
        ckernel::rsqrt_tile_init();
    }
    MATH((detail::run([&] { detail::norm_scalars_body<Norm>(acc, out, scale_bits, kEps); })));
}

// build_ebc16 with a fourth band: rows 12..15 <- rows 0..3 of DEST[slot_n] (the norm tile ncol16: column 0 = ik, 0,
// s_q, 0). All 16 rows are written, so slot_out may be slot_e (in place: band 0 is already e's).
inline void build_ebcn16(uint32_t slot_e, uint32_t slot_b, uint32_t slot_c, uint32_t slot_n, uint32_t slot_out) {
    ckernel::copy_dest_values_init();
    if (slot_out != slot_e) {
        ckernel::copy_dest_rows(slot_out, 0, slot_e, 0);
    }
    ckernel::copy_dest_rows(slot_out, 1, slot_b, 0);
    ckernel::copy_dest_rows(slot_out, 2, slot_c, 0);
    ckernel::copy_dest_rows(slot_out, 3, slot_n, 0);
}

// The vector pass's constants tile ebc16, built in DEST[slot_out] as a 16x32 tile ready for pack_tile:
//   rows 0..3  <- rows 0..3 of DEST[slot_e]  (ecol32 after the exp: column 0 = e)
//   rows 4..7  <- rows 0..3 of DEST[slot_b]  (bcol16: column 0 = beta)
//   rows 8..11 <- rows 0..3 of DEST[slot_c]  (the FPU's c tile: c everywhere)
//   rows 12..15 <- 0
// Only column 0 of each band is read by the consumers (mul_col_bcast_band(dst, ebc, 0, band) for e, beta and c read
// column 0 of their band and mask the rest), so the other columns of rows 0..11 carry whatever the sources hold.
// slot_out must be a free slot (it is zeroed first).
inline void build_ebc16(uint32_t slot_e, uint32_t slot_b, uint32_t slot_c, uint32_t slot_out) {
    ckernel::fill_tile_init();
    ckernel::fill_rows_tile<0, 16>(slot_out, 0.0f);
    ckernel::copy_dest_values_init();
    ckernel::copy_dest_rows(slot_out, 0, slot_e, 0);
    ckernel::copy_dest_rows(slot_out, 1, slot_b, 0);
    ckernel::copy_dest_rows(slot_out, 2, slot_c, 0);
}

// The fill and copy inits the three row compositions below rely on; once per acquire, before them.
inline void vector_chain_init() {
    ckernel::fill_tile_init();
    ckernel::copy_dest_values_init();
}

// Rows 0..3 of DEST[dst] <- DEST[dst][r][c] * DEST[dst_col][4 * band + r][0]: the band broadcast with the column
// values taken from band `band` of the constants tile (0 = e, 1 = beta, 2 = c). Writes rows 0..3 of dst only.
inline void mul_band0_by(uint32_t dst, uint32_t dst_col, uint32_t band) {
    ckernel::sfpu_mul_bcast_col_init();
    ckernel::sfpu_mul_bcast_col_band(dst, dst_col, 0, band);
}

// a <- a - b and a <- a + b on rows 0..3 only (the ITERATIONS = 2 form of the binary ops under VectorMode::R).
inline void sub_rows4(uint32_t a, uint32_t b) {
    ckernel::sub_binary_tile_init();
    ckernel::sub_binary_tile<ckernel::DstRoundingMode::Default, true, 2>(a, b, a, ckernel::VectorMode::R);
}
inline void add_rows4(uint32_t a, uint32_t b) {
    ckernel::add_binary_tile_init();
    ckernel::add_binary_tile<ckernel::DstRoundingMode::Default, true, 2>(a, b, a, ckernel::VectorMode::R);
}

// DEST[to] rows 0..3 <- DEST[from] rows 0..3, bit-exact. After vector_chain_init.
inline void copy_rows4(uint32_t to, uint32_t from) {
    ckernel::copy_dest_values<DataFormat::Float32, 2>(from, to, ckernel::VectorMode::R);
}

// 16x32 tile in slot dst: rows 1..15 <- 0, row 0 kept (the form u16 needs: a NaN in a padding row of r16 or v16 would
// otherwise survive into u16). After vector_chain_init.
inline void zero_rows_from_1(uint32_t dst) { ckernel::fill_rows_tile<1, 16>(dst, 0.0f); }

// 16x32 tile in slot dst: row 0 <- row 2, rows 1..15 <- 0 (r_q to the output row, the padding cleared). After
// vector_chain_init.
inline void move_row2_to_row0(uint32_t dst) {
    ckernel::copy_dest_row(dst, 0, dst, 2);
    ckernel::fill_rows_tile<1, 16>(dst, 0.0f);
}

}  // namespace gdn_decode_sfpu
