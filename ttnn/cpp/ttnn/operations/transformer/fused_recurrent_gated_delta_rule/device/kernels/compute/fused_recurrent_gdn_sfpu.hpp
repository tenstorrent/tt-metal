// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DEST-level SFPU helpers of the GDN decode compute kernel (fused_recurrent_gated_delta_rule, two-pass form).
// Every helper is called from a compute kernel between tile_regs_acquire() and tile_regs_commit(), takes DEST slot
// indices (0..3 with an fp32 DEST in half-sync), and is a no-op off the MATH thread. Each helper runs its own SFPU
// init: the inits are a handful of instructions, and the column broadcast, the exact exp and the sfpi passes below
// all overwrite the LREG / replay-buffer state the others rely on, so nothing may be hoisted.
//
// DEST geometry (Blackhole, fp32 DEST, confirmed by the probe in tests/tt_metal/tt_metal/llk/test_gdn_decode_sfpu.cpp):
//   - a DEST slot is 32 sfpi vectors (dst_reg[32 * slot + v]) whatever the tile shape; a 16x32 tile uses faces 0
//     and 1 = vectors 0..15 of its slot, a 32x32 tile all four faces at vectors 0, 8, 16, 24;
//   - within a face, vector v = 2 * band + grp holds tile rows 4 * band .. 4 * band + 3 of that face: lane 8 * j + k
//     is row 4 * band + j, column 2 * k + grp (grp 0 = even columns, grp 1 = odd columns);
//   - sfpi::vConstTileId = 2 * lane, so the row within the band is lane >> 3 = tileid >> 4.
//   Consequences: tile row 0 is lanes 0-7 of vectors 0 and 1 (and 8, 9 for columns 16-31), row 1 is lanes 8-15 of the
//   same vectors, row 2 lanes 16-23, row 3 lanes 24-31; rows 0 and 2 are NOT lane-aligned across vectors (the rows
//   lane-aligned with row 0 are 4, 8 and 12). Moving row 2 to row 0 is therefore a subvector transpose (SFPTRANSP
//   moves 8-lane subvector i of vector j to subvector j of vector i), not a vector copy.
//
// Column tiles: `mul_col_bcast` multiplies row r of the data tile by element [r, 0] of the column tile, i.e. it reads
// column 0 of faces 0 and 2 only; the other columns of the column tile are never read.
//
// DEST lifetime: tile_regs_release() clears the released half (the LLK's pack release issues ZEROACC CLR_HALF by
// default), so nothing survives from one acquire to the next: every slot must be rewritten in the acquire that reads
// it. The clear invalidates rows rather than overwriting them (the packer then reads zeros, the SFPU may still see the
// stale words), so a stale read is undefined behaviour, not a feature. A slot a 16x32 copy_tile fills has its faces
// 2 and 3 invalidated (packed as zeros), and partially written rows must not be read: every pass here writes whole
// rows (both column parities of every vector it touches).

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/sfpu_binary_bcast.h"
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
// The eight vectors that hold column 0 of a 32x32 tile: the even (even-column) vectors of faces 0 and 2.
inline constexpr uint32_t kCol0Vecs[8] = {0, 2, 4, 6, 16, 18, 20, 22};

// Brackets an sfpi body the way the LLK's unary / binary SFPU params functions do: DST base at slot 0 (the bodies
// address slots absolutely), then the address reset.
template <typename Body>
inline void run(Body&& body) {
    _llk_math_eltwise_sfpu_start_(0);
    body();
    _llk_math_eltwise_sfpu_done_();
}

// Rows FROM..15 of the 16x32 tile in slot `dst` <- 0.0 (faces 0 and 1 only). Rows 0..3 live in vectors 0 and 1 of
// each face as lane groups of 8, so rows FROM..3 are a lane-masked store; rows 4..15 are whole vectors.
template <uint32_t FROM>
inline void zero_rows_from_body(uint32_t dst) {
    static_assert(FROM >= 1 && FROM <= 3, "zero_rows_from: FROM must be in 1..3");
    const uint32_t base = dst * kSlotVecs;
#pragma GCC unroll 2
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t fb = base + f * kFaceVecs;
#pragma GCC unroll 2
        for (uint32_t g = 0; g < 2; g++) {
            sfpi::vFloat x = sfpi::dst_reg[fb + g];
            v_if(sfpi::vConstTileId >= static_cast<int>(16 * FROM)) { x = 0.0f; }  // lane >= 8 * FROM
            v_endif;
            sfpi::dst_reg[fb + g] = x;
        }
#pragma GCC unroll 6
        for (uint32_t v = 2; v < kFaceVecs; v++) {
            sfpi::dst_reg[fb + v] = 0.0f;
        }
    }
}

// Row 0 of the 16x32 tile in slot `dst` <- its row 2; rows 1..15 <- 0.0. Per face and column parity, the vector
// holding rows 0..3 goes through SFPTRANSP against three zero vectors: output 2 is [subvector 2 of the input, 0, 0, 0]
// = [row 2, 0, 0, 0]. Both parities of a face ride one transpose (inputs 0 and 4 of the 8-vector form).
inline void move_row2_to_row0_body(uint32_t dst) {
    const uint32_t base = dst * kSlotVecs;
#pragma GCC unroll 2
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t fb = base + f * kFaceVecs;
        sfpi::vFloat a = sfpi::dst_reg[fb + 0];
        sfpi::vFloat e = sfpi::dst_reg[fb + 1];
        sfpi::vFloat b = 0.0f, c = 0.0f, d = 0.0f, f2 = 0.0f, g = 0.0f, h = 0.0f;
        sfpi::subvec_transp(a, b, c, d, e, f2, g, h);
        sfpi::dst_reg[fb + 0] = c;
        sfpi::dst_reg[fb + 1] = g;
#pragma GCC unroll 6
        for (uint32_t v = 2; v < kFaceVecs; v++) {
            sfpi::dst_reg[fb + v] = 0.0f;
        }
    }
}

// Slot `to` <- slot `from`: rows 0..3 (ROWS = 4, the vectors of kBand0Vecs), faces 0 and 1 (ROWS = 16, a 16x32
// tile) or the whole slot (ROWS = 32). Bit-exact.
template <uint32_t ROWS>
inline void copy_dst_body(uint32_t to, uint32_t from) {
    static_assert(ROWS == 4 || ROWS == 16 || ROWS == 32, "copy_dst: rows 0..3, a 16x32 or a 32x32 tile");
    const uint32_t tb = to * kSlotVecs, fb = from * kSlotVecs;
    if constexpr (ROWS == 4) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; i++) {
            const sfpi::vFloat x = sfpi::dst_reg[fb + kBand0Vecs[i]];
            sfpi::dst_reg[tb + kBand0Vecs[i]] = x;
        }
    } else {
#pragma GCC unroll 8
        for (uint32_t v = 0; v < ROWS; v++) {  // 16 or 32 vectors
            const sfpi::vFloat x = sfpi::dst_reg[fb + v];
            sfpi::dst_reg[tb + v] = x;
        }
    }
}

// The LLK's column-broadcast band kernel over the first BANDS 4-row bands of faces 0 and 1 (BANDS = 4: rows 0..15,
// the first half of _calculate_sfpu_binary_bcast_col_full_tile_; BANDS = 1: rows 0..3 only), instruction for
// instruction, so the result is bit-identical to the full-tile API on those rows. Needs sfpu_bcast_col_init()
// (mask + replay program) immediately before.
template <ckernel::BinaryOp BINOP, uint32_t BANDS>
inline void bcast_col_bands_body(uint32_t dst, uint32_t dst_col) {
    using namespace ckernel::sfpu;
    static_assert(BANDS >= 1 && BANDS <= NUM_ROW_BANDS_PER_FACE_HALF, "bcast_col_bands: 1..4 bands");
    const uint32_t data_base = dst * DEST_TILE_SIZE_RAW;
    const uint32_t bcast_base = dst_col * DEST_TILE_SIZE_RAW;
    for (uint32_t band = 0; band < BANDS; band++) {
        const uint32_t off = band * ROW_BAND_STRIDE;
        _process_col_bcast_row_band_<BINOP>(
            bcast_base + FACE0_BASE + off, FACE0_BASE + off, FACE1_BASE + off, data_base, data_base);
    }
}

// a <- a - b (SUB) or a + b on rows 0..3 only (the vectors of kBand0Vecs): the same sfpi expression as the LLK's
// calculate_sfpu_binary body, so the bits are those of sub_binary_tile / add_binary_tile on those rows.
template <bool SUB>
inline void addsub_rows4_body(uint32_t a, uint32_t b) {
    const uint32_t ab = a * kSlotVecs, bb = b * kSlotVecs;
#pragma GCC unroll 4
    for (uint32_t i = 0; i < 4; i++) {
        const sfpi::vFloat in0 = sfpi::dst_reg[ab + kBand0Vecs[i]];
        const sfpi::vFloat in1 = sfpi::dst_reg[bb + kBand0Vecs[i]];
        if constexpr (SUB) {
            sfpi::dst_reg[ab + kBand0Vecs[i]] = in0 - in1;
        } else {
            sfpi::dst_reg[ab + kBand0Vecs[i]] = in0 + in1;
        }
    }
}

// exp (the fp32-accurate body exp_tile<false> runs on an fp32 DEST, _sfpu_exp_fp32_accurate_) on the eight vectors
// that hold column 0 of a 32x32 tile; the other columns are left as they are.
inline void exp_col0_body(uint32_t dst) {
    const uint32_t base = dst * kSlotVecs;
#pragma GCC unroll 0
    for (uint32_t i = 0; i < 8; i++) {
        const sfpi::vFloat x = sfpi::dst_reg[base + kCol0Vecs[i]];
        sfpi::dst_reg[base + kCol0Vecs[i]] = ckernel::sfpu::_sfpu_exp_fp32_accurate_(x);
    }
}

}  // namespace detail
#endif  // TRISC_MATH

// ---- public helpers ------------------------------------------------------------------------------------------------

// DEST[dst] <- exp(DEST[dst]) in exact mode (the fp32-accurate SFPU exp; math_approx_mode is irrelevant, the template
// flag selects it). MODE: VectorMode::RC = the whole 32x32 slot; VectorMode::C = faces 0 and 2 only, enough for a
// column tile whose other columns are never read (half the cost; faces 1 and 3 keep their old contents).
// Reads and writes slot dst only.
template <ckernel::VectorMode MODE = ckernel::VectorMode::RC>
inline void exp_tile_exact(uint32_t dst) {
    ckernel::exp_tile_init<false>();
    ckernel::exp_tile<false>(dst, MODE);
}

// exp in exact mode on column 0 only of the 32x32 tile in slot dst: the eight vectors that hold it, which also carry
// the other even columns of faces 0 and 2 (those get exp too); the odd columns and faces 1 and 3 keep their values.
// The per-element body is the one exp_tile_exact runs on an fp32 DEST, so column 0 is bit-identical to it. For
// gcol32 -> ecol32: the only consumer of ecol32 is mul_col_bcast, which reads column 0 alone. Needs no init.
inline void exp_col0_exact(uint32_t dst) {
    MATH((detail::run([&] { detail::exp_col0_body(dst); })));
}

// Rows 0..3 of DEST[dst] <- DEST[dst][r][c] * DEST[dst_col][4 * col_band + r][0] (r = 0..3): the LLK band kernel
// with the column-0 values taken from 4-row band `col_band` (0..3) of face 0 of the column tile instead of its band
// 0. One 16x32 constants tile can then carry several row-0 scalars (e in rows 0..3, beta in rows 4..7, c in rows
// 8..11 of column 0): one copy_tile per V tile instead of three. Bit-identical to mul_col_bcast<4> with a column tile
// holding those values in rows 0..3. Writes rows 0..3 of dst only; dst_col is read only.
inline void mul_col_bcast_band0(uint32_t dst, uint32_t dst_col, uint32_t col_band) {
    ckernel::sfpu_mul_bcast_col_init();
    MATH((detail::run([&] {
        using namespace ckernel::sfpu;
        const uint32_t data_base = dst * DEST_TILE_SIZE_RAW;
        _process_col_bcast_row_band_<ckernel::BinaryOp::MUL>(
            dst_col * DEST_TILE_SIZE_RAW + FACE0_BASE + col_band * ROW_BAND_STRIDE,
            FACE0_BASE,
            FACE1_BASE,
            data_base,
            data_base);
    })));
}

// DEST[dst][r][c] <- DEST[dst][r][c] * DEST[dst_col][r][0] for every column c (the column broadcast today's decode
// kernel uses). ROWS = 32: all four faces of a 32x32 data tile (reads column 0 of faces 0 and 2 of dst_col);
// ROWS = 16: faces 0 and 1 only, for a 16x32 data tile (reads column 0 of face 0 of dst_col, so a 16x32 column tile
// is fine too); ROWS = 4: rows 0..3 only (one 4-row band of faces 0 and 1), for a 16x32 tile whose rows 4..15 are
// never read or are zeroed afterwards (r16 / v16 / u16: rows 0 and 2 live). Every variant is the LLK's band kernel,
// so the rows it touches are bit-identical to the full-tile API. Writes slot dst only; dst_col is read only.
template <uint32_t ROWS = 32>
inline void mul_col_bcast(uint32_t dst, uint32_t dst_col) {
    static_assert(ROWS == 4 || ROWS == 16 || ROWS == 32, "mul_col_bcast: rows 0..3, a 16x32 or a 32x32 data tile");
    ckernel::sfpu_mul_bcast_col_init();
    if constexpr (ROWS == 32) {
        ckernel::sfpu_mul_bcast_col(dst, dst_col);
    } else {
        MATH((detail::run(
            [&] { detail::bcast_col_bands_body<ckernel::BinaryOp::MUL, (ROWS == 4 ? 1u : 4u)>(dst, dst_col); })));
    }
}

// DEST[dst_a] <- DEST[dst_a] - DEST[dst_b], elementwise fp32 (the compute API's sub_binary_tile). ROWS = 32: all
// four faces; ROWS = 16: faces 0 and 1 (VectorMode::R), for 16x32 tiles; ROWS = 4: rows 0..3 only (the same sfpi
// expression as the LLK body, so the same bits on those rows). Writes dst_a only.
template <uint32_t ROWS = 32>
inline void sub_dst(uint32_t dst_a, uint32_t dst_b) {
    static_assert(ROWS == 4 || ROWS == 16 || ROWS == 32, "sub_dst: rows 0..3, a 16x32 or a 32x32 tile");
    ckernel::sub_binary_tile_init();
    if constexpr (ROWS == 32) {
        ckernel::sub_binary_tile(dst_a, dst_b, dst_a);
    } else if constexpr (ROWS == 4) {
        MATH((detail::run([&] { detail::addsub_rows4_body<true>(dst_a, dst_b); })));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::SUB, 8 /* ITERATIONS */, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            dst_a,
            dst_b,
            dst_a,
            ckernel::VectorMode::R)));
    }
}

// DEST[dst_a] <- DEST[dst_a] + DEST[dst_b], elementwise fp32 (the compute API's add_binary_tile); ROWS as sub_dst.
template <uint32_t ROWS = 32>
inline void add_dst(uint32_t dst_a, uint32_t dst_b) {
    static_assert(ROWS == 4 || ROWS == 16 || ROWS == 32, "add_dst: rows 0..3, a 16x32 or a 32x32 tile");
    ckernel::add_binary_tile_init();
    if constexpr (ROWS == 32) {
        ckernel::add_binary_tile(dst_a, dst_b, dst_a);
    } else if constexpr (ROWS == 4) {
        MATH((detail::run([&] { detail::addsub_rows4_body<false>(dst_a, dst_b); })));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::ADD, 8 /* ITERATIONS */, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            dst_a,
            dst_b,
            dst_a,
            ckernel::VectorMode::R)));
    }
}

// 16x32 tile in slot dst: rows 2..15 <- 0.0, rows 0 and 1 kept. Writes faces 0 and 1 of dst only.
inline void zero_rows_from_2(uint32_t dst) {
    MATH((detail::run([&] { detail::zero_rows_from_body<2>(dst); })));
}

// 16x32 tile in slot dst: rows 1..15 <- 0.0, row 0 kept (the form u16 needs: a NaN in row 1 of r16 or v16 would
// otherwise survive into u16). Writes faces 0 and 1 of dst only.
inline void zero_rows_from_1(uint32_t dst) {
    MATH((detail::run([&] { detail::zero_rows_from_body<1>(dst); })));
}

// 16x32 tile in slot dst: row 0 <- row 2, rows 1..15 <- 0.0. Writes faces 0 and 1 of dst only.
inline void move_row2_to_row0(uint32_t dst) {
    MATH((detail::run([&] { detail::move_row2_to_row0_body(dst); })));
}

// DEST[to] <- DEST[from], bit-exact; ROWS = 4 copies rows 0..3, 16 faces 0 and 1 (a 16x32 tile), 32 the whole slot.
// Lets a tile that must survive a pack (u16) also feed a further in-place SFPU op in the same acquire, instead of a
// reload from L1 after the pack (which is impossible anyway: tile_regs_commit flips the DEST half).
template <uint32_t ROWS = 16>
inline void copy_dst(uint32_t to, uint32_t from) {
    static_assert(ROWS == 4 || ROWS == 16 || ROWS == 32, "copy_dst: rows 0..3, a 16x32 or a 32x32 tile");
    MATH((detail::run([&] { detail::copy_dst_body<ROWS>(to, from); })));
}

}  // namespace gdn_decode_sfpu
