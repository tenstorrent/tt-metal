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
// Inits: each op family's init is called right before its ops, inside the acquire, because the column broadcast's
// init (mask, replay program) and the parked-scalar load (LREG14) and the exact exp's init (LREG12..14) overwrite
// each other's SFPU state; nothing is hoisted out of the per-tile chain.

#pragma once

#include <cstdint>

#include "api/compute/copy_dest_rows.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/sfpu_binary_bcast.h"

namespace gdn_decode_sfpu {

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
    ckernel::fill_rows_tile(slot_out, 0, 16, 0.0f);
    ckernel::copy_dest_values_init();
    ckernel::copy_dest_rows(slot_out, 0, slot_e, 0);
    ckernel::copy_dest_rows(slot_out, 1, slot_b, 0);
    ckernel::copy_dest_rows(slot_out, 2, slot_c, 0);
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

// DEST[to] rows 0..3 <- DEST[from] rows 0..3, bit-exact.
inline void copy_rows4(uint32_t to, uint32_t from) {
    ckernel::copy_dest_values_init();
    ckernel::copy_dest_values<DataFormat::Float32, 2>(from, to, ckernel::VectorMode::R);
}

// 16x32 tile in slot dst: rows 1..15 <- 0, row 0 kept (the form u16 needs: a NaN in a padding row of r16 or v16 would
// otherwise survive into u16).
inline void zero_rows_from_1(uint32_t dst) {
    ckernel::fill_tile_init();
    ckernel::fill_rows_tile(dst, 1, 16, 0.0f);
}

// 16x32 tile in slot dst: row 0 <- row 2, rows 1..15 <- 0 (r_q to the output row, the padding cleared).
inline void move_row2_to_row0(uint32_t dst) {
    ckernel::copy_dest_values_init();
    ckernel::copy_dest_row(dst, 0, dst, 2);
    ckernel::fill_tile_init();
    ckernel::fill_rows_tile(dst, 1, 16, 0.0f);
}

}  // namespace gdn_decode_sfpu
