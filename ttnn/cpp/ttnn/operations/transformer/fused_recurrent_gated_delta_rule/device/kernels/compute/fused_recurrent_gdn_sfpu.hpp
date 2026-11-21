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
