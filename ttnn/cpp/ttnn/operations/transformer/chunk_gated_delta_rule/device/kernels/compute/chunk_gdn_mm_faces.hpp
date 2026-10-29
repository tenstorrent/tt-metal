// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gdn scan only: one 32x32 fp32 tile matmul whose in0 has known all-zero faces, issuing only the MVMULs of the
// non-zero face products.
//
// A full-tile matmul is 8 face products (out face (i,j) = B(i,0) A(0,j) + B(i,1) A(1,j), B = in0 in SrcB, A = in1 in
// SrcA), two 8-row MVMULs each, repeated per fidelity phase: 16 MVMULs x 4 phases at HiFi4. The scan's in0 operands
// have structure the matmul does not know about:
//   Diag  : in0 faces 1 and 2 are zero (dl*I, the identity): out = B0 A0 | B0 A1 | B3 A2 | B3 A3, 8 MVMULs per phase.
//   Lower : in0 face 1 is zero (lower-triangular T_inv, intra): the full sequence without B1 A2 / B1 A3, 12 MVMULs.
// The skipped products are exact zeros added into DST, so the result is the full matmul's, value for value, with the
// same per-tile accumulation order. The unpack side is the ordinary matmul unpack (llk_unpack_AB_matmul).
//
// Replay buffer / address modes (MATH thread). matmul_init records the full-tile sequence in slots 16..31 (the FPU
// half, ckernel::math::replay_buf_offset) and programs ADDR_MOD_0/1/2/4/5. The Diag sequence goes to slots 0..7 (the
// SFPU half: the scan runs no SFPU op) and the two extra jumps to ADDR_MOD_3 / ADDR_MOD_6, which the unthrottled matmul
// does not use. Lower reuses the recorded full sequence: slots 16..22 (steps 1-7), one MVMUL with its own jump, then
// slots 28..31 (steps 13-16).

#pragma once

#include <cstdint>
#include "api/compute/matmul.h"

namespace gdn_mmf {
enum class In0 : uint32_t { Diag, Lower };
#ifdef TRISC_MATH
inline constexpr bool kHiFi = MATH_FIDELITY != ckernel::MathFidelity::LoFi;
inline constexpr uint32_t kPhases = kHiFi ? static_cast<uint32_t>(MATH_FIDELITY) : 1;  // HiFi2/3/4 = 2/3/4 phases
inline constexpr uint32_t kDiagReplay = 0;                                             // 8 slots
inline constexpr uint32_t kFullReplay = ckernel::math::replay_buf_offset;
#endif
}  // namespace gdn_mmf

// Call after matmul_init (any time before the first gdn_matmul_tiles_faces of a block); its state survives later
// matmul_inits.
ALWI void gdn_mm_faces_init() {
#ifdef TRISC_MATH
    if constexpr (gdn_mmf::kHiFi) {
        using namespace ckernel;
        // Diag, after B0 A1's second half (srca 16, srcb 8, dest 24, CR_a = CR_b = 0): to B3 A2 (srca 32, srcb 48,
        // dest 32).
        addr_mod_t{
            .srca = {.incr = 32, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_3);
        // Lower, after B2 A1's second half (step 8 of the full sequence; CR_a = 0, CR_b = 32, CR_d = 0): to B3 A2
        // (srca 32, srcb 48, dest 32), where the full sequence's step 13 starts.
        addr_mod_t{
            .srca = {.incr = 32, .clr = 0, .cr = 1},
            .srcb = {.incr = 16, .clr = 0, .cr = 1},
            .dest = {.incr = 32, .clr = 0, .cr = 1},
        }
            .set(ADDR_MOD_6);
        load_replay_buf(gdn_mmf::kDiagReplay, 8, [] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0 A0 rows 0-7    srcb+=8, dest+=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B0 A0 rows 8-15   srca=16, srcb=0, dest=16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0 A1 rows 0-7
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);  // B0 A1 rows 8-15   srca=32, srcb=48, dest=32
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B3 A2 rows 0-7
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B3 A2 rows 8-15   srca=48, srcb=48, dest=48
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B3 A3 rows 0-7
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);  // B3 A3 rows 8-15   reset, next fidelity phase
        });
    }
#endif
}

// DST[idst] += in0[i0] @ in1[i1] for an in0 tile with KIND's zero faces (see the top of the file). Same operand
// mapping and requirements as matmul_tiles (matmul_init on these CBs, then gdn_mm_faces_init).
template <gdn_mmf::In0 KIND>
ALWI void gdn_matmul_tiles_faces(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
#ifdef TRISC_MATH
    if constexpr (gdn_mmf::kHiFi) {
        using namespace ckernel;
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(idst);
#pragma GCC unroll 8
        for (uint32_t ph = 0; ph < gdn_mmf::kPhases; ph++) {
            if constexpr (KIND == gdn_mmf::In0::Diag) {
                lltt::replay(gdn_mmf::kDiagReplay, 8);
            } else {
                lltt::replay(gdn_mmf::kFullReplay, 7);            // steps 1-7: B0 A0, B0 A1, B2 A0, B2 A1 (first half)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_6, 0);  // step 8: B2 A1 rows 8-15, jump to B3 A2
                lltt::replay(gdn_mmf::kFullReplay + 12, 4);       // steps 13-16: B3 A2, B3 A3, next phase
            }
        }
        // As the matmul MOP's end op (reuse A) and _llk_math_matmul_'s end of the reuse row: release both sources.
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    } else {
        llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst);
    }
#endif
}
