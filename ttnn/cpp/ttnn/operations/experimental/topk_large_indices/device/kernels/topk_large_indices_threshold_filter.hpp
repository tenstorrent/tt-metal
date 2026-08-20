// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Two-pass THRESHOLD-FILTER machinery for the ROW-PARALLEL topk_large_indices
// kernels (THRESHOLD_FILTER define; llk_k == 2048, stable=False only). Shared
// by reader.cpp / compute.cpp / writer.cpp so the cross-RISC protocol lives in
// one place.
//
// PER ROW:
//   1. SAMPLE: the reader gathers 64 x 64B strided blocks (2048 bf16 samples)
//      from the row's valid prefix into one chunk-shaped CB push. Compute runs
//      the classic single-chunk pipeline on it (sorted-descending unfused
//      window at DST slot 0), and MATH reads T = the rank-(r-1) value through
//      the DST MMIO window (chunk-skip calibration). r is host-derived from
//      the runtime valid length so E[survivors] ~= 2*USER_K; P(count<USER_K)
//      is negligible for ANY distribution (distribution-free order statistic),
//      and NEVER affects correctness (see RETRY).
//   2. SCAN: per chunk at DST 6-7: unpack (existing copy), MATH ORs the stamp
//      0x8000|engine_coord into each word and zeroes lanes strictly below
//      T_fused in sign-magnitude order (SFPSWAP ALL_ROWS_MAX comparator), PACK
//      packs both tiles with packer ZERO-COMPRESSION into cb_surv_comp.
//      A survivor word is never 0x00000000 (bit15 of the stamp).
//   3. PARSE (writer BRISC): walks the compressed streams (RSI section + 32-
//      datum groups + 16B counter blocks), copies nonzero words into dense
//      survivor chunks re-stamped as [v16 | (d+1)<<11 | j], appends the TRUE
//      row-major index into the side table at slot d*2048 + rm(j), publishes
//      monotone progress words, and emits the row decision.
//   4. FINISH (compute, interleaved): fold dense chunks into the row-fused
//      survivor at DST 4-5 (sort at 6-7 + fused merge), then the existing
//      global split + epilogue at base 4. The final split emits
//      (d+1)*2048 + rm(j); the writer remaps through the side table.
//   5. RETRY (exactness backstop): survivors < USER_K or > capacity -> the
//      reader re-streams the row and everyone runs the CLASSIC body.
//
// CTRL WORDS (cb_tf_ctrl base, BRISC single writer, monotone):
//   word[0] row decision: (row+1)*4 + {1=OK | 2=RETRY}
//   word[1] cumulative dense chunks pushed (across the whole kernel run)

#pragma once

#include <cstdint>

#if defined(TRISC_MATH) || defined(TRISC_UNPACK) || defined(TRISC_PACK)
#include "api/compute/compute_kernel_api.h"
#endif

namespace topk_large_indices_threshold_filter {

constexpr uint32_t kDenseCap = 8;          // dense survivor chunks per row (side table 64 KB)
constexpr uint32_t kCompPagesPerTile = 2;  // 4 KB pages; worst case 4672 B/tile
constexpr uint32_t kRss = 4;               // Row_start_section_size, 16 B units
constexpr uint32_t kSampleBlockBytes = 64;
constexpr uint32_t kSampleBlocks = 64;  // 64 x 32 bf16 = 2048 samples
constexpr uint32_t kDecOk = 1;
constexpr uint32_t kDecRetry = 2;

// Every ctrl word carries the launch epoch in its high 16 bits: stale L1 from
// a previous launch can never satisfy a spin (consecutive launches always get
// distinct epochs), so no pre-zeroing handshake is needed.
inline uint32_t ctrl_word(uint32_t epoch, uint32_t value) { return (epoch << 16) | (value & 0xFFFFu); }
inline uint32_t ctrl_value(uint32_t epoch, uint32_t word) { return (word >> 16) == epoch ? (word & 0xFFFFu) : 0u; }
inline uint32_t decision_value(uint32_t row, uint32_t code) { return (row + 1) * 4 + code; }

// Engine coordinate (11-bit payload, K=2048) -> row-major position inside the
// chunk. Mirror of the SFPU _topk_xl_decode_row_major_index_ bit shuffle.
inline uint32_t rm2048(uint32_t c) {
    return ((c >> 6) & 0xF) | ((c & 0xF) << 4) | (((c >> 10) & 1) << 8) | (((c >> 4) & 1) << 9) |
           (((c >> 5) & 1) << 10);
}

#ifdef TRISC_MATH
#include "sfpi.h"

namespace sfpu_detail {

// Threshold mask for one K=2048 chunk based at the wrapper's dst_index.
// Words arrive as [bf16 | 0x0000]. Survivors (sign-magnitude >= T_fused,
// tie-inclusive) get bit0 set -- a +0.0 survivor is never the all-zero word --
// and losers become 0x00000000 (compressed away by the packer). No per-lane
// coordinate is materialized: index recovery happens entirely in the parser
// from the zero-compression counters. (SFPI, not raw TTI: raw SFPMOV/SFPIADD
// reads of the LTILEID constant do not materialize per-lane values on BH --
// silicon-diagnosed here and independently recorded in the moe-gate kernel.)
inline void _tf_mask_stamp_2048_(const uint32_t t_fused)
{
    using namespace ckernel;
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    // Compare the sign-extended VALUE halves only: 16-bit ranges cannot
    // overflow the 32-bit subtract the vector compare lowers to (a full-word
    // signed compare wraps for large-magnitude negatives -- silicon-diagnosed
    // as "all positives plus small negatives survive"). Comparing values only
    // is also what makes the mask tie-INCLUSIVE regardless of low bits.
    const bool t_neg = (t_fused & 0x80000000u) != 0;
    const sfpi::vInt vt = static_cast<int32_t>(t_fused) >> 16;
    // Branchless select (no CC churn): a loser mask m (all-ones) is built
    // from sign bits of 16-bit-range subtractions (overflow-free), and the
    // stamped word is ANDed with ~m. Survivor low half = 0x8000 |
    // word-position-in-tile: rows k and k+1 interleave in DST word order
    // (word w <-> row 2*(w>>6) + (w&1), lane (w>>1)&31; silicon-calibrated),
    // and vConstTileId = 2*lane, so position = vConstTileId | rowscalar
    // (disjoint bits, OR == ADD). vConstTileId is only safe in SHIFT/OR
    // expressions (adds/moves of the LTILEID constant zero the whole row --
    // the moe-gate kernel's silicon note, re-confirmed here).
    if (!t_neg) {
        // T >= +0: survivor iff v16 >= vt (signed == sign-magnitude here;
        // negatives drop automatically). loser m = sign(v16 - vt).
        for (int k = 0; k < 64; ++k) {
            sfpi::vInt w = sfpi::dst_reg[k];
            sfpi::vInt v16 = w >> 16;
            sfpi::vInt stamp = sfpi::vConstTileId | (0x8000 | (((k >> 1) << 6) | (k & 1)));
            sfpi::vInt m = (v16 - vt) >> 31;
            sfpi::dst_reg[k] = ((v16 << 16) | stamp) & ~m;
        }
    } else {
        // T < 0: loser iff v16 < 0 AND v16 > vt (larger-magnitude negative);
        // m = sign(v16) & sign(vt - v16).
        // OPEN BUG: this branch leaks ~0.02% of survivors at specific
        // magnitudes on silicon (2-3 per row on all-negative rows); the
        // compute currently declines the filter for negative T (forces the
        // retry-classic path), so this branch never runs in production.
        for (int k = 0; k < 64; ++k) {
            sfpi::vInt w = sfpi::dst_reg[k];
            sfpi::vInt v16 = w >> 16;
            sfpi::vInt stamp = sfpi::vConstTileId | (0x8000 | (((k >> 1) << 6) | (k & 1)));
            sfpi::vInt m = (v16 >> 31) & ((vt - v16) >> 31);
            sfpi::dst_reg[k] = ((v16 << 16) | stamp) & ~m;
        }
    }
}

}  // namespace sfpu_detail
#endif  // TRISC_MATH

// Mask+stamp the chunk at DST [idst, idst+1] (K=2048). t_fused is the
// tie-inclusive fused threshold word [t16 | sign-extension] (MATH-only value;
// other TRISCs pass anything).
inline void tf_mask_stamp(uint32_t idst, uint32_t t_fused) {
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(sfpu_detail::_tf_mask_stamp_2048_, idst, VectorMode::RC_custom, t_fused);
#else
    (void)idst;
    (void)t_fused;
#endif
}

#ifdef TRISC_PACK
// Toggle packer zero-compression (silicon-validated by the zc probe:
// SEC0_REG1 alone controls all four coalesced interfaces on BH; RSS shifts
// the data stream so the RSI section lands at the slot base).
inline void tf_zc_set(bool en) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
    ckernel::cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Row_start_section_size_ADDR32,
        THCON_SEC0_REG1_Row_start_section_size_SHAMT,
        THCON_SEC0_REG1_Row_start_section_size_MASK>(en ? kRss : 0);
    ckernel::cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Disable_zero_compress_ADDR32,
        THCON_SEC0_REG1_Disable_zero_compress_SHAMT,
        THCON_SEC0_REG1_Disable_zero_compress_MASK>(en ? 0u : 1u);
}
#endif  // TRISC_PACK

}  // namespace topk_large_indices_threshold_filter
