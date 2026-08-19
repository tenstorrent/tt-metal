// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SFPU kernel: per-bank index filter + encode for SparseKDramReader.
//
// The index list is BANK-ADDRESSED: each uint32 already carries the token's
// (global bank, within-bank position) directly — the indexer stamps it that way,
// and its round-robin granularity is aligned to the main KV tokens_per_page — so
// there is NO round-robin reconstruction here:
//
//   bits [0, WITHIN_BANK_BITS)                within-bank position  (-> local)
//   bits [GLOBAL_BANK_SHIFT, +6-bit field)    global bank id (local_bank | device<<3)
//
// Each DST lane holds one uint32 index. Keep only tokens whose global-bank field
// matches this core's bank, and encode each match as local + 1:
//
//   local = idx & WITHIN_BANK_MASK
//   enc   = ((idx >> GLOBAL_BANK_SHIFT) & BANK_MASK) == MY_BANK
//             ? ((local + 1) << OUT_SHIFT) : 0
//
// The +1 keeps a real match at within-bank position 0 from colliding with the
// non-match sentinel 0. DM0 recovers local = enc - 1, then decodes the byte
// address with tokens_per_page (in_bank_page = local >> log_tpp, offset =
// local & tpp_mask) — that decode is unchanged; only the bank/local extraction
// here is a direct field read (no round-robin reconstruction).
//
// BANK_MASK         = num_global_banks - 1  (the 6-bit global-bank mask, 0x3F for 64 banks)
// MY_BANK           = this core's global_bank_id (device*num_local_banks + local_bank)
// GLOBAL_BANK_SHIFT = bit offset of the global-bank field (distributed_indexer config: 14)
// WITHIN_BANK_MASK  = (1 << WITHIN_BANK_BITS) - 1 (config: 0x3FFF)
// OUT_SHIFT         = 0 or 16 — packs two tiles two-per-uint32 (one in the low 16
//                     bits, one in the high 16) before OR-ing the dest tiles. enc
//                     fits 16 bits (op.py asserts), so the halves don't overlap.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// ITERATIONS rows of the current DST tile (32 lanes each); pass 32 for a full
// 32x32 tile with VectorMode::RC_custom.
//
// Requires a 32-bit integer DST (op runs with fp32_dest_acc_en, indices CB
// unpacked in UnpackToDestFp32), so `vInt idx = dst_reg[0]` loads the raw
// int32 page_id — the idiom the stock integer SFPU ops use.
template <
    int ITERATIONS,
    std::uint32_t BANK_MASK,
    std::uint32_t MY_BANK,
    std::uint32_t GLOBAL_BANK_SHIFT,
    std::uint32_t WITHIN_BANK_MASK,
    std::uint32_t OUT_SHIFT = 0>
inline void _sparse_k_filter_tile_()
{
    using namespace sfpi;
    // Bank-addressed index: bits [0, WITHIN_BANK_BITS) are the within-bank slot
    // (== DM0 `local`, decoded to page/offset via tokens_per_page), and the
    // 6-bit global-bank field sits at GLOBAL_BANK_SHIFT (local_bank | device<<3).
    // Test the bank field in place (constants shifted at compile time) so the
    // hot predicate stays a single AND + compare — no per-lane shift of idx.
    constexpr int BANK_FIELD = static_cast<int>(BANK_MASK) << static_cast<int>(GLOBAL_BANK_SHIFT);
    constexpr int MY_FIELD   = static_cast<int>(MY_BANK) << static_cast<int>(GLOBAL_BANK_SHIFT);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vInt idx = dst_reg[0];
        vInt enc = 0;
        v_if ((idx & BANK_FIELD) == MY_FIELD)
        {
            vInt local = idx & static_cast<int>(WITHIN_BANK_MASK);
            enc        = (local + 1) << static_cast<int>(OUT_SHIFT);
        }
        v_endif;
        dst_reg[0] = enc;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
