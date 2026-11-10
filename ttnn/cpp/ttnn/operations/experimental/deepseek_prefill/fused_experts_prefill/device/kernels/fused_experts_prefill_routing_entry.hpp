// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Layout of one routing entry in fused_experts_prefill's cb_meta token lists, shared by the reader (which
// packs it) and the writer (which unpacks it) so the two cannot drift:
//
//   bits [ 0, 11)  token index in the chunk   (kMaxTokens = 2048)
//   bits [11, 16)  slot, i.e. which of the token's top_k experts this is   (kMaxTopK = 16, 4 bits used)
//   bits [16, 32)  the token's routing weight, bf16
//
// The field widths are why kMaxTokens is a power of two: the token mask is derived from the width, not
// written out, so raising the token cap only means changing kTokenBits here (plus the L1 it costs in
// cb_meta, and re-deriving kMaxTokens where the op validates the shape).
namespace fused_experts_prefill_routing {

inline constexpr uint32_t kTokenBits = 11;
inline constexpr uint32_t kSlotBits = 4;
inline constexpr uint32_t kTokenMask = (1u << kTokenBits) - 1u;
inline constexpr uint32_t kSlotMask = (1u << kSlotBits) - 1u;
inline constexpr uint32_t kSlotShift = kTokenBits;
inline constexpr uint32_t kWeightShift = 16;
inline constexpr uint32_t kWeightMask = 0xFFFF0000u;

// Pack one (token, slot) pair with its pre-bf16-encoded weight (the caller keeps the weight's 16 bits).
inline uint32_t pack(uint32_t token, uint32_t slot, uint32_t weight_bits) {
    return (token & kTokenMask) | ((slot & kSlotMask) << kSlotShift) | (weight_bits & kWeightMask);
}

inline uint32_t token_of(uint32_t entry) { return entry & kTokenMask; }
inline uint32_t slot_of(uint32_t entry) { return (entry >> kSlotShift) & kSlotMask; }
inline uint32_t weight_bits_of(uint32_t entry) { return entry & kWeightMask; }

}  // namespace fused_experts_prefill_routing
