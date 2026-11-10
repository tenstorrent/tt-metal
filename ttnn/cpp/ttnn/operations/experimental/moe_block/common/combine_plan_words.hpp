// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The combine's device plan (the fused head's routing planner writes it, the combine's ports and accumulators read
// it): one uint32 row-major DRAM row of kPlanWords words, a single page. Shared by the host and the device kernels so
// the layout has one definition. Plain constexpr only (no includes): the kernels include it as is.
#pragma once

#include <cstdint>

namespace moe_block_plan_words {

inline constexpr uint32_t kAccMax = 32;    // accumulators at most (a home's want[] rows below)
inline constexpr uint32_t kMaxChips = 32;  // chips at most (a want[] row's columns, 32-bit chip masks)

inline constexpr uint32_t kPlanNinj = 0;   // n_inj[8]: rows a port injects, by port
inline constexpr uint32_t kPlanNret = 8;   // n_ret[8]: credits a port returns, by port
inline constexpr uint32_t kPlanNloc = 16;  // n_local[kAccMax]: an accumulator's rows from its own chip
inline constexpr uint32_t kPlanWant = 64;  // want[kAccMax][kMaxChips]: an accumulator's rows from each source chip
inline constexpr uint32_t kPlanRowsBlock = kPlanWant + kAccMax * kMaxChips;  // 16 words: the instance-rows block
inline constexpr uint32_t kPlanRows = kPlanRowsBlock + 8;                    // the rows the instances wrote
inline constexpr uint32_t kPlanOver = kPlanRowsBlock + 9;                    // 1: more rows than rows_cap
inline constexpr uint32_t kPlanWords = 2048;
inline constexpr uint32_t kPlanBytes = 4 * kPlanWords;  // the page (the whole row)
inline constexpr uint32_t kPlanHeadBytes = 4 * kPlanWant;  // words [0, kPlanWant): n_inj, n_ret, n_local

static_assert(kPlanNloc + kAccMax <= kPlanWant, "n_local[] ends before want[]");
static_assert(kPlanRowsBlock + 16 <= kPlanWords, "the rows block inside the page");

}  // namespace moe_block_plan_words
