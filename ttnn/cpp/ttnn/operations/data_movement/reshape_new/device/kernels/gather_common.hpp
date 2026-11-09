// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Layout of one stage DFB entry, shared by the gather reader (producer) and writer (consumer).
//
//   word 0          output page id of the unit
//   word 1          byte offset of the unit within that page
//   word 2          unit size in bytes
//   word 3          flags (kFlag*)
//   word 4          number of fragment records in this entry
//   word 5          reserved
//   word 6 + 3 * r  record r: {byte offset in the unit, byte offset of the staged bytes in this entry, size}
//   header_bytes..  staged fragment bytes
//
// A unit spans one or more consecutive entries: the first carries kFlagFirst, the last kFlagLast.
// The host-side constants are kGatherHeaderWords / kGatherRecordWords in reshape_new_plan.hpp.
namespace reshape_new {

constexpr uint32_t kHeaderPage = 0;
constexpr uint32_t kHeaderPageOffset = 1;
constexpr uint32_t kHeaderUnitBytes = 2;
constexpr uint32_t kHeaderFlags = 3;
constexpr uint32_t kHeaderCount = 4;
constexpr uint32_t kHeaderWords = 6;
constexpr uint32_t kRecordWords = 3;

constexpr uint32_t kFlagFirst = 1u << 0;
constexpr uint32_t kFlagLast = 1u << 1;
// Some bytes of the unit are padding, not covered by any record.
constexpr uint32_t kFlagPadded = 1u << 2;

constexpr uint32_t kNocWordBytes = 16;

}  // namespace reshape_new
