// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Word layout of the Quasar permute "block plan", shared by the host plan builder and the kernel.
//
// The op cuts the tensor into blocks. A block is a box in INPUT index space: on every input axis d it covers
// the chunk [j_d * C_d, j_d * C_d + n_d), n_d = min(C_d, S_d - j_d * C_d). One DM thread handles a block alone:
// it reads the input pages (or page segments) the box touches into its private L1 window, reorders the elements
// there, and writes the output pages (or page segments) the box produces. The chunk sizes are chosen on the host
// so that every output page (or segment) is produced by exactly one block.
//
// Each axis is described once per side (input, output) by its role there:
//   kOuter     the index selects pages:            page += index * page_stride
//   kRowElem   ROW_MAJOR in-page element index:    element offset = index
//   kTileRow   TILE row: page += (index / 32) * page_stride, element offset by the 16x16 face layout
//   kTileCol   TILE column, likewise
// and by its slot stride in the thread's L1 window (non-zero only for a kOuter axis with C_d > 1: such an axis
// enumerates the pages a block reads / writes).

#include <cstdint>

namespace ttnn::prim::qsr::permute_plan {

constexpr uint32_t kMaxRank = 8;

enum AxisKind : uint32_t { kOuter = 0, kRowElem = 1, kTileRow = 2, kTileCol = 3 };

// Header flags.
constexpr uint32_t kFlagDirect = 1u << 0;  // pages move unchanged: no element reorder, write from the read window
constexpr uint32_t kFlagInSeg = 1u << 1;   // reads fetch only the box's contiguous in-page run (else the whole page)
constexpr uint32_t kFlagOutSeg = 1u << 2;  // writes store only the box's contiguous in-page run (else the whole page)
constexpr uint32_t kFlagFill = 1u << 3;    // whole-page TILE output: pre-fill blocks that leave padding elements

// Header words.
constexpr uint32_t kRank = 0;
constexpr uint32_t kFlags = 1;
constexpr uint32_t kInSlotElems = 2;   // elements per read slot in the L1 window
constexpr uint32_t kOutSlotElems = 3;  // elements per write slot
constexpr uint32_t kInPageElems = 4;   // whole-page read length (elements)
constexpr uint32_t kOutPageElems = 5;  // whole-page write length (elements)
constexpr uint32_t kInnerAxis = 6;     // axis walked by the innermost copy loop
constexpr uint32_t kPadWord = 7;       // padding element (bit pattern in the low element bytes)
constexpr uint32_t kInBufBytes = 8;    // per-thread read window
constexpr uint32_t kOutBufBytes = 9;   // per-thread write window (0 when direct)
constexpr uint32_t kHeaderWords = 10;

// Per-axis words, axis d at kHeaderWords + d * kAxisWords.
constexpr uint32_t kAxisExtent = 0;  // S_d
constexpr uint32_t kAxisChunk = 1;   // C_d
constexpr uint32_t kAxisInKind = 2;
constexpr uint32_t kAxisInPageStride = 3;
constexpr uint32_t kAxisInSlotStride = 4;
constexpr uint32_t kAxisOutKind = 5;
constexpr uint32_t kAxisOutPageStride = 6;
constexpr uint32_t kAxisOutSlotStride = 7;
constexpr uint32_t kAxisWords = 8;

constexpr uint32_t kPlanWords = kHeaderWords + kMaxRank * kAxisWords;

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kTileElems = kTileDim * kTileDim;

}  // namespace ttnn::prim::qsr::permute_plan
