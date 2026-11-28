// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, argument slots and row geometry of the pack_fp8_rows kernel. The kernel and the host
// program factory index every slot through these names.
//
// One data-movement kernel (NoC 0). No per-core runtime args: core c = y * GRID_X + x takes
// ROWS_PER_CORE + (c < EXTRA_ROWS) rows from c * ROWS_PER_CORE + min(c, EXTRA_ROWS).
namespace pack_fp8_rows {

// Row geometry: [e4m3 values (128 B) | fp32 scale (4 B) | 60 zero B].
constexpr uint32_t VALUES = 128;        // e4m3 values (bytes) a row
constexpr uint32_t ROW_BYTES = 192;     // output row bytes
constexpr uint32_t SCALE_OFFSET = 128;  // byte offset of the fp32 scale in an output row
constexpr uint32_t SCALE_BYTES = 4;     // one fp32 scale a row
constexpr uint32_t ROWS_BLOCK = 16;     // rows staged per read / write batch (16 x 192 B = 3 KiB of L1 a core)
constexpr uint32_t STAGE_ALIGN = 64;    // the staging base is aligned up to this (Blackhole DRAM read alignment)
constexpr uint32_t STAGE_BYTES = ROWS_BLOCK * ROW_BYTES + STAGE_ALIGN;  // the staging CB's size and page

// Circular buffers: the generic op's ids, unchanged.
namespace cb {
constexpr uint32_t STAGE = 0;  // private L1 staging (never pushed)
}  // namespace cb

// The one kernel of the program.
namespace kernel_index {
enum : uint32_t { PACK, COUNT };
}  // namespace kernel_index

// Compile-time args; the e4m3, scale and out TensorAccessorArgs follow from COUNT, in that order.
namespace ct_arg {
enum : uint32_t { CB_STAGE, ROWS_BLOCK, GRID_X, COUNT };
}  // namespace ct_arg

// Common runtime args (the same on every core). The addresses are rewritten on every cache hit; the row
// split is fixed by the shapes and the grid in the cache key.
namespace common_arg {
enum : uint32_t { E4M3_ADDR, SCALE_ADDR, OUT_ADDR, ROWS_PER_CORE, EXTRA_ROWS, COUNT };
}  // namespace common_arg

}  // namespace pack_fp8_rows
