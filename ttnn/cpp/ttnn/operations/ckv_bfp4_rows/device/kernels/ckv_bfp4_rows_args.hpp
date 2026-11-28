// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernel indices and argument slots of the ckv_bfp4_rows program. The host factory and the
// kernels (ckv_bfp4_rows_reader.cpp, ckv_bfp4_writer.cpp with CKV_ROWS_IN, ckv_bfp4_compute.cpp with CKV_ROWS_IN)
// index every slot through these names.
//
// Every runtime arg is a common arg (no per-core args): the cores split the bands themselves (ckv_share.hpp, with
// the defines CKV_ROWS_GX = grid width and CKV_ROWS_NC = grid cores).
namespace ckv_bfp4_rows {

// Geometry of a row record: 512 bf16 values -> 320 B (32 exponents twice, then 256 nibble bytes).
constexpr uint32_t WIDTH = 512;
constexpr uint32_t ROW_BYTES = 320;
constexpr uint32_t SRC_ROW_BYTES = WIDTH * 2;  // one bf16 source row a page

// The ids the generic op used (the kernels' values): dense roles, unchanged.
namespace cb {
constexpr uint32_t IN = 0;        // a band's 32 rows row-major, 16 tile-sized pages, double-buffered
constexpr uint32_t ROWS = 16;     // the packer's 32 records of a band (one page of 32 * ROW_BYTES), 2 pages
constexpr uint32_t ROUNDED = 17;  // passed to writer / compute as an id only; never allocated in this program
constexpr uint32_t TILED = 18;    // the band tilized from cb IN, 16 bf16 tiles
}  // namespace cb

// Kernel order in the ProgramDescriptor (= kernel handle in the cached Program).
namespace kernel_index {
enum : uint32_t { READER, WRITER, COMPUTE, COUNT };
}  // namespace kernel_index

// Reader compile-time args; the src TensorAccessorArgs follow at COUNT.
namespace reader_ct {
enum : uint32_t { ROW_BYTES, COUNT };
}  // namespace reader_ct

// Writer compile-time args (ckv_bfp4_writer.cpp); the rows TensorAccessorArgs follow at COUNT.
namespace writer_ct {
enum : uint32_t { CB_ROWS, CB_ROUNDED, TILES_PER_BAND, ROW_BYTES, COUNT };
}  // namespace writer_ct

// Compute compile-time args (ckv_bfp4_compute.cpp).
namespace compute_ct {
enum : uint32_t { CB_IN, CB_ROWS, CB_ROUNDED, CB_TILED, COUNT };
}  // namespace compute_ct

// Reader common args: src address (Buffer*), first source row and band count (per-call values).
namespace reader_common {
enum : uint32_t { SRC_ADDR, SRC_ROW0, NUM_BANDS, COUNT };
}  // namespace reader_common

// Writer common args: rows address (Buffer*), band count and first record row (per-call values).
namespace writer_common {
enum : uint32_t { ROWS_ADDR, NUM_BANDS, ROW0, COUNT };
}  // namespace writer_common

// Compute common args: band count (per-call value).
namespace compute_common {
enum : uint32_t { NUM_BANDS, COUNT };
}  // namespace compute_common

}  // namespace ckv_bfp4_rows
