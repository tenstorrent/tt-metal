// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernels and argument slots of the unpack_fp8_rows kernels, shared by the kernels and the
// program factory. Values as the Python generic op used them.
namespace unpack_fp8_rows {

// A 32-row band: 4 e4m3 tiles, 1 fp32 scale tile, 4 bfloat8_b out tiles.
constexpr uint32_t ROW_BYTES = 192;   // one row a page of rows
constexpr uint32_t VALUES = 128;      // out's last dim
constexpr uint32_t STAGE_ALIGN = 64;  // the reader aligns its stage up to this

namespace cb {
constexpr uint32_t E = 0;       // FP8_E4M3 tiles, 8 pages (double-buffered band)
constexpr uint32_t S = 1;       // FLOAT32 scale tile, 2 pages; UnpackToDestFp32
constexpr uint32_t OUT = 2;     // BFLOAT8_B tiles, 8 pages
constexpr uint32_t RSTAGE = 3;  // uint8, one page of 32 * ROW_BYTES + STAGE_ALIGN bytes (reader scratch)
}  // namespace cb

constexpr uint32_t E_PAGES = 8;
constexpr uint32_t S_PAGES = 2;
constexpr uint32_t OUT_PAGES = 8;

// Kernel order in the descriptor (a kernel's handle on a hit).
namespace kernel_index {
enum : uint32_t { READER, WRITER, COMPUTE, COUNT };
}  // namespace kernel_index

// Compile-time args. The CB ids sit at these positions (the kernels take them there); the readers' and the
// writer's TensorAccessorArgs follow COUNT.
namespace reader_ct {
enum : uint32_t { CB_E, CB_S, CB_STAGE, GX, NCORES, COUNT };
}  // namespace reader_ct
namespace writer_ct {
enum : uint32_t { CB_OUT, GX, NCORES, COUNT };
}  // namespace writer_ct
namespace compute_ct {
enum : uint32_t { CB_E, CB_S, CB_OUT, GX, NCORES, COUNT };
}  // namespace compute_ct

// Per-core runtime args (without UNPACK_COMMON_VALUES), the same values on every core.
namespace reader_rt {
enum : uint32_t { SRC_PAGE0, NBANDS, COUNT };
}  // namespace reader_rt
namespace writer_rt {
enum : uint32_t { OUT_PAGE0, NBANDS, COUNT };
}  // namespace writer_rt
namespace compute_rt {
enum : uint32_t { NBANDS, COUNT };
}  // namespace compute_rt

// Common runtime args. SRC_PAGE0 / OUT_PAGE0 / NBANDS only with UNPACK_COMMON_VALUES; the address always.
namespace reader_common {
enum : uint32_t { ROWS_ADDR, SRC_PAGE0, NBANDS, COUNT };
constexpr uint32_t COUNT_ADDR_ONLY = SRC_PAGE0;
}  // namespace reader_common
namespace writer_common {
enum : uint32_t { OUT_ADDR, OUT_PAGE0, NBANDS, COUNT };
constexpr uint32_t COUNT_ADDR_ONLY = OUT_PAGE0;
}  // namespace writer_common
namespace compute_common {
enum : uint32_t { NBANDS, COUNT };  // none without UNPACK_COMMON_VALUES
}  // namespace compute_common

}  // namespace unpack_fp8_rows
