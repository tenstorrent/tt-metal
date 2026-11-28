// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffer, kernel and argument slots of the block_max_bf16 kernels (block_max_bf16.cpp, the reduce variant,
// and block_max_strip.cpp, the strip_major variant). The host and the kernels index every slot through these names.
//
// Each variant runs one data-movement source twice on every core of the grid: kernel READER (NCRISC, HALF 0) and
// kernel WRITER (BRISC, HALF 1). Work is split over the 2 x NCORES RISCs at run time, so there are no per-core
// runtime args: everything per call is a common runtime arg.
namespace block_max_bf16 {

// The one scratch CB (uint8, one page of the whole scratch size), the generic op's id.
namespace cb {
constexpr uint32_t SCRATCH = 0;
}  // namespace cb

// Kernel order in the descriptor (and so the kernel handle in the cached Program), both variants.
namespace kernel_index {
enum : uint32_t { READER, WRITER, COUNT };
}  // namespace kernel_index

// The value of the HALF compile-time arg of each kernel.
namespace half {
constexpr uint32_t READER = 0;
constexpr uint32_t WRITER = 1;
}  // namespace half

// block_max_bf16.cpp compile-time args; TensorAccessorArgs of in, out, len (out again when there is no n_tensor)
// follow from COUNT.
namespace reduce_ct {
enum : uint32_t { CB, FACTOR, NI, GX, NCORES, CHUNK, HAS_LEN, HALF, COUNT };
}  // namespace reduce_ct

// block_max_strip.cpp compile-time args; TensorAccessorArgs of in, out, len (out again when there is no n_tensor)
// follow from COUNT.
namespace strip_ct {
enum : uint32_t { CB, NB, GX, NCORES, CHUNK_BLOCKS, HAS_LEN, HALF, COUNT };
}  // namespace strip_ct

// Common runtime args, the same on both kernels of both variants. LEN_ADDR is 0 without n_tensor; N is 0 with one
// (the kernel reads n from the tensor). ROWS is rows (reduce) or strips (strip_major), fixed by the shapes.
namespace common_arg {
enum : uint32_t { IN_ADDR, OUT_ADDR, LEN_ADDR, N, ROWS, COUNT };
}  // namespace common_arg

// Input values one reduce work unit holds (8 KiB of L1).
constexpr uint32_t CAST_CHUNK = 2048;
// Blocks one strip_major work unit holds.
constexpr uint32_t STRIP_CHUNK = 128;
// Alignment of the kernels' scratch buffers inside the CB (bytes), and the bytes of one block's 32 bf16 values.
constexpr uint32_t SCRATCH_ALIGN = 64;
constexpr uint32_t STRIP_RUN_BYTES = 64;

}  // namespace block_max_bf16
