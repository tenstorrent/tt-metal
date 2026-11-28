// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernel indices and argument slots of the pair_pool kernels, shared by the kernels and the
// program factory. No kernel has per-core runtime args: every core derives its share of the units from the
// common args PER / EXTRA and the define PAIR_POOL_GX (pair_pool_common.hpp).
namespace pair_pool {

// The generic op's ids, kept as they were. The kernels take them as compile-time args at the positions below.
namespace cb {
constexpr uint32_t STAGE = 0;  // reader's private staging (4 tiles, never pushed)
constexpr uint32_t IN = 1;     // shuffled [kv even | gate even | kv odd | gate odd], double-buffered (8 tiles)
constexpr uint32_t OUT = 16;   // pooled tile (2 tiles)
constexpr uint32_t STAGE_TILES = 4;
constexpr uint32_t IN_TILES = 8;
constexpr uint32_t OUT_TILES = 2;
}  // namespace cb

// Order of ProgramDescriptor::kernels.
namespace kernel_index {
enum : uint32_t { READER, WRITER, COMPUTE, COUNT };
}  // namespace kernel_index

// Reader compile-time args; the input's TensorAccessorArgs follow from COUNT.
namespace reader_ct {
enum : uint32_t { CB_STAGE, CB_IN, HT, COUNT };
}  // namespace reader_ct

// Reader common runtime args.
namespace reader_common {
enum : uint32_t { SRC_ADDR, PER, EXTRA, COUNT };
}  // namespace reader_common

// Writer compile-time args; the output's TensorAccessorArgs follow from COUNT.
namespace writer_ct {
enum : uint32_t { CB_OUT, HT, COUNT };
}  // namespace writer_ct

// Writer common runtime args.
namespace writer_common {
enum : uint32_t { OUT_ADDR, PER, EXTRA, COUNT };
}  // namespace writer_common

// Compute compile-time args.
namespace compute_ct {
enum : uint32_t { CB_IN, CB_OUT, COUNT };
}  // namespace compute_ct

// Compute common runtime args.
namespace compute_common {
enum : uint32_t { PER, EXTRA, COUNT };
}  // namespace compute_common

}  // namespace pair_pool
