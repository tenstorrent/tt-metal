// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernels and argument slots of the block_keys program. The kernel and the host index every
// slot through these names. One data-movement kernel; no per-core runtime args: the rows are split at run time
// from the common args PER / EXTRA and the core's logical position (c = y * GX + x).
namespace block_keys {

// The one scratch CB (uint8, one page of the whole scratch size): pos row, ids row, 64-byte aligned.
namespace cb {
constexpr uint32_t SCRATCH = 0;
}  // namespace cb

// Kernel indices in the ProgramDescriptor (and the cached Program).
namespace kernel_index {
enum : uint32_t { KEYS, COUNT };
}  // namespace kernel_index

// Compile-time args. The TensorAccessorArgs of pos, ids and out follow from COUNT, in that order.
namespace ct_arg {
enum : uint32_t {
    CB,     // cb::SCRATCH (the kernel takes its CB id at this position)
    BLOCK,  // B: a power of two <= 32
    K,      // ids row length
    KP,     // pos / out row length
    NBW,    // width / B: ids >= NBW are none
    GX,     // grid columns, for the core's linear index
    COUNT
};
}  // namespace ct_arg

// Common runtime args (every core reads the same values).
namespace common_arg {
enum : uint32_t {
    POS_ADDR,
    IDS_ADDR,
    OUT_ADDR,
    PER,    // rows / ncores
    EXTRA,  // rows % ncores
    COUNT
};
}  // namespace common_arg

// The kernel aligns its scratch base and each row buffer (pos, ids) to this many bytes; the host sizes the CB from it.
constexpr uint32_t SCRATCH_ALIGN = 64;

// The kernel's sentinel for "none".
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

}  // namespace block_keys
