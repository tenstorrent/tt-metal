// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernels and argument slots of the block_gather kernel (block_gather.cpp). The host factory and
// the kernel index every slot through these names.
//
// One data-movement kernel source, launched twice on the same cores: HALF 0 on RISCV_1 (reader config) and HALF 1
// on RISCV_0 (writer config). No per-core runtime args: rows are split over the 2 NCORES workers inside the kernel
// from the common args.
namespace block_gather {

// The kernel's one scratch CB (uint8 pages, one page of the whole scratch size).
namespace cb {
constexpr uint32_t SCRATCH = 0;
}  // namespace cb

// Kernel index in ProgramDescriptor::kernels (and the cached Program).
namespace kernel_index {
enum : uint32_t { HALF0, HALF1, COUNT };
}  // namespace kernel_index

// Compile-time args, both kernels; the scores, ids and out TensorAccessorArgs follow from COUNT, in that order.
namespace ct_arg {
enum : uint32_t {
    CB,      // cb::SCRATCH
    BLOCK,   // B: a power of two <= 32
    K,       // block ids per row
    NBW,     // W / B: blocks per scores row
    GX,      // grid width (core index c = y * GX + x)
    PARTS,   // work units per row (4, 2 or 1)
    HALF,    // 0 / 1: this RISC's worker index within the core
    NCORES,  // GX * GY
    COUNT
};
}  // namespace ct_arg

// Common runtime args, both kernels (the same values on each).
namespace common_arg {
enum : uint32_t { SCORES_ADDR, IDS_ADDR, OUT_ADDR, ROWS, COUNT };
}  // namespace common_arg

}  // namespace block_gather
