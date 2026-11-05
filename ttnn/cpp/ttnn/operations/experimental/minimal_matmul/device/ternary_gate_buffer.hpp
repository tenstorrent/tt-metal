// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::experimental::prim {

// L1 left free, beyond the program's own buffers, before a full-shape gate may take a whole-block buffer. Covers
// what the static estimate below cannot see: L1-resident tensors and the L1_SMALL region.
inline constexpr uint64_t kTernaryGateFullBlockL1Reserve = 128 * 1024;

// Whether a full-shape [M, N] addcmul gate (ternary_b) gets a buffer holding the whole output block instead of one
// row of it. Shared by minimal_matmul and all_gather_minimal_matmul_async, whose readers fetch the gate row by row
// after an output block's K loop and whose compute pops it row by row.
//
// With a one-row buffer the reader can only fetch row r + 1 once compute has popped row r, so compute waits on a
// DRAM round trip per row, and the reader - which also prefetches the next block's in0 - stalls behind it. A
// whole-block buffer lets the reader fetch the entire gate while compute is still in the K loop. It costs
// (M_block - 1) * N_block gate tiles of L1, so it is taken only when `buffer_bytes_with_full_block` (every buffer
// the program allocates per core, with the gate at full size) fits the static L1 space with the reserve to spare.
//
// The decision is a pure function of the op's attributes, its tensor specs and the device, all of which the program
// cache already keys on, so it is part of the program hash without a separate field. It deliberately does not read
// live L1 occupancy, which the cache does not key on.
inline bool use_full_block_ternary_gate(
    tt::tt_metal::IDevice* device, bool gate_is_full_shape, uint64_t buffer_bytes_with_full_block) {
    if (!gate_is_full_shape) {
        return false;
    }
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t ceiling = device->l1_size_per_core();
    return ceiling > base + kTernaryGateFullBlockL1Reserve &&
           buffer_bytes_with_full_block <= ceiling - base - kTernaryGateFullBlockL1Reserve;
}

}  // namespace ttnn::experimental::prim
