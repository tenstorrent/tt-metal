// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Convenience header that includes all experimental device 2.0 APIs
// and provides short type aliases for kernel code in conv/pool operations.
// Safe to include from both dataflow and compute (TRISC) kernels.

#pragma once

#include "experimental/circular_buffer.h"

#ifndef COMPILE_FOR_TRISC
#include "experimental/noc.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"
#endif

namespace experimental {

// Short alias for CircularBuffer (available on both dataflow and compute)
using CB = CircularBuffer;

#ifndef COMPILE_FOR_TRISC
// Short aliases for NOC types (dataflow kernels only)
// Note: CoreLocalMem<uint32_t> is used internally by the read_with_state raw-address overload below.
// Prefer passing experimental::CB with offset_bytes args over constructing CoreLocalMem directly.

// Single-packet convenience wrappers for set_async_read_state / async_read_with_state.
// All conv/pool activation reads are single-packet (size <= NOC_MAX_BURST_SIZE),
// but the upstream API defaults to multi-packet mode. These wrappers put max_page_size
// first so callers don't need to spell out VcSelection::DEFAULT every time.

// Helper to create UnicastEndpoint src_args for local L1 self-reads.
// Must use my_x/my_y for the correct NOC — NOC 0 and NOC 1 have different coordinate spaces.
FORCE_INLINE auto local_addr(uint32_t addr, uint8_t noc_id = noc_index) {
    return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = addr};
}

// Single-packet read with state: call set_read_state once, then read_with_state in a loop.
// transfer_size is baked into both calls so they always agree on single-packet mode.
template <uint32_t transfer_size>
FORCE_INLINE void set_read_state(Noc noc, uint32_t src_addr) {
    static_assert(transfer_size <= NOC_MAX_BURST_SIZE, "Use noc.async_read for multi-packet transfers");
    UnicastEndpoint ep;
    noc.set_async_read_state<Noc::VcSelection::DEFAULT, transfer_size>(
        ep, transfer_size, local_addr(src_addr, noc.get_noc_id()));
}

// Simple form: local L1 src addr -> local L1 dst addr
FORCE_INLINE void read_with_state(Noc noc, uint32_t dst_addr, uint32_t src_addr) {
    UnicastEndpoint ep;
    noc.async_read_with_state<Noc::VcSelection::DEFAULT, 1>(
        ep, CoreLocalMem<uint32_t>(dst_addr), 0, local_addr(src_addr, noc.get_noc_id()), {});
}

// CB/typed destination form with dst_args (e.g. offset_bytes)
template <typename Dst>
FORCE_INLINE void read_with_state(
    Noc noc, const Dst& dst, uint32_t src_addr, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
    UnicastEndpoint ep;
    noc.async_read_with_state<Noc::VcSelection::DEFAULT, 1>(
        ep, dst, 0, local_addr(src_addr, noc.get_noc_id()), dst_args);
}

// CB/typed destination form, no offset
template <typename Dst>
FORCE_INLINE void read_with_state(Noc noc, const Dst& dst, uint32_t src_addr) {
    UnicastEndpoint ep;
    noc.async_read_with_state<Noc::VcSelection::DEFAULT, 1>(ep, dst, 0, local_addr(src_addr, noc.get_noc_id()), {});
}

#endif

}  // namespace experimental
