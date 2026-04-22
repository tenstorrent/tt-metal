// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared guard helpers for routed_matmul kernels.
//
// Each dataflow kernel (BRISC / NCRISC) independently reads max_expert_iter directly
// from DRAM into a small L1 scratch region (cb_guard), then compares against
// curr_expert_iter.  Because max_expert_iter is a read-only scalar that never changes during
// the FFN pass, no token-based synchronization between BRISC and NCRISC is
// needed — both simply read the same immutable value and reach the same
// skip/execute decision independently.
//
// TRISC (compute) cannot issue NOC reads, so its guard always returns false
// (never skip).  This is safe for the common case where max_expert_iter >= all
// curr_expert_iter values; production skip support for TRISC is a future TODO.
//
// Named compile-time args set by the program factory (via named_compile_args):
//   GUARD_CB_ID     - CB index for cb_guard (scratch for the DRAM read)
//   GUARD_ARG_BASE  - starting runtime arg index for the 2 guard args
//
// Runtime args at positions GUARD_ARG_BASE+{0,1}:
//   [0] max_expert_iter DRAM buffer address (uint32)
//   [1] curr_expert_iter scalar           (uint32)
//
// If ROUTED_GUARD_ENABLED is not defined, the helpers compile to no-ops.

#pragma once

#include <cstdint>

namespace routed_guard_detail {
constexpr uint32_t kScalarOffset = 0;  // byte offset of scalar within cb_guard
}  // namespace routed_guard_detail

#ifdef ROUTED_GUARD_ENABLED

#ifdef GUARD_COMPUTE_KERNEL

// TRISC cannot issue NOC reads — always proceed (no skip).
// TODO: implement proper TRISC guard when DRAM-read support is available.
FORCE_INLINE bool guard_check_wait() { return false; }

#else  // dataflow side (BRISC and NCRISC)

// BRISC: read max_expert_iter from DRAM into L1 scratch, compare with curr_expert_iter.
FORCE_INLINE bool guard_check_brisc() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");

    const uint32_t max_iter_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 1);

    const uint32_t scratch = get_write_ptr(kCbId) + routed_guard_detail::kScalarOffset;
    uint64_t dram_src = get_noc_addr_from_bank_id<true>(0, max_iter_addr);
    // 32-byte DRAM transaction; the uint32 scalar occupies the first 4 bytes.
    noc_async_read(dram_src, scratch, 32);
    noc_async_read_barrier();

    return curr_expert_iter > *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
}

// NCRISC: same direct DRAM read — no token wait needed, max_expert_iter is immutable.
FORCE_INLINE bool guard_check_wait() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");

    const uint32_t max_iter_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 1);

    const uint32_t scratch = get_read_ptr(kCbId) + routed_guard_detail::kScalarOffset;
    uint64_t dram_src = get_noc_addr_from_bank_id<true>(0, max_iter_addr);
    noc_async_read(dram_src, scratch, 32);
    noc_async_read_barrier();

    return curr_expert_iter > *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
}

#endif  // GUARD_COMPUTE_KERNEL

#else  // !ROUTED_GUARD_ENABLED

FORCE_INLINE bool guard_check_brisc() { return false; }
FORCE_INLINE bool guard_check_wait() { return false; }

#endif  // ROUTED_GUARD_ENABLED
