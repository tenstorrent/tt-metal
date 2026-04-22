// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared guard helpers for routed_matmul kernels.
//
// Contract:
//   - BRISC (in0 reader) calls guard_check_brisc() at kernel entry.
//     It reads the max_iter scalar from DRAM into cb_guard[0..3], writes the
//     guard_token from its runtime args to cb_guard[32..35], and returns whether
//     this core should skip (expert_iter > max_iter).
//   - NCRISC (in1 reader/writer) and TRISC (compute) call guard_check_wait().
//     They spin until cb_guard[32..35] == their own guard_token runtime arg,
//     then read cb_guard[0..3] and compare against expert_iter.
//
// Token scheme avoids the semaphore-reset-on-program-cache-hit problem: each
// host-side dispatch passes a unique guard_token (monotonic counter), so stale
// L1 from the previous dispatch can never coincidentally satisfy the wait.
//
// cb_guard layout (64 bytes, one page):
//   [0..3]   max_iter scalar (uint32)   — BRISC writes via DRAM read
//   [32..35] guard_token (uint32)       — BRISC writes after DRAM barrier
//
// Compile-time defines (set by the program factory, per-kernel):
//   GUARD_CB_ID     - CB index for cb_guard
//   GUARD_ARG_BASE  - starting runtime arg index for the 3 guard args
//
// Runtime args at positions GUARD_ARG_BASE+{0,1,2}:
//   [0] max_iter DRAM buffer address
//   [1] expert_iter scalar
//   [2] guard_token
//
// If ROUTED_GUARD_ENABLED is not defined, the helpers compile to no-ops that
// always return false (never skip). Lets the factory bring routed_matmul up
// without the guard wiring first, then enable it by defining the flag.

#pragma once

#include <cstdint>

namespace routed_guard_detail {

constexpr uint32_t kScalarOffset = 0;
constexpr uint32_t kTokenOffset = 32;

}  // namespace routed_guard_detail

#ifdef ROUTED_GUARD_ENABLED

#ifdef GUARD_COMPUTE_KERNEL

// TRISC-side: wait for BRISC token, then compare.
FORCE_INLINE bool guard_check_wait() {
    const uint32_t expert_iter = get_arg_val<uint32_t>(GUARD_ARG_BASE + 1);
    const uint32_t guard_token = get_arg_val<uint32_t>(GUARD_ARG_BASE + 2);

    const uint32_t base_addr = get_read_ptr(GUARD_CB_ID);
    volatile tt_l1_ptr uint32_t* token_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr + routed_guard_detail::kTokenOffset);
    while (*token_ptr != guard_token) {
    }
    volatile tt_l1_ptr uint32_t* scalar_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr + routed_guard_detail::kScalarOffset);
    return expert_iter > *scalar_ptr;
}

#else  // dataflow side

FORCE_INLINE bool guard_check_brisc() {
    const uint32_t max_iter_addr = get_arg_val<uint32_t>(GUARD_ARG_BASE);
    const uint32_t expert_iter = get_arg_val<uint32_t>(GUARD_ARG_BASE + 1);
    const uint32_t guard_token = get_arg_val<uint32_t>(GUARD_ARG_BASE + 2);

    const uint32_t base_addr = get_write_ptr(GUARD_CB_ID);
    const uint32_t scratch_addr = base_addr + routed_guard_detail::kScalarOffset;
    uint64_t dram_src = get_noc_addr_from_bank_id<true>(0, max_iter_addr);
    // 32-byte DRAM transaction; the uint32 scalar occupies the first 4 bytes.
    noc_async_read(dram_src, scratch_addr, 32);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* token_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr + routed_guard_detail::kTokenOffset);
    *token_ptr = guard_token;

    volatile tt_l1_ptr uint32_t* scalar_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    return expert_iter > *scalar_ptr;
}

// NCRISC: same wait-and-compare as TRISC. BRISC on the same core owns the DRAM read.
FORCE_INLINE bool guard_check_wait() {
    const uint32_t expert_iter = get_arg_val<uint32_t>(GUARD_ARG_BASE + 1);
    const uint32_t guard_token = get_arg_val<uint32_t>(GUARD_ARG_BASE + 2);

    const uint32_t base_addr = get_read_ptr(GUARD_CB_ID);
    volatile tt_l1_ptr uint32_t* token_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr + routed_guard_detail::kTokenOffset);
    while (*token_ptr != guard_token) {
    }
    volatile tt_l1_ptr uint32_t* scalar_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_addr + routed_guard_detail::kScalarOffset);
    return expert_iter > *scalar_ptr;
}

#endif  // GUARD_COMPUTE_KERNEL

#else  // !ROUTED_GUARD_ENABLED

FORCE_INLINE bool guard_check_brisc() { return false; }
FORCE_INLINE bool guard_check_wait() { return false; }

#endif  // ROUTED_GUARD_ENABLED
