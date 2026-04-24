// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared guard helpers for routed_matmul kernels.
//
// Each dataflow kernel independently reads two DRAM row-major uint32 tables
// (global_expert_idx_table, expert_token_counts) into a small L1 scratch region
// (cb_guard). The skip predicate is:
//
//   global_idx = global_expert_idx_table[local_expert_idx]
//   skip       = expert_token_counts[global_idx] <= curr_expert_iter * expert_iter_length
//
// Because both tensors are read-only for the duration of the FFN pass, no
// token-based synchronization between the two dataflow threads is needed —
// both read the same immutable values and reach the same decision independently.
//
// TRISC (compute) cannot issue NOC reads, so it cannot re-read DRAM itself.
// Instead, the BRISC-side dataflow kernel publishes the skip decision to the
// three TRISC thread mailboxes (Unpack/Math/Pack) inside guard_check_wait(),
// and TRISC pops the flag with the hardware-blocking mailbox_read inside its
// own guard_check_wait(). mailbox_read stalls automatically until BRISC has
// written, so no extra synchronization is needed. NCRISC cannot link
// ckernel::mailbox_base and therefore does not participate in the BRISC→TRISC
// handoff; its decision (identical to BRISC's by construction) is consumed
// only by its own kernel's early-return.
//
// Function names vs. processors (mapping set by the program factory):
//   guard_check_brisc() — called from in0_* readers, compiled onto RISCV_1 (NCRISC).
//   guard_check_wait()  — called from in1_* readers, compiled onto RISCV_0 (BRISC).
//                       — also called from bmm_routed (compute/TRISC) via GUARD_COMPUTE_KERNEL.
// The historical names predate the BRISC⇄NCRISC swap in the factory; the
// implementations here reflect the actual processor assignment.
//
// Named compile-time args set by the program factory (via named_compile_args):
//   GUARD_CB_ID     - CB index for cb_guard (scratch for the DRAM reads; dataflow only).
//                     cb_guard is 64 bytes — partitioned into two 32-byte halves, one
//                     per DRAM read. Using separate halves is required: reusing the
//                     same scratch between two sequential noc_async_read + barrier
//                     pairs deadlocks (observed empirically; likely a NoC/barrier
//                     ordering issue). 32 bytes = 8 uint32s per half, so indices
//                     0..7 into each table are addressable with a single NoC read.
//   GUARD_ARG_BASE  - starting runtime arg index for the 5 guard args (dataflow only).
//
// Runtime args at positions GUARD_ARG_BASE+{0..4} (dataflow only):
//   [0] global_expert_idx_table DRAM buffer address (uint32)
//   [1] expert_token_counts     DRAM buffer address (uint32)
//   [2] local_expert_idx                           (uint32)
//   [3] curr_expert_iter                           (uint32)
//   [4] expert_iter_length                         (uint32)
//
// Table layout: both DRAM tensors are ROW_MAJOR_LAYOUT uint32.
//
// TODO — per-index reads: the current implementation does a naive
// get_noc_addr_from_bank_id<true>(0, buffer_addr) + 32-byte NoC read, which
// only retrieves whatever element(s) of a DRAM-INTERLEAVED buffer happen to
// map to bank 0's first page. To read an arbitrary index i correctly, the
// kernel must use InterleavedAddrGen<true> with the buffer's page size (or
// a non-interleaved memory config on the host side). The stub callers today
// work around this by filling each table uniformly so every bank-0 page has
// the expected value. The ROW_MAJOR_LAYOUT choice keeps the page = uint32
// contract clean for that upgrade.
//
// If ROUTED_GUARD_ENABLED is not defined, the helpers compile to no-ops.

#pragma once

#include <cstdint>

namespace routed_guard_detail {
constexpr uint32_t kScalarOffset = 0;  // byte offset of scratch region within cb_guard
// 32-byte DRAM transaction (8 uint32 values). Stub limitation: only indices 0..7
// are addressable. This matches the pre-refactor single-scalar read size.
constexpr uint32_t kReadBytes = 32;
}  // namespace routed_guard_detail

#ifdef ROUTED_GUARD_ENABLED

#include "ckernel.h"
#include "ckernel_defs.h"

#ifdef GUARD_COMPUTE_KERNEL

// TRISC: read the skip decision published by the BRISC-side dataflow kernel
// (in1_* reader/writer) via the hardware mailbox.  mailbox_read is a blocking
// read — it stalls until BRISC has written — so no explicit synchronization
// is needed.  Each of Unpack / Math / Pack receives its own message; the
// compute kernel is compiled once but executes on all three threads, and the
// UNPACK/MATH/PACK macros select the correct slot per thread.
FORCE_INLINE bool guard_check_wait() {
    uint32_t skip = 0;
    UNPACK(skip = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(skip = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(skip = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    return skip != 0;
}

#else  // dataflow side

namespace routed_guard_detail {

// Read 32 bytes (8 uint32s of face-0 row 0) from DRAM bank 0 `dram_addr` into
// `scratch_l1`, barrier, and return the value at logical row-0 column `idx`.
// Assumes idx < 8 and `scratch_l1` points to an exclusive 32-byte L1 region
// (callers that issue two reads must supply two disjoint scratch regions —
// sharing scratch across back-to-back reads was observed to deadlock).
FORCE_INLINE uint32_t read_indexed_u32(uint32_t dram_addr, uint32_t scratch_l1, uint32_t idx) {
    const uint64_t dram_src = get_noc_addr_from_bank_id<true>(0, dram_addr);
    noc_async_read(dram_src, scratch_l1, kReadBytes);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1)[idx];
}

}  // namespace routed_guard_detail

// NCRISC (in0_* readers): direct DRAM reads, no mailbox — ckernel::mailbox_base
// is not linked for NCRISC, and BRISC already owns the BRISC→TRISC handoff.
FORCE_INLINE bool guard_check_brisc() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");

    const uint32_t global_table_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t token_counts_addr = get_arg_val<uint32_t>(kArgBase + 1);
    const uint32_t local_expert_idx = get_arg_val<uint32_t>(kArgBase + 2);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 3);
    const uint32_t expert_iter_length = get_arg_val<uint32_t>(kArgBase + 4);

    const uint32_t scratch_base = get_write_ptr(kCbId) + routed_guard_detail::kScalarOffset;
    const uint32_t global_idx =
        routed_guard_detail::read_indexed_u32(global_table_addr, scratch_base, local_expert_idx);
    const uint32_t token_count = routed_guard_detail::read_indexed_u32(
        token_counts_addr, scratch_base + routed_guard_detail::kReadBytes, global_idx);

    return token_count <= curr_expert_iter * expert_iter_length;
}

// BRISC (in1_* readers): read the two DRAM tables, compute the skip decision,
// publish it to the three TRISC thread mailboxes so the compute kernel can
// early-return too, then return the decision for BRISC's own early-return.
// Always writes to all three mailboxes (even on skip=false) so TRISC always
// has a defined value to pop.
FORCE_INLINE bool guard_check_wait() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");

    const uint32_t global_table_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t token_counts_addr = get_arg_val<uint32_t>(kArgBase + 1);
    const uint32_t local_expert_idx = get_arg_val<uint32_t>(kArgBase + 2);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 3);
    const uint32_t expert_iter_length = get_arg_val<uint32_t>(kArgBase + 4);

    const uint32_t scratch_base = get_read_ptr(kCbId) + routed_guard_detail::kScalarOffset;
    const uint32_t global_idx =
        routed_guard_detail::read_indexed_u32(global_table_addr, scratch_base, local_expert_idx);
    const uint32_t token_count = routed_guard_detail::read_indexed_u32(
        token_counts_addr, scratch_base + routed_guard_detail::kReadBytes, global_idx);

    const bool skip = token_count <= curr_expert_iter * expert_iter_length;
    const uint32_t skip_u = skip ? 1u : 0u;
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, skip_u);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, skip_u);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, skip_u);
    return skip;
}

#endif  // GUARD_COMPUTE_KERNEL

#else  // !ROUTED_GUARD_ENABLED

FORCE_INLINE bool guard_check_brisc() { return false; }
FORCE_INLINE bool guard_check_wait() { return false; }

#endif  // ROUTED_GUARD_ENABLED
