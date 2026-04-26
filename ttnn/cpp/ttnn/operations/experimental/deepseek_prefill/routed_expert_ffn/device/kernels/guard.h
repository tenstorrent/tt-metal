// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared guard helpers for routed_matmul kernels.
//
// Each dataflow kernel independently reads two DRAM-interleaved ROW_MAJOR uint32
// tables (global_expert_idx_table, expert_token_counts) into an L1 scratch region
// (cb_guard) via TensorAccessor. The skip predicate is:
//
//   global_idx = global_expert_idx_table[local_expert_idx]
//   skip       = expert_token_counts[global_idx] <= curr_expert_iter * expert_iter_length
//
// TensorAccessor resolves the correct bank + offset for a given page, so this
// works with arbitrary DRAM-interleaved memory configs — no assumption that the
// data lives in bank 0. Both tables are read as a single page (their innermost
// dim × dtype is one page) and then indexed within L1 scratch.
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
// ckernel::mailbox_base and therefore does not participate in the BRISC->TRISC
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
//   GUARD_CB_ID                       - CB index for cb_guard (scratch for the DRAM reads;
//                                       dataflow only). Must be large enough to hold one
//                                       page of each table back-to-back (see program factory).
//   GUARD_ARG_BASE                    - starting runtime arg index for the 5 guard args
//                                       (dataflow only).
//   GUARD_GLOBAL_TABLE_CTA_OFFSET     - compile-time-arg offset for global_expert_idx_table's
//                                       TensorAccessorArgs (dataflow only).
//   GUARD_COUNTS_CTA_OFFSET           - compile-time-arg offset for expert_token_counts'
//                                       TensorAccessorArgs (dataflow only).
//
// Runtime args at positions GUARD_ARG_BASE+{0..4} (dataflow only):
//   [0] global_expert_idx_table DRAM buffer address (uint32)
//   [1] expert_token_counts     DRAM buffer address (uint32)
//   [2] local_expert_idx                           (uint32)
//   [3] curr_expert_iter                           (uint32)
//   [4] expert_iter_length                         (uint32)
//
// Current limitation: each table is read as a single page (page 0), so the
// full table must fit in one ROW_MAJOR page (= innermost dim × dtype_bytes)
// AND in one 256-byte half of cb_guard (= 64 uint32 elements). Larger tables
// need either multi-page reads keyed on the logical index or a bigger cb_guard.
//
// If ROUTED_GUARD_ENABLED is not defined, the helpers compile to no-ops.

#pragma once

#include <cstdint>

namespace routed_guard_detail {
// Each table's page is read into its own 256-byte scratch half of cb_guard so
// both values are resident simultaneously and the second DRAM read cannot
// stomp on the first's L1 destination.
constexpr uint32_t kGuardScratchHalfBytes = 256;
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

#include "api/tensor/tensor_accessor.h"

namespace routed_guard_detail {

// Read page 0 of `accessor` (one innermost-dim row) into `scratch_l1`, barrier,
// and return the uint32 at position `idx` within that row.
template <typename Accessor>
FORCE_INLINE uint32_t read_page_indexed_u32(const Accessor& accessor, uint32_t scratch_l1, uint32_t idx) {
    noc_async_read_page(0, accessor, scratch_l1);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1)[idx];
}

}  // namespace routed_guard_detail

// NCRISC (in0_* readers): direct DRAM reads, no mailbox — ckernel::mailbox_base
// is not linked for NCRISC, and BRISC already owns the BRISC->TRISC handoff.
FORCE_INLINE bool guard_check_brisc() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");
    constexpr uint32_t kGlobalTableCtaOff = get_named_compile_time_arg_val("GUARD_GLOBAL_TABLE_CTA_OFFSET");
    constexpr uint32_t kCountsCtaOff = get_named_compile_time_arg_val("GUARD_COUNTS_CTA_OFFSET");

    const uint32_t global_table_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t token_counts_addr = get_arg_val<uint32_t>(kArgBase + 1);
    const uint32_t local_expert_idx = get_arg_val<uint32_t>(kArgBase + 2);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 3);
    const uint32_t expert_iter_length = get_arg_val<uint32_t>(kArgBase + 4);

    constexpr auto global_table_args = TensorAccessorArgs<kGlobalTableCtaOff>();
    constexpr auto counts_args = TensorAccessorArgs<kCountsCtaOff>();
    const auto global_table_accessor = TensorAccessor(global_table_args, global_table_addr);
    const auto counts_accessor = TensorAccessor(counts_args, token_counts_addr);

    const uint32_t scratch_a = get_write_ptr(kCbId);
    const uint32_t scratch_b = scratch_a + routed_guard_detail::kGuardScratchHalfBytes;

    const uint32_t global_idx =
        routed_guard_detail::read_page_indexed_u32(global_table_accessor, scratch_a, local_expert_idx);
    const uint32_t token_count = routed_guard_detail::read_page_indexed_u32(counts_accessor, scratch_b, global_idx);

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
    constexpr uint32_t kGlobalTableCtaOff = get_named_compile_time_arg_val("GUARD_GLOBAL_TABLE_CTA_OFFSET");
    constexpr uint32_t kCountsCtaOff = get_named_compile_time_arg_val("GUARD_COUNTS_CTA_OFFSET");

    const uint32_t global_table_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t token_counts_addr = get_arg_val<uint32_t>(kArgBase + 1);
    const uint32_t local_expert_idx = get_arg_val<uint32_t>(kArgBase + 2);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 3);
    const uint32_t expert_iter_length = get_arg_val<uint32_t>(kArgBase + 4);

    constexpr auto global_table_args = TensorAccessorArgs<kGlobalTableCtaOff>();
    constexpr auto counts_args = TensorAccessorArgs<kCountsCtaOff>();
    const auto global_table_accessor = TensorAccessor(global_table_args, global_table_addr);
    const auto counts_accessor = TensorAccessor(counts_args, token_counts_addr);

    const uint32_t scratch_a = get_read_ptr(kCbId);
    const uint32_t scratch_b = scratch_a + routed_guard_detail::kGuardScratchHalfBytes;

    const uint32_t global_idx =
        routed_guard_detail::read_page_indexed_u32(global_table_accessor, scratch_a, local_expert_idx);
    const uint32_t token_count = routed_guard_detail::read_page_indexed_u32(counts_accessor, scratch_b, global_idx);

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
