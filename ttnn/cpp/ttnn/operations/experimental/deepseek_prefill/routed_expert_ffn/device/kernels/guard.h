// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared guard helpers for routed_matmul kernels.
//
// Each dataflow kernel independently reads max_expert_iter directly from DRAM
// into a small L1 scratch region (cb_guard), then compares against
// curr_expert_iter.  Because max_expert_iter is a read-only scalar that never
// changes during the FFN pass, no token-based synchronization between the two
// dataflow threads is needed — both read the same immutable value and reach
// the same skip/execute decision independently.
//
// TRISC (compute) cannot issue NOC reads, so it cannot re-read the DRAM scalar
// itself.  Instead, the BRISC-side dataflow kernel publishes the skip decision
// to the three TRISC thread mailboxes (Unpack/Math/Pack) inside
// guard_check_wait(), and TRISC pops the flag with the hardware-blocking
// mailbox_read inside its own guard_check_wait().  mailbox_read stalls
// automatically until BRISC has written, so no extra synchronization is
// needed.  NCRISC cannot link ckernel::mailbox_base and therefore does not
// participate in the BRISC→TRISC handoff; its decision (identical to BRISC's
// by construction) is consumed only by its own kernel's early-return.
//
// Function names vs. processors (mapping set by the program factory):
//   guard_check_brisc() — called from in0_* readers, compiled onto RISCV_1 (NCRISC).
//   guard_check_wait()  — called from in1_* readers, compiled onto RISCV_0 (BRISC).
//                       — also called from bmm_routed (compute/TRISC) via GUARD_COMPUTE_KERNEL.
// The historical names predate the BRISC⇄NCRISC swap in the factory; the
// implementations here reflect the actual processor assignment.
//
// Semantics: skip iff curr_expert_iter > max_expert_iter.  Populate the DRAM
// tensor with the max *valid* iteration index (set to UINT32_MAX to disable).
//
// Named compile-time args set by the program factory (via named_compile_args):
//   GUARD_CB_ID     - CB index for cb_guard (scratch for the DRAM read; dataflow only)
//   GUARD_ARG_BASE  - starting runtime arg index for the 2 guard args (dataflow only)
//
// Runtime args at positions GUARD_ARG_BASE+{0,1} (dataflow only):
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

// NCRISC (in0_* readers): direct DRAM read, no mailbox — ckernel::mailbox_base
// is not linked for NCRISC, and BRISC already owns the BRISC→TRISC handoff.
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

// BRISC (in1_* readers): read DRAM, compute the skip decision, publish it to
// the three TRISC thread mailboxes so the compute kernel can early-return
// too, then return the decision for BRISC's own early-return.  Always writes
// to all three mailboxes (even on skip=false) so TRISC always has a defined
// value to pop.
FORCE_INLINE bool guard_check_wait() {
    constexpr uint32_t kArgBase = get_named_compile_time_arg_val("GUARD_ARG_BASE");
    constexpr uint32_t kCbId = get_named_compile_time_arg_val("GUARD_CB_ID");

    const uint32_t max_iter_addr = get_arg_val<uint32_t>(kArgBase);
    const uint32_t curr_expert_iter = get_arg_val<uint32_t>(kArgBase + 1);

    const uint32_t scratch = get_read_ptr(kCbId) + routed_guard_detail::kScalarOffset;
    uint64_t dram_src = get_noc_addr_from_bank_id<true>(0, max_iter_addr);
    noc_async_read(dram_src, scratch, 32);
    noc_async_read_barrier();

    const uint32_t max_expert_iter = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const bool skip = curr_expert_iter > max_expert_iter;
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
