// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — declarations shared by ALL THREE kernels.
//
// Deliberately include-free so the COMPUTE translation unit can pull it in as well: a compute
// kernel must not see the dataflow API. The dataflow-only bank-run reader/writer lives next door in
// `moe_fused_swiglu_bank_runs.hpp`.
//
// SINGLE SOURCE OF TRUTH for the token-count mailbox word layout. The reader publishes it, and
// compute AND the writer read it, so the word indices must be written down exactly once (they were
// bare literals in three files before).

#pragma once

#include <cstdint>

namespace moe_fused_swiglu {

// L1 mailbox word layout. The reader fills 0..2 and then stamps MAGIC into word 3; every other
// kernel spins on word 3 and only then reads 0..2. One page (64 B) per core, zeroed host-side so a
// stale magic from a previous dispatch can never be mistaken for a fresh publish.
constexpr uint32_t MBOX_COUNT = 0;     // counts[idx[local_expert_id]] — the RUNTIME token count
constexpr uint32_t MBOX_M_T = 1;       // ceil(count/32), clamped to M_T_MAX
constexpr uint32_t MBOX_M_BLOCKS = 2;  // ceil(M_t / M_BLOCK) — the outer-loop trip count
constexpr uint32_t MBOX_READY = 3;     // == MAILBOX_MAGIC once words 0..2 are valid

}  // namespace moe_fused_swiglu
