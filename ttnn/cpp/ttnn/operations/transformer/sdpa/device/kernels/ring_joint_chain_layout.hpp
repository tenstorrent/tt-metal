// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

constexpr uint32_t kChainConfigRuntimeArgCount = 16;

constexpr uint32_t kChainSenderSemaphoreCompileArgOffset = 0;
constexpr uint32_t kChainReceiverSemaphoreCompileArgOffset = 1;
constexpr uint32_t kChainValidSemaphoreCompileArgOffset = 2;
constexpr uint32_t kChainSemaphoreCompileArgCount = 3;
constexpr uint32_t kChainMcastEnabledCompileArgOffset = kChainSemaphoreCompileArgCount;
constexpr uint32_t kChainCompileArgCount = kChainSemaphoreCompileArgCount + 1;

// Tensor K/V GQA: fewer K/V heads than Q heads, one K/V head shared by each query-head group.
// Latent-V modes are excluded because V is packed into K and uses the existing shared-K path.
constexpr bool is_gqa_grouped_kv_head_mode(bool v_shares_k_buffer, uint32_t nqh, uint32_t nkh, uint32_t nvh) {
    return !v_shares_k_buffer && (nkh == nvh) && (nkh > 0) && (nkh < nqh) && (nqh % nkh == 0);
}

// The batch chain is only for non-GQA shared-K cases (separate-V shared-K and latent-V MLA).
// GQA with one local KV head still has NHK == 1, but it must use grouped KV-head transport.
constexpr bool uses_shared_k_batch_chain(bool gqa_grouped_kv, uint32_t nkh) { return !gqa_grouped_kv && (nkh == 1); }

// The per-head store-and-forward chain carries MHA K/V, or separate-V's V. Latent-V never streams V
// through it -- V is materialized from, or read in place of, the mcast K^T -- and latent-V is
// validated to NKH == 1, so K already rides the batch chain there and the head chain carried
// nothing. Skipping it also frees its 3 program semaphores, which the ring-8 rotated-Q-split
// handoff needs to stay under NUM_SEMAPHORES. Shared so the factory's two arg-layout derivations
// and the reader cannot disagree: a mismatch silently shifts compile-time arg offsets.
constexpr bool uses_v_head_chain(bool enable_kv_chains, bool gqa_grouped_kv, bool v_shares_k_buffer) {
    return enable_kv_chains && !gqa_grouped_kv && !v_shares_k_buffer;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
