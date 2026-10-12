// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// vsa_ring_sdpa: the VSA fine stage fused with the SP-ring all-gather of K/V (VSA_RING_SDPA_SPEC.md).
// `vsa` is the plain vsa_sdpa contract (raw-selection streaming kernel); `ag` carries the ring geometry
// (links, topology, cluster axis, the two GlobalSemaphores [direction 0, direction 1]); the gather's sender
// cores fill the compute grid's first row(s), the VSA engine the rest.
// Which all-gather forwards K/V around the ring inside the fused program.
//   RingAttention: the stock ring_attention_all_gather_async helper, unmodified (one worker per link per
//                  direction, direct fabric connections, per-shard signals incl. the even-ring split of the
//                  diametric shard). The leaders gate per landed SHARD. The mergeable default.
//   FusedKv:       the op's own multi-worker (MUX) token-major gather (vsa_kv_gather_*.cpp); the leaders gate per
//                  BLOCK by polling the workers' landed-tile counters (VSA_RING_SDPA_SPEC.md sections 13-15).
enum class VsaRingGather : uint32_t { RingAttention = 0, FusedKv = 1 };

struct VsaRingSdpaParams {
    VsaSdpaParams vsa;
    ttnn::experimental::prim::RingAttentionAllGatherAsyncParams ag;  // ring geometry: links, topology, axis, semaphores
    VsaRingGather gather = VsaRingGather::RingAttention;
    uint32_t num_workers_per_link = 2;  // FusedKv only: gather workers per direction per link (+1 MUX core each)

    // Explicit reflection: `ag` is not an aggregate (constructor-only), so the framework's automatic
    // member introspection cannot describe this struct; the hash is custom (compute_program_hash).
    static constexpr auto attribute_names = std::forward_as_tuple("vsa", "ag", "gather", "num_workers_per_link");
    auto attribute_values() const { return std::forward_as_tuple(vsa, ag, gather, num_workers_per_link); }
};

struct VsaRingSdpaInputs {
    // vsa.k / vsa.v: this device's K and V shards, the plain head-split [1, H, T_local, d] (create_heads output).
    // indices/counts index the global sequence.
    VsaSdpaInputs vsa;
    Tensor gathered_k;  // [1, H, T_local*ring_size, d] persistent all-gather buffers (shard s at rows
    Tensor gathered_v;  //   [s*T_local, (s+1)*T_local)); the local shard is never written into them
};

// Sender cores the gather occupies, placed from (0, 0) row-major: RingAttention one per link per direction; FusedKv
// two directions of (workers + one MUX core) per link.
inline uint32_t vsa_ring_sender_cores(const VsaRingSdpaParams& args) {
    if (args.gather == VsaRingGather::RingAttention) {
        return 2 * args.ag.num_links;
    }
    const uint32_t mux = args.num_workers_per_link == 1 ? 0u : 1u;
    return args.ag.num_links * 2 * (args.num_workers_per_link + mux);
}

}  // namespace ttnn::prim
