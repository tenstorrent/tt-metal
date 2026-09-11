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
// (links, topology, cluster axis, the two GlobalSemaphores [backward, forward]); the sender cores start
// at `ccl_core_grid_offset`, which must be the first row of the grid's last column.
struct VsaRingSdpaParams {
    VsaSdpaParams vsa;
    ttnn::experimental::prim::RingAttentionAllGatherAsyncParams ag;  // ring geometry: links, topology, axis, semaphores
    uint32_t num_workers_per_link = 2;  // all-gather workers per direction per link (senders = 2*links*(w+1) cores)

    // Explicit reflection: `ag` is not an aggregate (constructor-only), so the framework's automatic
    // member introspection cannot describe this struct; the hash is custom (compute_program_hash).
    static constexpr auto attribute_names = std::forward_as_tuple("vsa", "ag", "num_workers_per_link");
    auto attribute_values() const { return std::forward_as_tuple(vsa, ag, num_workers_per_link); }
};

struct VsaRingSdpaInputs {
    // vsa.k and vsa.v are BOTH the local concatenated K/V shard [1, 2H, T_local, d] (K heads first): the
    // kernels read K at head h and V at head H + h. indices/counts index the global sequence.
    VsaSdpaInputs vsa;
    Tensor gathered_kv;  // [1, 2H, T_local*ring_size, d] persistent all-gather buffer (shard s at rows
                         //   [s*T_local, (s+1)*T_local)); the local shard is never written into it
};

}  // namespace ttnn::prim
