// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// sparse_kv_dispatch: the KV rows every chip's top-k needs, sent over sparse_dispatch_2d-style trees into each chip's
// full-size gather buffer at the rows' owner-major pages (KV_DISPATCH_2D_PLAN.md).
struct SparseKvDispatchParams {
    // Hashed.
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t num_links = 2;
    uint32_t rows_per_packet = 0;  // 0: as many as the fabric payload fits
    uint32_t landing_slots = 16;
    uint32_t planners = 48;
    uint32_t lanes_per_link = 2;  // independent core pairs per (direction, link); > 1 shares the link through a mux
    CoreRangeSet core_grid;
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;

    // Not hashed: runtime args. Persistent words on the core grid, zero between calls (each call returns them to
    // zero). Optional only because GlobalSemaphore is not default constructible and the params are reflected.
    std::optional<GlobalSemaphore> fence_semaphore;
    std::optional<GlobalSemaphore> arrived_semaphore;
    std::optional<GlobalSemaphore> credit_semaphore;
    std::optional<GlobalSemaphore> inbox_semaphore;

    uint32_t batch_index = 0;
    uint32_t batch_slot_num_layers = 1;
    uint32_t batch_slot_layer_idx = 0;

    // Fused plan (kvd4, with SparseKvDispatchInputs::plan_indices): the chunk start as a host value (a runtime arg,
    // not hashed; its presence is) or, in the inputs, as a device tensor.
    std::optional<uint32_t> plan_chunk_start;

    // Active only for chunks starting at or after this row (0: always). The kernels read the chunk start from the plan
    // table's header and, below it, return before any fabric or semaphore traffic (every chip sees the same start).
    uint32_t active_from = 0;

    // Which of each requester's needed rows this call sends (sparse_push::PushPart: 0 ALL, 1 PREFIX = rows of blocks
    // before the plan's split block, 2 REST = the others). Hashed (a compile-time define of the planners).
    uint32_t part = 0;

    // The inbox already holds this plan's slices (an earlier call on the same plan build exchanged them): skip the
    // slice exchange. Hashed (a compile-time define of TX and the planners).
    bool skip_exchange = false;
};

struct SparseKvDispatchInputs {
    Tensor input_tensor;   // KVPE cache, sharded over rows (dim 2) across the whole mesh
    Tensor output_tensor;  // the full-size gather buffer [1, 1, max_rows, width], replicated, owner-major pages
    Tensor table;          // this chip's owner-major plan table (sparse_push_plan owner_major)
    Tensor inbox;          // one slot per requester, as the push's inbox
    Tensor scratch;        // DRAM own lists and onward queues (sparse_kv_dispatch_scratch_spec)
    std::optional<Tensor> batch_index_tensor;
    // Fused plan (kvd4): this chip's top-k indices [1, 1, rows, topk] UINT32; the op builds `table` itself (the
    // dispatch-only owner-major plan on every core's five RISCs) before dispatching.
    std::optional<Tensor> plan_indices;
    std::optional<Tensor> plan_chunk_start_tensor;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
