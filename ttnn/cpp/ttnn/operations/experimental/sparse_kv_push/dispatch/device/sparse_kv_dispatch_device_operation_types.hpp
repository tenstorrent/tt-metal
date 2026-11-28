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

    // Which of each requester's needed rows this call sends (sparse_push::PushPart: 0 ALL, 1 PREFIX = rows of blocks
    // before the plan's split block, 2 REST = the others). Hashed (a compile-time define of the planners).
    uint32_t part = 0;

    // The inbox already holds this plan's slices (an earlier call on the same plan build exchanged them): skip the
    // slice exchange. Hashed (a compile-time define of TX and the planners).
    bool skip_exchange = false;

    // The mesh axis the cache's blocks stripe along first (0 or 1), as sparse_kv_push: owner o of block b (= b % N)
    // is the chip at o / chips_on_other_axis on this axis and o % chips_on_other_axis on the other. Hashed.
    uint32_t sp_axis = 0;

    // Plan once, dispatch many (KVD_DESIGN=3): 0 off, 1 record the planners' routing into `replay_buffer` (every
    // part's rows; this call still sends its own part), 2 replay a recording of the same plan build (no slice reads or
    // routing; needs skip_exchange). Hashed (a compile-time define of the planners).
    uint32_t replay = 0;
    // The plan build's generation (replay != 0; not hashed, a common runtime arg): the record holds it and a replay
    // of another generation routes instead (the caller's skip_exchange / replay flags belong to one plan build).
    uint32_t replay_generation = 0;
    // v3: one program for the mesh even when the chips' port cores do not fit one program's grid (every chip the same
    // cores, next to the first chip's Ethernet cores) instead of a program per chip: one enqueue on a large mesh (8x4:
    // ~80 us less host time a call), slower ports on chips whose Ethernet cores are elsewhere. Hashed.
    bool single_program = false;
};

// The routing record's size (replay): a 4 KB header page, then per recorder (planner of the recording) two streams
// (prefix, rest) of up to 2 * ceil(5 * owner_rows / P) + 64 entries of 16 B each (a row's 4 directions and its local
// copy, x2 for the deal's imbalance; a recorder past that marks overflow and the replays route as usual). Sized for
// any P <= 62.
inline uint64_t sparse_kv_dispatch_replay_entries(uint32_t max_rows, uint32_t stripe_chunk, uint32_t num_devices) {
    const uint64_t owner_blocks = ((max_rows / stripe_chunk) + num_devices - 1) / num_devices;
    const uint64_t rows = owner_blocks * stripe_chunk;
    uint64_t worst = 0;
    for (uint64_t p = 1; p < 63; ++p) {
        const uint64_t e = 2 * p * ((2 * ((5 * rows + p - 1) / p) + 64 + 3) / 4 * 4);  // (kvd3_planner.cpp rec_cap)
        worst = e > worst ? e : worst;
    }
    return 256 + worst;
}

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
    // Compact landing: sparse_push_plan's rank table (rank_table=True) for this plan. Rows then land in a compact
    // buffer (the push's layout: the plan's compact indices address it) at compact_page(run_base[owner], rank).
    std::optional<Tensor> rank_table;
    // The routing record (replay != 0): a tensor of sparse_kv_dispatch_replay_spec, one per plan build.
    std::optional<Tensor> replay_buffer;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
