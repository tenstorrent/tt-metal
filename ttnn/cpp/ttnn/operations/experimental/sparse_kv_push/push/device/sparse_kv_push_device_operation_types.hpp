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
#include "ttnn/operations/experimental/sparse_kv_push/push/sparse_kv_push_types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvPushParams {
    // Hashed.
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    SparseKvPushPart part = SparseKvPushPart::kAll;
    uint32_t num_links = 2;
    CoreRangeSet core_grid;
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;

    // Not hashed: runtime args, patched and validated on every dispatch.
    // Required; optional only because GlobalSemaphore is not default constructible and the params are reflected.
    std::optional<GlobalSemaphore> ready_semaphore;
    std::optional<GlobalSemaphore> done_semaphore;

    uint32_t batch_index = 0;
    uint32_t batch_slot_num_layers = 1;
    uint32_t batch_slot_layer_idx = 0;
};

struct SparseKvPushInputs {
    Tensor input_tensor;   // KVPE cache, sharded over rows (dim 2) across the whole mesh
    Tensor output_tensor;  // the requester's compact buffer: one per chip, holding the rows its own top-k needs
    Tensor table;          // all-gathered sparse_push_plan tables
    std::optional<Tensor> batch_index_tensor;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
