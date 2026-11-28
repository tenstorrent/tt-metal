// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "sparse_kv_dispatch_device_operation_types.hpp"
#include "sparse_kv_dispatch_program_factory.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvDispatchDeviceOperation {
    using operation_attributes_t = SparseKvDispatchParams;
    using tensor_args_t = SparseKvDispatchInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<SparseKvDispatchMeshWorkloadFactory>;

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The mesh's axes as the dispatch sees them: rows x columns, and whether each axis is a ring.
struct KvDispatchMesh {
    uint32_t rows = 0, cols = 0;
    bool wrap_x = false, wrap_y = false;
    uint32_t gmax() const { return (cols - 1) + (rows - 1) - 1; }
};
KvDispatchMesh kv_dispatch_mesh(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t num_links);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

Tensor sparse_kv_dispatch(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    const Tensor& inbox,
    const Tensor& scratch,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& fence_semaphore,
    const GlobalSemaphore& arrived_semaphore,
    const GlobalSemaphore& credit_semaphore,
    const GlobalSemaphore& inbox_semaphore,
    uint32_t num_links,
    uint32_t rows_per_packet,
    uint32_t landing_slots,
    uint32_t planners,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid,
    uint32_t lanes_per_link,
    const std::optional<Tensor>& plan_indices = std::nullopt,
    std::optional<uint32_t> plan_chunk_start = std::nullopt,
    const std::optional<Tensor>& plan_chunk_start_tensor = std::nullopt,
    uint32_t part = 0,
    bool skip_exchange = false,
    const std::optional<Tensor>& rank_table = std::nullopt,
    uint32_t sp_axis = 0,
    uint32_t replay = 0,
    const std::optional<Tensor>& replay_buffer = std::nullopt,
    uint32_t replay_generation = 0,
    bool single_program = false);

}  // namespace ttnn::prim
