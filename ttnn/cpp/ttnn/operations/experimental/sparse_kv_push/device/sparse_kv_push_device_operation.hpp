// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "sparse_kv_push_device_operation_types.hpp"
#include "sparse_kv_push_program_factory.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvPushDeviceOperation {
    using operation_attributes_t = SparseKvPushParams;
    using tensor_args_t = SparseKvPushInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<SparseKvPushProgramFactory>;

    // The cache slot and semaphore addresses are runtime args, so one program serves every slot and layer.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    uint32_t part = 0,
    uint32_t num_links = 2,
    uint32_t batch_index = 0,
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt);

}  // namespace ttnn::prim
