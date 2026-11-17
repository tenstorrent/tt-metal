// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "sparse_push_plan_device_operation_types.hpp"
#include "sparse_push_plan_program_factory.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparsePushPlanDeviceOperation {
    using operation_attributes_t = SparsePushPlanParams;
    using tensor_args_t = SparsePushPlanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;  // {table, compact indices}
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<SparsePushPlanProgramFactory>;

    // A host chunk start is a runtime arg, so one program serves every chunk.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    std::optional<uint32_t> chunk_start = std::nullopt,
    const std::optional<Tensor>& chunk_start_tensor = std::nullopt);

}  // namespace ttnn::prim
