// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_scatter_minimal_direct_op_device_operation_types.hpp"
#include "reduce_scatter_minimal_direct_factory.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::experimental::prim {

// Device operation for the direct (one-shot) reduce-scatter. Modern device_operation::launch pattern:
// the primitive is the free function ttnn::prim::reduce_scatter_minimal_direct below.
struct ReduceScatterMinimalDirectDeviceOperation {
    using operation_attributes_t = ReduceScatterMinimalDirectParams;
    using tensor_args_t = ReduceScatterMinimalDirectInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ReduceScatterMinimalDirectMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // [0] = output slice, [1] = staging for the incoming contributions.
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Primitive entry (registered by definition, via device_operation::launch). Returns [output, staging];
// the host entry returns index 0.
std::vector<ttnn::Tensor> reduce_scatter_minimal_direct(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const ttnn::MemoryConfig& output_mem_config,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<ttnn::Tensor>& persistent_staging_tensor,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<CoreRangeSet>& sub_core_grid);

}  // namespace ttnn::prim
