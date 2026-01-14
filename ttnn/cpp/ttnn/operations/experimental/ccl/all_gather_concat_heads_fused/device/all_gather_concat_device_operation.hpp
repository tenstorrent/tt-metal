// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device operation header for AllGatherConcat (heads-fused) using the TMP pattern.

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include <optional>
#include <variant>

namespace ttnn::operations::experimental::ccl::all_gather_concat_heads_fused {

struct AllGatherConcatDeviceOperation {
    using operation_attributes_t = all_gather_concat_heads_fused::operation_attributes_t;
    using tensor_args_t = all_gather_concat_heads_fused::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::AllGatherConcatMeshWorkloadFactory>;
    using shared_variables_t = program::AllGatherConcatMeshWorkloadFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::all_gather_concat_heads_fused

namespace ttnn::prim {

ttnn::operations::experimental::ccl::all_gather_concat_heads_fused::AllGatherConcatDeviceOperation::
    tensor_return_value_t
    all_gather_concat(
        const Tensor& input_tensor,
        Tensor& buffer_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& global_semaphore,
        uint32_t num_heads,
        const MemoryConfig& memory_config,
        bool use_noc1_only,
        std::optional<uint32_t> num_links,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id);

}  // namespace ttnn::prim
