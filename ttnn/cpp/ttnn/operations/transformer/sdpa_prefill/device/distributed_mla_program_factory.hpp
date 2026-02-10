// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include <tt-metalium/buffer_types.hpp>

namespace ttnn::operations::transformer::sdpa_prefill {

struct DistributedMlaSharedVariables {
    uint32_t device_order;
};

struct DistributedMlaMeshWorkloadFactory {
    struct operation_attributes_t {
        std::optional<uint32_t> cluster_axis;
        tt::tt_metal::MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor input_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using shared_variables_t = DistributedMlaSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::transformer::sdpa_prefill
