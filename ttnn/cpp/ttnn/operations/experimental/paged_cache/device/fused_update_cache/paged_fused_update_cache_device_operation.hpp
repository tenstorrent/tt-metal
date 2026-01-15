// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "paged_fused_update_cache_device_operation_types.hpp"
#include "paged_tiled_fused_update_cache_program_factory.hpp"
#include "paged_row_major_fused_update_cache_program_factory.hpp"

namespace ttnn::operations::experimental::paged_cache::fused_update {

struct PagedFusedUpdateCacheDeviceOperation {
    using operation_attributes_t = FusedUpdateParams;
    using tensor_args_t = FusedUpdateInputs;
    using spec_return_value_t = fused_update::spec_return_value_t;
    using tensor_return_value_t = fused_update::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::tiled::PagedTiledFusedUpdateCacheProgramFactory,
        program::rm::PagedRowMajorFusedUpdateCacheProgramFactory,
        program::tiled::PagedTiledFusedUpdateCacheMeshWorkloadFactory,
        program::rm::PagedRowMajorFusedUpdateCacheMeshWorkloadFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::paged_cache::fused_update

namespace ttnn::prim {

ttnn::operations::experimental::paged_cache::fused_update::PagedFusedUpdateCacheDeviceOperation::tensor_return_value_t
paged_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords);

}  // namespace ttnn::prim
