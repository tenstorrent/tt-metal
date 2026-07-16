// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "paged_update_cache_device_operation_types.hpp"
#include "paged_update_cache_program_factory.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedUpdateCacheDeviceOperation {
    using operation_attributes_t = PagedUpdateCacheParams;
    using tensor_args_t = PagedUpdateCacheInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PagedUpdateCacheProgramFactory, PagedUpdateCacheMeshWorkloadFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // update_idxs is excluded from the program hash (so decode steps that differ only in position
    // cache-hit). The cache-write offsets it determines (cache_start_id, tile_update_offset_B) are
    // DYNAMIC and re-applied to the cached program on every dispatch. Returns empty in index-tensor
    // mode (positions read on-device) and for coords excluded from a mesh dispatch.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PagedUpdateCacheDeviceOperation::tensor_return_value_t paged_update_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords,
    std::optional<uint32_t> block_size_override = std::nullopt,
    std::optional<uint32_t> num_kv_heads_override = std::nullopt,
    std::optional<uint32_t> cache_position_modulo = std::nullopt);

}  // namespace ttnn::prim
