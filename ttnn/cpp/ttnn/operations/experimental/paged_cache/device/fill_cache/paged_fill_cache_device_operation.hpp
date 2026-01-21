// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "paged_fill_cache_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "paged_fill_cache_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct PagedFillCacheDeviceOperation {
    using operation_attributes_t = PagedFillCacheParams;
    using tensor_args_t = PagedFillCacheInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PagedFillCacheProgramFactory, PagedFillCacheMeshWorkloadFactory>;
    using shared_variables_t = PagedFillCacheProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor paged_fill_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<Tensor>& batch_idx_tensor,
    uint32_t batch_idx_fallback,
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords = std::nullopt);

}  // namespace ttnn::prim
