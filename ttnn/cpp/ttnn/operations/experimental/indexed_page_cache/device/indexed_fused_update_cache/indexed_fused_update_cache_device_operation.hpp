// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "indexed_fused_update_cache_device_operation_types.hpp"
#include "indexed_fused_update_cache_program_factory.hpp"

namespace ttnn::experimental::prim::indexed_fused_update_cache {

struct IndexedFusedUpdateCacheDeviceOperation {
    using operation_attributes_t = IndexedFusedUpdateCacheParams;
    using tensor_args_t = IndexedFusedUpdateCacheInputs;
    using spec_return_value_t = IndexedFusedUpdateCacheResultSpec;
    // The device-operation topology hook currently accepts this exact vector type.
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = IndexedFusedUpdateCacheResult;
    using program_factory_t = std::variant<IndexedFusedUpdateCacheProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim::indexed_fused_update_cache

namespace ttnn::prim {

ttnn::experimental::prim::indexed_fused_update_cache::IndexedFusedUpdateCacheResult indexed_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const Tensor& physical_update_idxs_tensor);

}  // namespace ttnn::prim
