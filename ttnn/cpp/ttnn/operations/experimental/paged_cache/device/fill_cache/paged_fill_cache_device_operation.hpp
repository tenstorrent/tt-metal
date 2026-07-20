// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "paged_fill_cache_program_factory.hpp"

#include "paged_fill_cache_device_operation_types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedFillCacheDeviceOperation {
    using operation_attributes_t = PagedFillCacheParams;
    using tensor_args_t = PagedFillCacheInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PagedFillCacheProgramFactory, PagedFillCacheMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // batch_idx_fallback and noop are excluded from the program hash (so calls differing only in
    // them cache-hit). Of these only batch_idx_fallback is genuinely dynamic: in the scalar-fallback
    // path (no batch_idx_tensor) it is baked into a writer runtime arg, so it must be re-applied to
    // the cached program on every dispatch. Returns empty in batch-idx-tensor mode (the writer pushes
    // a Buffer* the framework already re-patches) and for coords excluded from a mesh dispatch. noop
    // is derived from the hashed mesh_coords and is therefore stable across cache hits, so it is not
    // re-patched.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor paged_fill_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<Tensor>& batch_idx_tensor,
    uint32_t batch_idx_fallback,
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords = std::nullopt,
    std::optional<uint32_t> block_size_override = std::nullopt,
    std::optional<uint32_t> cache_position_modulo = std::nullopt,
    const std::optional<Tensor>& valid_seq_len_tensor = std::nullopt);

}  // namespace ttnn::prim
