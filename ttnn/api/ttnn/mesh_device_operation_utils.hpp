// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/device_operation_detail.hpp"

namespace ttnn::device_operation::mesh_device_operation_utils {

template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

// Returns true if all tensors have uniform storage, false otherwise.
template <typename TensorArgs>
bool all_tensors_have_uniform_storage(const TensorArgs& tensor_args) {
    bool uniform_storage = true;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) { uniform_storage &= tensor.device_storage().is_uniform_storage(); }, tensor_args);
    return uniform_storage;
}

// Filters shards from `tensor_return_value` that are in `tensor_coordinates`.
// Returns a new TensorReturnValue with filtered coordinates (immutable operation).
template <typename TensorReturnValue>
TensorReturnValue filter_tensor_shards(
    const std::vector<ttnn::MeshCoordinate>& tensor_coordinates, const TensorReturnValue& tensor_return_value) {
    return tt::stl::reflection::transform_object_of_type<Tensor>(
        [&](const Tensor& tensor) -> Tensor {
            const auto& old_storage = tensor.device_storage();

            // Build filtered coords list
            std::vector<ttnn::MeshCoordinate> filtered_coords;
            filtered_coords.reserve(tensor_coordinates.size());

            auto coord_it = tensor_coordinates.cbegin();
            auto storage_it = old_storage.coords.cbegin();
            while (coord_it != tensor_coordinates.end() && storage_it != old_storage.coords.end()) {
                if (*storage_it == *coord_it) {
                    filtered_coords.push_back(*storage_it);
                    ++coord_it;
                    ++storage_it;
                } else {
                    ++storage_it;
                }
            }

            // Create new storage with filtered coords, sharing the mesh_buffer
            tt::tt_metal::DeviceStorage new_storage(old_storage.mesh_buffer, std::move(filtered_coords));

            // Return new tensor with new storage
            return Tensor(std::move(new_storage), tensor.tensor_spec(), tensor.tensor_topology());
        },
        tensor_return_value);
}

// Verifies all tensors span the same set of coordinates, and returns them in a vector.
// If no tensors are found, returns zero coordinate.
// This template extracts tensors and delegates to the non-template implementation
// in device_operation_detail.cpp to reduce per-operation template instantiation cost.
template <typename TensorArgs>
std::vector<ttnn::MeshCoordinate> extract_tensor_coordinates(
    const TensorArgs& tensor_args, ttnn::MeshDevice* mesh_device = nullptr) {
    std::vector<std::reference_wrapper<const Tensor>> tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&tensors](const Tensor& t) { tensors.push_back(std::cref(t)); }, tensor_args);
    return ttnn::device_operation::detail::extract_tensor_coordinates_impl(tensors, mesh_device);
}

// Sets runtime ID for all programs in `workload`.
inline void set_runtime_id(tt::tt_metal::distributed::MeshWorkload& workload) {
    auto op_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();
    tt::tt_metal::experimental::inspector::EmitMeshWorkloadRuntimeId(workload, op_id);
    for (auto& [_, program] : workload.get_programs()) {
        program.set_runtime_id(op_id);
    }
}

// Tracks all programs in `workload` and returns true if any program was hooked.
inline bool track_workload(tt::tt_metal::distributed::MeshWorkload& workload, ttnn::MeshDevice* mesh_device) {
    bool hook_program = false;
    for (auto& [_, program] : workload.get_programs()) {
        tt::tt_metal::GraphTracker::instance().track_program(&program, mesh_device);
        hook_program |= tt::tt_metal::GraphTracker::instance().hook_program(&program);
    }
    return hook_program;
}

// Applies override_runtime_arguments in a uniform way across different program factory interfaces because
// we have many different program factory interfaces we're currently supporting.
template <
    ProgramFactoryConcept ProgramFactory,
    typename OperationAttributes,
    typename TensorArgs,
    typename TensorReturnValue>
void apply_override_runtime_arguments(
    const ProgramFactory& factory,
    tt::tt_metal::Program& program,
    typename ProgramFactory::shared_variables_t& shared_vars,
    const OperationAttributes& attrs,
    const ttnn::MeshCoordinate& /*coord*/,
    const TensorArgs& tensor_args,
    TensorReturnValue& return_value) {
    if constexpr (requires {
                      typename ProgramFactory::cached_program_t;
                      factory.override_runtime_arguments(
                          std::declval<typename ProgramFactory::cached_program_t&>(), attrs, tensor_args, return_value);
                  }) {
        // Proxy references to `program` and `shared_vars` as a `CachedProgram` object.
        auto cached_program_proxy = ProgramFactory::cached_program_t::proxy(program, shared_vars);
        factory.override_runtime_arguments(cached_program_proxy, attrs, tensor_args, return_value);
    } else {
        factory.override_runtime_arguments(program, shared_vars, attrs, tensor_args, return_value);
    }
}

}  // namespace ttnn::device_operation::mesh_device_operation_utils
