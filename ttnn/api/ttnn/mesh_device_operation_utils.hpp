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

#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation_concepts.hpp"

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
template <typename TensorReturnValue>
void filter_tensor_shards(
    const std::vector<ttnn::MeshCoordinate>& tensor_coordinates, TensorReturnValue& tensor_return_value) {
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor_to_return) {
            auto& tensor_storage =
                std::get<tt::tt_metal::DeviceStorage>(tensor_to_return.tensor_attributes->get_storage());

            auto coord_it = tensor_coordinates.cbegin();
            auto storage_it = tensor_storage.coords.begin();
            auto insert_it = tensor_storage.coords.begin();
            while (coord_it != tensor_coordinates.end() && storage_it != tensor_storage.coords.end()) {
                if (*storage_it == *coord_it) {
                    std::swap(*insert_it, *storage_it);
                    ++insert_it;
                    ++coord_it;
                    ++storage_it;
                } else {
                    ++storage_it;
                }
            }
            tensor_storage.coords.erase(insert_it, tensor_storage.coords.end());
        },
        tensor_return_value);
}

// Checks if the MeshCoordinateRangeSet containing all coordinates in b is a subset of a.
inline bool is_subset_of(const std::vector<MeshCoordinate>& a, const std::vector<MeshCoordinate>& b) {
    MeshCoordinateRangeSet a_set;
    MeshCoordinateRangeSet b_set;

    // Generate a MeshCoordinateRangeSet from the vectors of coordinates passed in
    for (const auto& coord : a) {
        a_set.merge(MeshCoordinateRange(coord));
    }
    for (const auto& coord : b) {
        b_set.merge(MeshCoordinateRange(coord));
    }
    // Check if b_set is a subset of a_set
    // This is true if every range in b_set is completely contained in some range
    // in a_set
    bool is_subset = false;
    for (const auto& b_range : b_set.ranges()) {
        is_subset = false;
        for (const auto& a_range : a_set.ranges()) {
            if (a_range.contains(b_range)) {
                is_subset = true;
                break;
            }
        }
        if (not is_subset) {
            return is_subset;
        }
    }
    return is_subset;
}

// Verifies all tensors span the same set of coordinates, and returns them in a vector.
// If no tensors are found, returns zero coordinate.
template <typename TensorArgs>
std::vector<ttnn::MeshCoordinate> extract_tensor_coordinates(
    const TensorArgs& tensor_args, ttnn::MeshDevice* mesh_device = nullptr) {
    auto first_tensor_opt = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);

    // If no tensor is found, return zero coordinate
    if (!first_tensor_opt.has_value()) {
        if (mesh_device == nullptr) {
            TT_THROW("No tensors found in tensor_args and no mesh_device provided to extract_tensor_coordinates");
        }
        return {MeshCoordinate::zero_coordinate(mesh_device->shape().dims())};
    }

    const Tensor& first_tensor = first_tensor_opt.value();
    std::vector<ttnn::MeshCoordinate> tensor_coordinates;
    std::transform(
        first_tensor.device_storage().coords.begin(),
        first_tensor.device_storage().coords.end(),
        std::back_inserter(tensor_coordinates),
        [](const auto& coord) { return coord; });
    // Verification Step: Assert if the tensors are placed on different coordinate ranges
    // that do not overlap.
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) {
            if (tensor.device_storage().coords.size() != tensor_coordinates.size()) {
                std::vector<ttnn::MeshCoordinate> tensor_mesh_coords;
                std::transform(
                    tensor.device_storage().coords.begin(),
                    tensor.device_storage().coords.end(),
                    std::back_inserter(tensor_mesh_coords),
                    [](const auto& coord) { return coord; });
                if (tensor_mesh_coords.size() < tensor_coordinates.size()) {
                    // Case 1: Current tensor is placed on a smaller set of coordinates than tensor_coordinates.
                    TT_ASSERT(
                        is_subset_of(tensor_coordinates, tensor_mesh_coords),
                        "Tensors are placed on different MeshCoordinate ranges that do not intersect.");
                    tensor_coordinates = std::move(tensor_mesh_coords);
                } else {
                    // Case 2: Current tensor is placed on a larger set of coordinates than tensor_coordinates.
                    TT_ASSERT(
                        is_subset_of(tensor_mesh_coords, tensor_coordinates),
                        "Tensors are placed on different MeshCoordinate ranges that do not intersect.");
                }
            }
        },
        tensor_args);
    return tensor_coordinates;
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
