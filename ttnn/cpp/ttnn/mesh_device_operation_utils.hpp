// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program_cache.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::mesh_device_operation_utils {

template <typename T>
using CachedMeshWorkload = tt::tt_metal::program_cache::detail::CachedMeshWorkload<T>;

// Determines if the device operation and a given program factory use heterogenous dispatch (constructing a single
// program for the entire mesh workload, or creating a program for each device within the mesh).
// For the old infra type-erased `DeviceOperation`, this is determined by `uses_heterogenous_dispatch` method on the
// program factory.
// For the new infra, this is determined by the presence of `create_at` method for creating programs at
// a specific location.
template <typename ConcreteProgramFactory, typename OperationAttributes>
auto uses_heterogenous_dispatch(const OperationAttributes& attrs) {
    if constexpr (requires { ConcreteProgramFactory::uses_heterogenous_dispatch(attrs); }) {
        return ConcreteProgramFactory::uses_heterogenous_dispatch(attrs);
    } else {
        constexpr bool has_create_at = requires { &ConcreteProgramFactory::create_at; };
        return has_create_at;
    }
}

// Returns true if all tensors have uniform storage, false otherwise.
template <typename TensorArgs>
bool all_tensors_have_uniform_storage(const TensorArgs& tensor_args) {
    Tensor first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const bool first_uniform = first_tensor.device_storage().is_uniform_storage();
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) {
            TT_FATAL(
                tensor.device_storage().is_uniform_storage() == first_uniform,
                "Expected either all or none of the tensors to have uniform storage.");
        },
        tensor_args);
    return first_uniform;
}

// Filters shards from `tensor_return_value` that are in `tensor_coordinates`.
template <typename TensorReturnValue>
void filter_tensor_shards(
    const std::vector<ttnn::MeshCoordinate>& tensor_coordinates, TensorReturnValue& tensor_return_value) {
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor_to_return) {
            auto& tensor_storage = std::get<tt::tt_metal::DeviceStorage>(tensor_to_return.tensor_attributes->storage);

            auto coord_it = tensor_coordinates.cbegin();
            auto storage_it = tensor_storage.specs.begin();
            auto insert_it = tensor_storage.specs.begin();
            while (coord_it != tensor_coordinates.end() && storage_it != tensor_storage.specs.end()) {
                if (storage_it->first == *coord_it) {
                    std::swap(*insert_it, *storage_it);
                    ++insert_it;
                    ++coord_it;
                    ++storage_it;
                } else {
                    ++storage_it;
                }
            }
            tensor_storage.specs.erase(insert_it, tensor_storage.specs.end());
        },
        tensor_return_value);
}

// Verifies all tensors span the same set of coordinates, and returns them in a vector.
template <typename TensorArgs>
std::vector<ttnn::MeshCoordinate> extract_tensor_coordinates(const TensorArgs& tensor_args) {
    Tensor first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    std::vector<ttnn::MeshCoordinate> tensor_coordinates;
    std::transform(
        first_tensor.device_storage().specs.begin(),
        first_tensor.device_storage().specs.end(),
        std::back_inserter(tensor_coordinates),
        [](const auto& spec) { return spec.first; });
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& tensor) {
            TT_FATAL(
                tensor.device_storage().specs.size() == tensor_coordinates.size(),
                "Tensors with non-uniform storage must have the same number of coordinates");
            auto tensor_coordinates_it = tensor_coordinates.begin();
            for (const auto& [coord, _] : tensor.device_storage().specs) {
                TT_FATAL(
                    coord == *tensor_coordinates_it, "Tensors with non-uniform storage must have the same coordinates");
                ++tensor_coordinates_it;
            }
        },
        tensor_args);
    return tensor_coordinates;
}

// Apply override_runtime_arguments in a uniform way across different program factory interfaces because
// we have many different program factory interfaces we're currently supporting.
template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
void apply_override_runtime_arguments(
    ProgramFactoryT& factory,
    tt::tt_metal::Program& program,
    typename ProgramFactoryT::shared_variables_t& shared_vars,
    const AttributesT& attrs,
    const ttnn::MeshCoordinate& coord,
    const TensorArgsT& tensor_args,
    ReturnValueT& return_value) {
    if constexpr (
        requires {
            typename ProgramFactoryT::cached_program_t;
            factory.override_runtime_arguments(
                std::declval<typename ProgramFactoryT::cached_program_t&>(), attrs, tensor_args, return_value);
        } ||
        requires {
            typename ProgramFactoryT::cached_program_t;
            factory.override_runtime_arguments_at(
                std::declval<typename ProgramFactoryT::cached_program_t&>(), attrs, coord, tensor_args, return_value);
        }) {
        typename ProgramFactoryT::cached_program_t cached_program{program, shared_vars};
        if constexpr (requires { &ProgramFactoryT::override_runtime_arguments_at; }) {
            factory.override_runtime_arguments_at(cached_program, attrs, coord, tensor_args, return_value);
        } else {
            factory.override_runtime_arguments(cached_program, attrs, tensor_args, return_value);
        }
    } else {
        if constexpr (requires { &ProgramFactoryT::override_runtime_arguments_at; }) {
            factory.override_runtime_arguments_at(program, shared_vars, attrs, coord, tensor_args, return_value);
        } else {
            factory.override_runtime_arguments(program, shared_vars, attrs, tensor_args, return_value);
        }
    }
}

template <
    typename DeviceOperationT,
    typename ConcreteProgramFactory,
    typename AttributesT,
    typename TensorArgsT,
    typename ReturnValueT>
CachedMeshWorkload<typename ConcreteProgramFactory::shared_variables_t> create_mesh_workload(
    ttnn::MeshDevice* mesh_device,
    const AttributesT& attrs,
    const TensorArgsT& tensor_args,
    ReturnValueT& tensor_return_value) {
    using shared_vars_t = typename ConcreteProgramFactory::shared_variables_t;
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    ConcreteProgramFactory program_factory;
    constexpr bool has_create = requires { &ConcreteProgramFactory::create; };

    std::unordered_map<ttnn::MeshCoordinateRange, shared_vars_t> coordinate_range_to_shared_variables;

    auto make_program = [&](const ttnn::MeshCoordinate& coord) {
        if constexpr (has_create) {
            return program_factory.create(attrs, tensor_args, tensor_return_value);
        } else {
            return program_factory.create_at(attrs, coord, tensor_args, tensor_return_value);
        }
    };

    // Fast path - create a single program for all devices.
    // No customization and uniform tensor storage spanning all devices.
    if (!uses_heterogenous_dispatch<ConcreteProgramFactory>(attrs) &&
        mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
        const ttnn::MeshCoordinateRange mesh_coordinate_range(mesh_device->shape());
        auto cached_program = make_program(ttnn::MeshCoordinate(0, 0));
        mesh_workload.add_program(mesh_coordinate_range, std::move(cached_program.program));
        coordinate_range_to_shared_variables[mesh_coordinate_range] = std::move(cached_program.shared_variables);
        return CachedMeshWorkload<shared_vars_t>{
            std::move(mesh_workload), std::move(coordinate_range_to_shared_variables)};
    } else {
        // Create separate programs for each device.
        for (const auto& coord : extract_tensor_coordinates(tensor_args)) {
            auto cached_program = make_program(coord);
            const ttnn::MeshCoordinateRange coordinate_range(coord, coord);
            mesh_workload.add_program(coordinate_range, std::move(cached_program.program));
            coordinate_range_to_shared_variables[coordinate_range] = std::move(cached_program.shared_variables);
        }
    }

    return CachedMeshWorkload<shared_vars_t>{std::move(mesh_workload), std::move(coordinate_range_to_shared_variables)};
}

template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
void override_mesh_runtime_arguments(
    CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t>& cached_workload,
    ttnn::MeshDevice* mesh_device,
    const AttributesT& attrs,
    const TensorArgsT& tensor_args,
    ReturnValueT& tensor_return_value) {
    ProgramFactoryT program_factory;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.coordinate_range_to_shared_variables[coordinate_range];

        for (const auto& coord : coordinate_range) {
            apply_override_runtime_arguments(
                program_factory, program, shared_variables, attrs, coord, tensor_args, tensor_return_value);
        }
    }
}

}  // namespace ttnn::mesh_device_operation_utils
