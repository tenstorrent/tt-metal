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

// If all tensors span the entire mesh of devices, returns nullopt.
// Otherwise, verifies all tensors span the same set of coordinates, and returns them in a vector.
template <typename TensorArgs>
std::optional<std::vector<ttnn::MeshCoordinate>> extract_tensor_coordinates(const TensorArgs& tensor_args) {
    Tensor first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    const bool expect_uniform_storage = first_tensor.device_storage().is_uniform_storage();
    if (expect_uniform_storage) {
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&](const Tensor& tensor) {
                TT_FATAL(tensor.device_storage().is_uniform_storage(), "Expected all tensors to have uniform storage.");
            },
            tensor_args);
        return std::nullopt;
    }

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
    const TensorArgsT& tensor_args,
    ReturnValueT& return_value) {
    if constexpr (requires {
                      typename ProgramFactoryT::cached_program_t;
                      factory.override_runtime_arguments(
                          std::declval<typename ProgramFactoryT::cached_program_t&>(),
                          attrs,
                          tensor_args,
                          return_value);
                  }) {
        typename ProgramFactoryT::cached_program_t cached_program{program, shared_vars};
        factory.override_runtime_arguments(cached_program, attrs, tensor_args, return_value);
    } else {
        factory.override_runtime_arguments(program, shared_vars, attrs, tensor_args, return_value);
    }
}

template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t> create_mesh_workload(
    ttnn::MeshDevice* mesh_device,
    const AttributesT& attrs,
    const TensorArgsT& tensor_args,
    ReturnValueT& tensor_return_value,
    std::function<AttributesT(const AttributesT&, const ttnn::MeshCoordinate&, ttnn::MeshDevice*)>
        per_device_attribute_customizer = nullptr) {
    using shared_vars_t = typename ProgramFactoryT::shared_variables_t;
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    ProgramFactoryT program_factory;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_vars_t> coordinate_range_to_shared_variables;

    auto tensor_coordinates = extract_tensor_coordinates(tensor_args);
    if (per_device_attribute_customizer || tensor_coordinates.has_value()) {
        // Create separate programs for each device with customized attributes
        if (!tensor_coordinates.has_value()) {
            std::vector<ttnn::MeshCoordinate> coords;
            coords.reserve(mesh_device->shape().mesh_size());
            const ttnn::MeshCoordinateRange r(mesh_device->shape());
            std::transform(r.begin(), r.end(), std::back_inserter(coords), [](const auto& c) { return c; });
            tensor_coordinates = std::move(coords);
        }
        for (const auto& coord : *tensor_coordinates) {
            auto cached_program = program_factory.create(
                per_device_attribute_customizer ? per_device_attribute_customizer(attrs, coord, mesh_device) : attrs,
                tensor_args,
                tensor_return_value);
            const ttnn::MeshCoordinateRange coordinate_range(coord, coord);
            mesh_workload.add_program(coordinate_range, std::move(cached_program.program));
            coordinate_range_to_shared_variables[coordinate_range] = std::move(cached_program.shared_variables);
        }
    } else {
        // Create a single program for all devices
        const ttnn::MeshCoordinateRange mesh_coordinate_range(mesh_device->shape());
        auto cached_program = program_factory.create(attrs, tensor_args, tensor_return_value);

        mesh_workload.add_program(mesh_coordinate_range, std::move(cached_program.program));
        coordinate_range_to_shared_variables[mesh_coordinate_range] = std::move(cached_program.shared_variables);
    }

    return CachedMeshWorkload<shared_vars_t>{std::move(mesh_workload), std::move(coordinate_range_to_shared_variables)};
}

template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
void override_mesh_runtime_arguments(
    CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t>& cached_workload,
    ttnn::MeshDevice* mesh_device,
    const AttributesT& attrs,
    const TensorArgsT& tensor_args,
    ReturnValueT& tensor_return_value,
    std::function<AttributesT(const AttributesT&, const ttnn::MeshCoordinate&, ttnn::MeshDevice*)>
        attribute_customizer = nullptr) {
    ProgramFactoryT program_factory;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.coordinate_range_to_shared_variables[coordinate_range];

        if (attribute_customizer) {
            for (auto coordinate : coordinate_range) {
                auto device_attrs = attribute_customizer(attrs, coordinate, mesh_device);

                apply_override_runtime_arguments(
                    program_factory, program, shared_variables, device_attrs, tensor_args, tensor_return_value);
            }
        } else {
            apply_override_runtime_arguments(
                program_factory, program, shared_variables, attrs, tensor_args, tensor_return_value);
        }
    }
}

}  // namespace ttnn::mesh_device_operation_utils
