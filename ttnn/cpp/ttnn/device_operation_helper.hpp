// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program_cache.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation {

template <typename T>
using CachedMeshWorkload = tt::tt_metal::program_cache::detail::CachedMeshWorkload<T>;

// This class is used to encapsulate default behavior for mesh device operations.
class MeshDeviceOperationHelper {
private:
    // Apply override_runtime_arguments in a uniform way across different program factory interfaces because
    // we have many different program factory interfaces we're currently supporting.
    template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
    static void apply_override_runtime_arguments(
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

public:
    template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
    static ttnn::device_operation::CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t>
    create_mesh_workload(
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

        // TODO: #19177 - Verify that all tensors target the same set of devices, and have the same tensor specs across
        // shards.
        const Tensor first_tensor = tt::stl::reflection::get_first_object_of_type<ttnn::Tensor>(tensor_args);

        if (per_device_attribute_customizer || !first_tensor.device_storage().is_uniform_storage()) {
            // Create separate programs for each device with customized attributes
            for (const auto& [coord, _] : first_tensor.device_storage().specs) {
                auto cached_program = program_factory.create(
                    per_device_attribute_customizer ? per_device_attribute_customizer(attrs, coord, mesh_device)
                                                    : attrs,
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

        return ttnn::device_operation::CachedMeshWorkload<shared_vars_t>{
            std::move(mesh_workload), std::move(coordinate_range_to_shared_variables)};
    }

    template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
    static void override_mesh_runtime_arguments(
        ttnn::device_operation::CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t>& cached_workload,
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
};

}  // namespace ttnn::device_operation
