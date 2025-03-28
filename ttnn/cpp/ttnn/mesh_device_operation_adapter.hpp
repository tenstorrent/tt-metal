// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_cache.hpp>

#include <memory>
#include <optional>
#include <functional>
#include <concepts>
#include <variant>
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operation_concepts.hpp"

namespace ttnn::device_operation {

template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

/**
 * A generic adapter that adds mesh device capabilities to any existing device operation.
 * This adapter delegates to the base operation for standard functionality while providing
 * default implementations for mesh-specific operations.
 *
 * Usage:
 * 1. From an existing device operation, derive a new operation that uses this adapter
 * 2. The operation will now work correctly on mesh devices without additional code
 */
template <typename DeviceOperation>
struct MeshDeviceOperationAdapter {
    // Add type aliases to identify the template parameters
    using device_operation_t = DeviceOperation;

    // Inherit all typedefs from base operation
    using operation_attributes_t = typename DeviceOperation::operation_attributes_t;
    using tensor_args_t = typename DeviceOperation::tensor_args_t;
    using spec_return_value_t = typename DeviceOperation::spec_return_value_t;
    using tensor_return_value_t = typename DeviceOperation::tensor_return_value_t;
    using program_factory_t = typename DeviceOperation::program_factory_t;

    // Delegate to base operation methods
    static program_factory_t select_program_factory(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return DeviceOperation::select_program_factory(attrs, tensor_args);
    }

    template <typename... Args>
    static auto invoke(Args&&... args) {
        return DeviceOperation::invoke(std::forward<Args>(args)...);
    }

    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        DeviceOperation::validate_on_program_cache_hit(attrs, tensor_args);
    }

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        DeviceOperation::validate_on_program_cache_miss(attrs, tensor_args);
    }

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return DeviceOperation::compute_output_specs(attrs, tensor_args);
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return DeviceOperation::create_output_tensors(attrs, tensor_args);
    }

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else {
            return tt::stl::hash::hash_objects_with_default_seed(
                tt::stl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
        }
    }

    template <ProgramFactoryConcept ProgramFactory>
    static auto create_mesh_workload(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        ProgramFactory program_factory;
        using shared_vars_t = typename ProgramFactory::shared_variables_t;

        tt::tt_metal::distributed::MeshWorkload mesh_workload;
        std::unordered_map<ttnn::MeshCoordinateRange, shared_vars_t> shared_variables;

        auto make_program = [&](const ttnn::MeshCoordinate& coord) {
            if constexpr (requires { &ProgramFactory::create; }) {
                return program_factory.create(attrs, tensor_args, tensor_return_value);
            } else {
                return program_factory.create_at(attrs, coord, tensor_args, tensor_return_value);
            }
        };

        // Fast path - create a single program for all devices.
        // No customization and uniform tensor storage spanning all devices.
        if (!mesh_device_operation_utils::uses_heterogenous_dispatch<ProgramFactory>(attrs) &&
            mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
            const ttnn::MeshCoordinateRange mesh_coordinate_range(mesh_device->shape());
            auto cached_program = make_program(ttnn::MeshCoordinate(0, 0));
            mesh_workload.add_program(mesh_coordinate_range, std::move(cached_program.program));
            shared_variables[mesh_coordinate_range] = std::move(cached_program.shared_variables);
        } else {
            // Create separate programs for each device.
            tt::log_warning(
                tt::LogOp,
                "Tensors that are distributed across mesh device unevenly negatively affect Op dispatch performance.");
            for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
                auto cached_program = make_program(coord);
                const ttnn::MeshCoordinateRange coordinate_range(coord, coord);
                mesh_workload.add_program(coordinate_range, std::move(cached_program.program));
                shared_variables[coordinate_range] = std::move(cached_program.shared_variables);
            }
        }

        return AdaptedCachedMeshWorkload<shared_vars_t>{std::move(mesh_workload), std::move(shared_variables)};
    }

    template <ProgramFactoryConcept ProgramFactory>
    static void override_mesh_runtime_arguments(
        tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<typename ProgramFactory::shared_variables_t>&
            cached_workload,
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        ProgramFactory program_factory;

        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

            for (const auto& coord : coordinate_range) {
                mesh_device_operation_utils::apply_override_runtime_arguments(
                    program_factory,
                    program,
                    shared_variables,
                    attrs,
                    *(coordinate_range.begin()),
                    tensor_args,
                    tensor_return_value);
            }
        }
    }

    static tt::stl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        // Hash the program hash and the tensor coordinates the workload is targeting.
        auto hash = compute_program_hash(attrs, tensor_args);
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
            tt::utils::hash_combine(hash, coord);
        }
        return hash;
    }
};

template <typename T>
struct is_mesh_device_operation_adapter : std::false_type {};

template <typename DeviceOp>
struct is_mesh_device_operation_adapter<MeshDeviceOperationAdapter<DeviceOp>> : std::true_type {};

template <typename T>
inline constexpr bool is_mesh_device_operation_adapter_v = is_mesh_device_operation_adapter<T>::value;

/**
 * @brief Concept that defines a device operation that has a mesh device adapter.
 *
 * This concept requires that the type satisfies both the DeviceOperationConcept
 * and the MeshDeviceOperationAdapterType concept. It represents operations that
 * can be executed across multiple devices in a mesh configuration using the
 * adapter pattern.
 */
template <typename device_operation_t>
concept DeviceOperationWithMeshDeviceAdapter =
    DeviceOperationConcept<device_operation_t> && is_mesh_device_operation_adapter_v<device_operation_t>;

}  // namespace ttnn::device_operation
