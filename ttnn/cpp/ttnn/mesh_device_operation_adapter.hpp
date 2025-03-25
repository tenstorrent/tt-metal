// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <functional>
#include <concepts>
#include <variant>
#include <tt-metalium/program_cache.hpp>
#include "ttnn/mesh_device_operation_utils.hpp"
#include "tools/profiler/op_profiler.hpp"

namespace ttnn {

namespace {
template <typename... Ts>
[[nodiscard]] std::variant<Ts...> map_index_to_variant(std::size_t i, std::variant<Ts...>) {
    assert(i < sizeof...(Ts));
    static constexpr std::variant<Ts...> table[] = {Ts{}...};
    return table[i];
}
}  // namespace

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

    template <
        typename DeviceOperationT,
        typename ConcreteProgramFactory,
        typename AttributesT,
        typename TensorArgsT,
        typename ReturnValueT>
    static tt::tt_metal::program_cache::detail::CachedMeshWorkload<typename ConcreteProgramFactory::shared_variables_t>
    create_mesh_workload(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const AttributesT& attrs,
        const TensorArgsT& tensor_args,
        ReturnValueT& tensor_return_value) {
        return std::visit(
            [&]<typename ProgramFactoryT>(const ProgramFactoryT&) {
                return ttnn::mesh_device_operation_utils::
                    template create_mesh_workload<DeviceOperation, ProgramFactoryT>(
                        mesh_device, attrs, tensor_args, tensor_return_value);
            },
            select_program_factory(attrs, tensor_args));
    }

    template <typename ProgramFactoryT, typename AttributesT, typename TensorArgsT, typename ReturnValueT>
    static void override_mesh_runtime_arguments(
        tt::tt_metal::program_cache::detail::CachedMeshWorkload<typename ProgramFactoryT::shared_variables_t>&
            cached_workload,
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const AttributesT& attrs,
        const TensorArgsT& tensor_args,
        ReturnValueT& tensor_return_value) {
        std::visit(
            [&]<typename ConcreteProgramFactory>(const ConcreteProgramFactory&) {
                mesh_device_operation_utils::override_mesh_runtime_arguments<ProgramFactoryT>(
                    cached_workload, mesh_device, attrs, tensor_args, tensor_return_value);
            },
            select_program_factory(attrs, tensor_args));
    }

    static tt::stl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        return compute_program_hash(attrs, tensor_args);
    }
};

template <typename T>
concept MeshDeviceOperationAdapterType = requires {
    typename T::device_operation_t;
    typename T::operation_attributes_t;
    typename T::tensor_args_t;
    typename T::spec_return_value_t;
    typename T::tensor_return_value_t;
    typename T::program_factory_t;

    // Check for the existence of key mesh-related methods
    requires requires(
        typename T::operation_attributes_t attrs,
        typename T::tensor_args_t tensor_args,
        typename T::tensor_return_value_t tensor_return_value,
        ttnn::MeshDevice* mesh_device) {
        { T::compute_mesh_workload_hash(mesh_device, attrs, tensor_args) } -> std::same_as<size_t>;
    };
};

template <typename T>
struct is_mesh_device_operation_adapter : std::false_type {};

template <typename DeviceOp>
struct is_mesh_device_operation_adapter<MeshDeviceOperationAdapter<DeviceOp>> : std::true_type {};

}  // namespace ttnn
