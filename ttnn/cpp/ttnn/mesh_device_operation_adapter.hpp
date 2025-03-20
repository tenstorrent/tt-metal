// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <functional>
#include <concepts>
#include <tt-metalium/program_cache.hpp>
#include "ttnn/mesh_device_operation_utils.hpp"

namespace ttnn {

template <typename AttributeType>
struct AttributeCustomizerBase {
    virtual ~AttributeCustomizerBase() = default;
    virtual AttributeType customize(
        const AttributeType& attrs,
        const tt::tt_metal::distributed::MeshCoordinate& coordinate,
        tt::tt_metal::distributed::MeshDevice* mesh_device) const = 0;
};

// Define concept for customizer functions - ensures type safety
template <typename F, typename AttributeType>
concept AttributeCustomizerCallable = requires(
    F f,
    const AttributeType& attrs,
    const tt::tt_metal::distributed::MeshCoordinate& coord,
    tt::tt_metal::distributed::MeshDevice* device) {
    { f(attrs, coord, device) } -> std::convertible_to<AttributeType>;
};

// Template class that adapts any callable to the AttributeCustomizerBase interface
template <typename AttributeType, AttributeCustomizerCallable<AttributeType> F>
struct FunctionAttributeCustomizer : public AttributeCustomizerBase<AttributeType> {
    FunctionAttributeCustomizer(F func) : func_(std::move(func)) {}

    AttributeType customize(
        const AttributeType& attrs,
        const tt::tt_metal::distributed::MeshCoordinate& coord,
        tt::tt_metal::distributed::MeshDevice* device) const override {
        return func_(attrs, coord, device);
    }

private:
    F func_;
};

// Helper function to create a customizer from any callable
template <typename AttributeType, AttributeCustomizerCallable<AttributeType> F>
std::shared_ptr<AttributeCustomizerBase<AttributeType>> make_attribute_customizer(F func) {
    return std::make_shared<FunctionAttributeCustomizer<AttributeType, F>>(std::move(func));
}

/**
 * A generic adapter that adds mesh device capabilities to any existing device operation.
 * This adapter delegates to the base operation for standard functionality while providing
 * default implementations for mesh-specific operations.
 *
 * Usage:
 * 1. From an existing device operation, derive a new operation that uses this adapter
 * 2. The operation will now work correctly on mesh devices without additional code
 * 3. (Optional) Provide a custom attribute customizer to the adapter by specifying it as the CustomizerType:
 *    e.g., using MyMeshOperation = MeshDeviceOperationAdapter<MyDeviceOperation, MyCustomizer>;
 *    This will be used to customize the attributes for each device in the mesh, which in turn allows
 *    different programs per-device in the mesh.
 * 4. (Optional) Use set_attribute_customizer to provide a dynamic customizer at runtime
 */
template <typename DeviceOperation, typename CustomizerType = void>
struct MeshDeviceOperationAdapter {
    // Add type aliases to identify the template parameters
    using device_operation_t = DeviceOperation;
    using attribute_customizer_t = CustomizerType;

    // Inherit all typedefs from base operation
    using operation_attributes_t = typename DeviceOperation::operation_attributes_t;
    using tensor_args_t = typename DeviceOperation::tensor_args_t;
    using spec_return_value_t = typename DeviceOperation::spec_return_value_t;
    using tensor_return_value_t = typename DeviceOperation::tensor_return_value_t;
    using program_factory_t = typename DeviceOperation::program_factory_t;

    // Extension point for customizing attribute transformation
    // Using the base class from device_operation_helper.hpp to avoid circular dependencies
    using AttributeCustomizer = ttnn::AttributeCustomizerBase<operation_attributes_t>;

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

    static auto create_mesh_workload(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        auto customizer_func = create_attribute_customizer_function();
        auto factory = select_program_factory(attrs, tensor_args);

        return std::visit(
            [&](auto&& concrete_factory) {
                using ConcreteFactory = std::decay_t<decltype(concrete_factory)>;
                using concrete_shared_vars_t = typename ConcreteFactory::shared_variables_t;

                return mesh_device_operation_utils::create_mesh_workload<ConcreteFactory>(
                    mesh_device, attrs, tensor_args, tensor_return_value, customizer_func);
            },
            factory);
    }

    template <typename ConcreteWorkload>
    static void override_mesh_runtime_arguments(
        ConcreteWorkload& cached_workload,
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        auto customizer_func = create_attribute_customizer_function();
        auto factory = select_program_factory(attrs, tensor_args);

        std::visit(
            [&](auto&& concrete_factory) {
                using ConcreteFactory = std::decay_t<decltype(concrete_factory)>;
                mesh_device_operation_utils::override_mesh_runtime_arguments<ConcreteFactory>(
                    cached_workload, mesh_device, attrs, tensor_args, tensor_return_value, customizer_func);
            },
            factory);
    }

    static tt::stl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        if (auto customizer_func = create_attribute_customizer_function(); customizer_func != nullptr) {
            auto hash = tt::stl::hash::hash_t{0};
            for (auto coordinate : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
                auto device_attrs = customizer_func(attrs, coordinate, mesh_device);
                tt::utils::hash_combine(hash, compute_program_hash(device_attrs, tensor_args));
            }
            return hash;
        } else {
            return compute_program_hash(attrs, tensor_args);
        }
    }

public:
    // Method to set a dynamic customizer
    static void set_attribute_customizer(std::shared_ptr<AttributeCustomizer> customizer) {
        get_dynamic_customizer() = std::move(customizer);
    }

    // Reset to the default customizer
    static void reset_attribute_customizer() { get_dynamic_customizer() = nullptr; }

private:
    // Thread-local storage for dynamic customizer
    static std::shared_ptr<AttributeCustomizer>& get_dynamic_customizer() {
        thread_local static std::shared_ptr<AttributeCustomizer> dynamic_customizer;
        return dynamic_customizer;
    }

    static std::shared_ptr<AttributeCustomizer> get_attribute_customizer() {
        // Check for dynamic customizer first
        if (auto& dynamic_customizer = get_dynamic_customizer()) {
            return dynamic_customizer;
        }

        // Fall back to static customizer if specified
        if constexpr (!std::is_same_v<CustomizerType, void>) {
            // Create a shared pointer of CustomizerType and then cast it to AttributeCustomizer
            return std::static_pointer_cast<
                typename MeshDeviceOperationAdapter<DeviceOperation, CustomizerType>::AttributeCustomizer>(
                std::make_shared<CustomizerType>());
        }
        return nullptr;
    }

    static auto create_attribute_customizer_function() {
        using customizer_function_t = std::function<operation_attributes_t(
            const operation_attributes_t&,
            const tt::tt_metal::distributed::MeshCoordinate&,
            tt::tt_metal::distributed::MeshDevice*)>;

        customizer_function_t customizer_func;
        if (auto customizer = get_attribute_customizer()) {
            customizer_func = [customizer](
                                  const operation_attributes_t& attrs,
                                  const tt::tt_metal::distributed::MeshCoordinate& coord,
                                  tt::tt_metal::distributed::MeshDevice* device) {
                return customizer->customize(attrs, coord, device);
            };
        }
        return customizer_func;
    }
};

template <typename T>
concept MeshDeviceOperationAdapterType = requires {
    typename T::device_operation_t;
    typename T::attribute_customizer_t;
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
        tt::tt_metal::distributed::MeshDevice* mesh_device) {
        T::create_mesh_workload(mesh_device, attrs, tensor_args, tensor_return_value);
        T::compute_mesh_workload_hash(mesh_device, attrs, tensor_args);
    };
};

template <typename T>
struct is_mesh_device_operation_adapter : std::false_type {};

template <typename DeviceOp, typename Customizer>
struct is_mesh_device_operation_adapter<MeshDeviceOperationAdapter<DeviceOp, Customizer>> : std::true_type {};

}  // namespace ttnn
