// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include <memory>
#include <optional>
#include <concepts>
#include <variant>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operation.hpp"
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"

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

private:
    struct DirectDescriptorFactory {
        static auto create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            return DeviceOperation::create_descriptor(attrs, tensor_args, tensor_return_value);
        }
    };

    template <typename T, typename = void>
    struct resolve_program_factory {
        using type = std::variant<DirectDescriptorFactory>;
    };
    template <typename T>
    struct resolve_program_factory<T, std::void_t<typename T::program_factory_t>> {
        using type = typename T::program_factory_t;
    };

public:
    using program_factory_t = typename resolve_program_factory<DeviceOperation>::type;

    static_assert(
        HasDirectDescriptor<DeviceOperation> || HasSelectProgramFactory<DeviceOperation> ||
            std::variant_size_v<program_factory_t> == 1,
        "DeviceOperation must implement select_program_factory when program_factory_t has more than one type.");

    static program_factory_t select_program_factory(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (HasDirectDescriptor<DeviceOperation>) {
            return program_factory_t{DirectDescriptorFactory{}};
        } else if constexpr (HasSelectProgramFactory<DeviceOperation>) {
            return DeviceOperation::select_program_factory(attrs, tensor_args);
        } else {
            return program_factory_t{std::variant_alternative_t<0, program_factory_t>{}};
        }
    }

    template <typename... Args>
    static auto invoke(Args&&... args) {
        return DeviceOperation::invoke(std::forward<Args>(args)...);
    }

    // Returns type name of the underlying device operation.
    // Used for logging and debugging; in particular, Tracy profiler uses this to identify operations.
    static std::string get_type_name(const operation_attributes_t& /* attribute */) {
        return std::string(ttsl::get_type_name<device_operation_t>());
    }

    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (HasValidateOnProgramCacheHit<DeviceOperation>) {
            DeviceOperation::validate_on_program_cache_hit(attrs, tensor_args);
        } else {
            DeviceOperation::validate_on_program_cache_miss(attrs, tensor_args);
        }
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

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else {
            return ttsl::hash::hash_objects_with_default_seed(
                ttsl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
        }
    }

    // An adapter for creating a factory that abides to `MeshWorkloadFactoryConcept` out of `ProgramFactoryConcept`
    // types.
    template <ProgramFactoryConcept ProgramFactory>
    struct MeshWorkloadFactoryAdapter {
        using shared_variables_t = typename ProgramFactory::shared_variables_t;
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            ProgramFactory program_factory;

            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            for (const auto& range : tensor_coords.ranges()) {
                auto cached_program = program_factory.create(attrs, tensor_args, tensor_return_value);
                mesh_workload.add_program(range, std::move(cached_program.program));
                shared_variables[range] = std::move(cached_program.shared_variables);
            }

            return AdaptedCachedMeshWorkload<shared_variables_t>{std::move(mesh_workload), std::move(shared_variables)};
        }

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            ProgramFactory program_factory;

            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

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
    };

    // -----------------------------------------------------------------------
    // DescriptorMeshWorkloadFactoryAdapter
    //
    // Adapts a ProgramDescriptorFactoryConcept factory for mesh dispatch.
    // The developer writes ONLY create_descriptor (and optionally
    // prepare_resources).
    //
    // The descriptor is created fresh on every dispatch.  On cache miss
    // the framework builds a Program from it;
    // on cache hit the generic apply_descriptor_runtime_args() copies all
    // runtime args from the fresh descriptor into the cached Program.
    // No override_runtime_arguments, no address scanning, no patching logic.
    // -----------------------------------------------------------------------
    template <ProgramDescriptorFactoryConcept DescriptorFactory>
    struct DescriptorMeshWorkloadFactoryAdapter {
        // --- Optional hook detection ---

        static constexpr bool has_prepare_resources =
            requires(const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& r) {
                DescriptorFactory::prepare_resources(a, t, r);
            };

        struct empty_resource_t {};

        // Deduce the return type of prepare_resources when it exists; fall back to
        // an empty resource otherwise.
        template <typename T, bool HasPrepareResources = false>
        struct deduce_resource_type {
            using type = empty_resource_t;
        };
        template <typename T>
        struct deduce_resource_type<T, true> {
            using type = decltype(T::prepare_resources(
                std::declval<const operation_attributes_t&>(),
                std::declval<const tensor_args_t&>(),
                std::declval<tensor_return_value_t&>()));
        };

        using resource_t = typename deduce_resource_type<DescriptorFactory, has_prepare_resources>::type;

        struct shared_variables_t {
            [[no_unique_address]] resource_t resources{};
            // Resolved buffer bindings for the fast cache-hit path.
            // Non-empty when the factory used emplace_runtime_args() with Buffer* args.
            tt::tt_metal::ResolvedBindings resolved_bindings;
        };
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Enumerate all Buffer* reachable from tensor_args and tensor_return_value,
        // in a stable field-declaration order via reflection.  Used to map buffer
        // bindings to indices that survive across calls without storing raw pointers.
        static std::vector<tt::tt_metal::Buffer*> collect_tensor_buffers(
            const tensor_args_t& tensor_args, const tensor_return_value_t& tensor_return_value) {
            std::vector<tt::tt_metal::Buffer*> buffers;
            auto collect = [&buffers](const Tensor& t) { buffers.push_back(t.buffer()); };
            ttsl::reflection::visit_object_of_type<Tensor>(collect, tensor_args);
            ttsl::reflection::visit_object_of_type<Tensor>(collect, tensor_return_value);
            return buffers;
        }

        static tt::tt_metal::ProgramDescriptor invoke_create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            resource_t& resources) {
            if constexpr (has_prepare_resources) {
                return DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value, resources);
            } else {
                return DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value);
            }
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

            for (const auto& range : tensor_coords.ranges()) {
                resource_t resources{};
                if constexpr (has_prepare_resources) {
                    resources = DescriptorFactory::prepare_resources(attrs, tensor_args, tensor_return_value);
                }

                auto desc = invoke_create_descriptor(attrs, tensor_args, tensor_return_value, resources);
                tt::tt_metal::Program program{desc};
                auto tensor_buffers = collect_tensor_buffers(tensor_args, tensor_return_value);
                auto resolved = tt::tt_metal::resolve_bindings(program, desc, tensor_buffers);
                mesh_workload.add_program(range, std::move(program));
                shared_variables[range] =
                    shared_variables_t{.resources = std::move(resources), .resolved_bindings = std::move(resolved)};
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto& sv = cached_workload.shared_variables.at(coordinate_range);
                if (!sv.resolved_bindings.empty()) {
                    // Fast path: patch only the buffer positions using current tensor addresses.
                    // No create_descriptor() call — tensor_buffers enumeration is O(n_tensors).
                    auto current_buffers = collect_tensor_buffers(tensor_args, tensor_return_value);
                    tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, current_buffers);
                } else {
                    // Slow path: full descriptor rebuild + bulk copy.
                    // Used by factories that have not yet adopted emplace_runtime_args().
                    auto desc = invoke_create_descriptor(attrs, tensor_args, tensor_return_value, sv.resources);
                    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
                }
            }
        }
    };

    static ttsl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        ttsl::hash::hash_t hash;

        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            hash = DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else {
            hash =
                ttsl::hash::hash_objects_with_default_seed(ttsl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
        }

        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
        return hash;
    }

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        if constexpr (requires {
                          device_operation_t::create_op_performance_model(attributes, tensor_args, tensor_return_value);
                      }) {
            // Custom Performance Model detected for this Op
            return device_operation_t::create_op_performance_model(attributes, tensor_args, tensor_return_value);
        } else {
            // Use generic Op Performance Models
            if constexpr (requires { tensor_args.input_tensors; }) {
                // tensor_args_t for Op contains input_tensors attribute
                return tt::tt_metal::operation::OpPerformanceModelGeneral(
                    tensor_args.input_tensors,
                    tensor_return_value,
                    1 /* ideal_compute_cycles: specify as 1, since op perf model is not provided*/);
            } else {
                // tensor_args_t does not have input_tensors, use default performance model
                return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>{};
            }
        }
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
