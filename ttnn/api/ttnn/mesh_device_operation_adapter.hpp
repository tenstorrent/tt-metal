// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>

#include <memory>
#include <optional>
#include <functional>
#include <concepts>
#include <variant>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operation.hpp"
#include <tt_stl/reflection.hpp>

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

    // Returns type name of the underlying device operation.
    // Used for logging and debugging; in particular, Tracy profiler uses this to identify operations.
    static std::string get_type_name(const operation_attributes_t& /* attribute */) {
        return std::string(tt::stl::get_type_name<device_operation_t>());
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

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else {
            return tt::stl::hash::hash_objects_with_default_seed(
                tt::stl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
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

    // An adapter for creating a factory that abides to `MeshWorkloadFactoryConcept` out of
    // `ProgramDescriptorFactoryConcept` types.
    //
    // On cache miss: calls create_descriptor to build the Program, scans runtime args
    // to precompute address slot positions, and (for pure factories) verifies that
    // create_descriptor is deterministic by calling it a second time and comparing.
    //
    // On cache hit: patches the precomputed address slots with fresh buffer addresses.
    // If the factory provides an optional override_runtime_arguments(Program&, ...),
    // it is called AFTER address patching to update non-address runtime args (e.g. seeds).
    template <ProgramDescriptorFactoryConcept DescriptorFactory>
    struct DescriptorMeshWorkloadFactoryAdapter {
        struct AddressSlot {
            uint32_t kernel_handle;
            CoreCoord core;
            uint32_t arg_index;
            uint16_t buffer_id;  // index into the flat vector returned by collect_buffers
        };

        struct CBSlot {
            tt::tt_metal::CBHandle cb_handle;
            uint16_t buffer_id;  // index into the flat vector returned by collect_buffers
        };

        struct shared_variables_t {
            std::vector<AddressSlot> address_slots;
            std::vector<CBSlot> cb_slots;
            uint32_t num_kernel_handles{};
        };
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Detect whether the factory provides an optional override_runtime_arguments.
        static constexpr bool has_factory_override = requires(
            tt::tt_metal::Program& p,
            const operation_attributes_t& a,
            const tensor_args_t& t,
            tensor_return_value_t& r) { DescriptorFactory::override_runtime_arguments(p, a, t, r); };

        // Collect all buffer pointers from tensor_args and tensor_return_value in a
        // deterministic order (using the reflection visitor).  The same visitation order
        // on cache hit produces new buffers at the same indices.
        static std::vector<tt::tt_metal::Buffer*> collect_buffers(
            const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
            std::vector<tt::tt_metal::Buffer*> bufs;
            tt::stl::reflection::visit_object_of_type<Tensor>(
                [&](const Tensor& t) {
                    if (t.buffer()) {
                        bufs.push_back(t.buffer());
                    }
                },
                tensor_args);
            tt::stl::reflection::visit_object_of_type<Tensor>(
                [&](const Tensor& t) {
                    if (t.buffer()) {
                        bufs.push_back(t.buffer());
                    }
                },
                tensor_return_value);
            return bufs;
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

            // Collect current buffer pointers and build address-to-id / pointer-to-id maps.
            auto bufs = collect_buffers(tensor_args, tensor_return_value);
            std::unordered_map<uint32_t, uint16_t> addr_to_id;
            std::unordered_map<tt::tt_metal::Buffer*, uint16_t> buf_to_id;
            bool has_addr_collision = false;
            for (uint16_t i = 0; i < static_cast<uint16_t>(bufs.size()); ++i) {
                auto [it, inserted] = addr_to_id.emplace(bufs[i]->address(), i);
                if (!inserted) {
                    has_addr_collision = true;
                }
                buf_to_id.emplace(bufs[i], i);
            }

            for (const auto& range : tensor_coords.ranges()) {
                auto desc = DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value);
                uint32_t num_kernels = static_cast<uint32_t>(desc.kernels.size());

                // --- Runtime arg address slots ---
                std::vector<AddressSlot> slots;
                if (!has_addr_collision) {
                    // Scan the descriptor's runtime args to find buffer address positions.
                    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
                        const auto& kernel_desc = desc.kernels[k];
                        for (const auto& [core, args] : kernel_desc.runtime_args) {
                            for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                                auto it = addr_to_id.find(args[i]);
                                if (it != addr_to_id.end()) {
                                    slots.push_back({k, core, i, it->second});
                                }
                            }
                        }
                    }
                }

                // For pure descriptor factories (no override_runtime_arguments),
                // verify determinism now: call create_descriptor a second time and
                // compare non-address runtime args.  Any difference means the factory
                // has non-deterministic state (e.g. random seeds) that would go stale
                // on cache hits.
                if constexpr (!has_factory_override) {
                    std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> slot_positions;
                    for (const auto& slot : slots) {
                        slot_positions.emplace(slot.kernel_handle, slot.core.x, slot.core.y, slot.arg_index);
                    }

                    auto desc2 = DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value);
                    for (uint32_t k = 0; k < num_kernels; ++k) {
                        const auto& k1 = desc.kernels[k];
                        const auto& k2 = desc2.kernels[k];
                        for (size_t ci = 0; ci < k1.runtime_args.size(); ++ci) {
                            const auto& [core1, args1] = k1.runtime_args[ci];
                            const auto& [core2, args2] = k2.runtime_args[ci];
                            for (uint32_t ai = 0; ai < static_cast<uint32_t>(args1.size()); ++ai) {
                                TT_FATAL(
                                    args1[ai] == args2[ai] || slot_positions.count({k, core1.x, core1.y, ai}),
                                    "ProgramDescriptorFactory produces non-deterministic runtime "
                                    "args (kernel={}, core=({},{}), arg_index={}, values {} vs "
                                    "{}). create_descriptor must be deterministic: only buffer "
                                    "addresses may vary between calls. If this factory "
                                    "intentionally produces changing non-address runtime args "
                                    "(e.g. random seeds), add a static "
                                    "override_runtime_arguments(Program&, operation_attributes_t, "
                                    "tensor_args_t, tensor_return_value_t) method to the factory "
                                    "to update them explicitly on each cache hit.",
                                    k,
                                    core1.x,
                                    core1.y,
                                    ai,
                                    args1[ai],
                                    args2[ai]);
                            }
                        }
                    }
                }

                tt::tt_metal::Program program{desc};

                // --- Dynamic circular buffer slots ---
                // Scan descriptor CBs for those backed by tensor buffers (dynamic CBs).
                // The Program constructor creates CBs in the same order as desc.cbs,
                // so we can match by index to obtain the opaque CBHandle.
                std::vector<CBSlot> cb_slots;
                {
                    auto program_cbs = program.circular_buffers();
                    for (uint32_t ci = 0; ci < static_cast<uint32_t>(desc.cbs.size()); ++ci) {
                        if (desc.cbs[ci].buffer) {
                            auto it = buf_to_id.find(desc.cbs[ci].buffer);
                            if (it != buf_to_id.end()) {
                                cb_slots.push_back({program_cbs[ci]->id(), it->second});
                            }
                        }
                    }
                }

                mesh_workload.add_program(range, std::move(program));
                shared_variables[range] = shared_variables_t{
                    .address_slots = std::move(slots),
                    .cb_slots = std::move(cb_slots),
                    .num_kernel_handles = num_kernels,
                };
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                const auto& sv = cached_workload.shared_variables.at(coordinate_range);

                // Collect all tensor buffers (used for both runtime arg and CB patching).
                auto bufs = collect_buffers(tensor_args, tensor_return_value);

                // Patch precomputed runtime arg address slots.
                for (const auto& slot : sv.address_slots) {
                    auto& args = tt::tt_metal::GetRuntimeArgs(program, slot.kernel_handle, slot.core);
                    args[slot.arg_index] = bufs[slot.buffer_id]->address();
                }

                // Patch dynamic circular buffer addresses.
                for (const auto& cb_slot : sv.cb_slots) {
                    tt::tt_metal::UpdateDynamicCircularBufferAddress(
                        program, cb_slot.cb_handle, *bufs[cb_slot.buffer_id]);
                }

                // If the factory handles non-address runtime args, call it.
                if constexpr (has_factory_override) {
                    DescriptorFactory::override_runtime_arguments(program, attrs, tensor_args, tensor_return_value);
                }
            }
        }
    };

    static tt::stl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        // Three-tier hash logic:
        //  1. If the operation provides a custom compute_program_hash, use it.
        //  2. If all factories are ProgramDescriptorFactoryConcept, auto-derive
        //     from create_descriptor + compute_program_descriptor_hash.
        //  3. Otherwise, fall back to the default hash of type + attrs + tensor_args.
        tt::stl::hash::hash_t hash;

        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            hash = DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else if constexpr ([]<typename... Ts>(std::variant<Ts...>*) {
                                 return (ProgramDescriptorFactoryConcept<Ts> && ...);
                             }(static_cast<program_factory_t*>(nullptr))) {
            // All factories are descriptor-based: auto-hash from the descriptor's
            // compile-time parts (kernel sources, core ranges, compile-time args,
            // defines, CB configs).  Runtime args are excluded automatically.
            auto factory = DeviceOperation::select_program_factory(attrs, tensor_args);
            hash = std::visit(
                [&](auto& f) -> tt::stl::hash::hash_t {
                    using F = std::decay_t<decltype(f)>;
                    auto desc = F::create_descriptor(attrs, tensor_args, tensor_return_value);
                    return tt::tt_metal::compute_program_descriptor_hash(desc);
                },
                factory);
        } else {
            hash = tt::stl::hash::hash_objects_with_default_seed(
                tt::stl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
        }

        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            ttsl::hash::hash_combine(hash, coord);
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
