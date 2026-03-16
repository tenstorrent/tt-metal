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

// Detect if operation_attributes_t has a uint32_t seed field.
// When true, the framework automatically:
//   1. Excludes seed from the program hash (so different seeds share cached programs)
//   2. Patches compute kernel runtime_args[0] with (attrs.seed + core_index) on cache hits
template <typename T>
concept HasSeed = requires(const T& t) {
    { t.seed } -> std::convertible_to<uint32_t>;
};

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

    // Early validation: if program_factory_t has more than one variant alternative,
    // the operation must provide select_program_factory to choose between them.
    static_assert(
        HasSelectProgramFactory<DeviceOperation> || std::variant_size_v<program_factory_t> == 1,
        "DeviceOperation must implement select_program_factory when program_factory_t has more than one type. "
        "For single-variant program_factory_t, the framework auto-selects it.");

    // Delegate to base operation methods.
    // Uses the framework helper: if the operation provides select_program_factory, it is called;
    // otherwise, for single-variant program_factory_t, returns the sole type automatically.
    static program_factory_t select_program_factory(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (HasSelectProgramFactory<DeviceOperation>) {
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
        } else if constexpr (HasSeed<operation_attributes_t>) {
            auto attrs_copy = attrs;
            attrs_copy.seed = 0;
            return ttsl::hash::hash_objects_with_default_seed(
                ttsl::hash::type_hash<DeviceOperation>, attrs_copy, tensor_args);
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
    // Adapts a ProgramDescriptorFactoryConcept factory into a full
    // MeshWorkloadFactoryConcept.  The developer writes ONLY:
    //
    //   1. create_descriptor(attrs, tensor_args, output)
    //        -- Builds the declarative ProgramDescriptor.
    //        -- Must be deterministic given the same inputs.
    //
    //   2. prepare_resources(attrs, tensor_args, output)   [OPTIONAL]
    //        -- Use ONLY when create_descriptor needs a device-side resource
    //           (e.g. config tensor) that isn't in tensor_args or the output.
    //        -- Called once on cache miss.  Its return value is passed to
    //           create_descriptor and kept alive across cache hits.
    //        -- Most factories do NOT need this.
    //
    //   3. post_create_validation(Program&, attrs, tensor_args, output)  [OPTIONAL]
    //        -- Called once on cache miss after the Program is constructed from the
    //           descriptor.  Use for sanity checks that require the materialized
    //           Program (e.g. verifying CB sizes match expected L1 usage).
    //
    // Automatic seed handling: if operation_attributes_t has a uint32_t seed
    // field (detected via HasSeed concept), the framework automatically patches
    // compute kernel runtime_args[0] with (attrs.seed + core_index) on cache
    // hits.  No developer code needed -- just put seed in your attrs and use
    // (seed + i) at arg[0] in create_descriptor.
    //
    // The framework handles everything else: address slot scanning, dynamic
    // CB patching, seed patching, cache-hit dispatch, and determinism verification.
    // -----------------------------------------------------------------------
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

        struct ComputeKernelCores {
            uint32_t kernel_handle;
            std::vector<CoreCoord> cores;
        };

        // --- Optional hook detection ---

        // Detect prepare_resources: returns a resource object that create_descriptor
        // needs (e.g. DeviceStorage for config tensors).  The return value is stored
        // in shared_variables for lifetime management across cache hits.
        static constexpr bool has_prepare_resources =
            requires(const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& r) {
                DescriptorFactory::prepare_resources(a, t, r);
            };

        // Resource type: what prepare_resources returns, or monostate for factories
        // that don't use it (zero storage overhead via [[no_unique_address]]).
        // Uses lazy evaluation via partial specialization to avoid evaluating the
        // decltype when prepare_resources doesn't exist.
    private:
        template <bool HasHook, typename = void>
        struct ResourceTypeHelper {
            using type = std::monostate;
        };
        template <typename Dummy>
        struct ResourceTypeHelper<true, Dummy> {
            using type = decltype(DescriptorFactory::prepare_resources(
                std::declval<const operation_attributes_t&>(),
                std::declval<const tensor_args_t&>(),
                std::declval<tensor_return_value_t&>()));
        };

    public:
        using resource_t = typename ResourceTypeHelper<has_prepare_resources>::type;

        static constexpr bool has_seed = HasSeed<operation_attributes_t>;

        // Detect post_create_validation: called once on cache miss after the Program
        // is constructed from the descriptor.  Use for sanity checks that require
        // the materialized Program (e.g. CB size verification).
        static constexpr bool has_post_create_validation = requires(
            tt::tt_metal::Program& p,
            const operation_attributes_t& a,
            const tensor_args_t& t,
            tensor_return_value_t& r) { DescriptorFactory::post_create_validation(p, a, t, r); };

        struct shared_variables_t {
            std::vector<AddressSlot> address_slots;
            std::vector<CBSlot> cb_slots;
            std::vector<ComputeKernelCores> compute_kernel_cores;
            uint32_t num_kernel_handles{};
            // Keeps prepare_resources return value alive across cache hits.
            // Zero-cost (empty base optimization) when resource_t is monostate.
            [[no_unique_address]] resource_t resources{};
        };
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Collect all buffer pointers from tensor_args and tensor_return_value in a
        // deterministic order (using the reflection visitor).  The same visitation order
        // on cache hit produces new buffers at the same indices.
        static std::vector<tt::tt_metal::Buffer*> collect_buffers(
            const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
            std::vector<tt::tt_metal::Buffer*> bufs;
            // buffer() returns a valid pointer or throws; optional tensors are
            // already skipped by the reflection visitor's std::optional handling.
            ttsl::reflection::visit_object_of_type<Tensor>(
                [&](const Tensor& t) { bufs.push_back(t.buffer()); }, tensor_args);
            ttsl::reflection::visit_object_of_type<Tensor>(
                [&](const Tensor& t) { bufs.push_back(t.buffer()); }, tensor_return_value);
            return bufs;
        }

        // Helper: call create_descriptor with or without the resource argument.
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

        // Check if kernel k is a compute kernel.
        static bool is_compute_kernel(const tt::tt_metal::KernelDescriptor& kd) {
            return std::holds_alternative<tt::tt_metal::ComputeConfigDescriptor>(kd.config);
        }

        // Build set of (kernel, core, arg_index) positions that hold seed values.
        // These are excluded from both address-slot scanning and determinism checks.
        static std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> build_seed_positions(
            const tt::tt_metal::ProgramDescriptor& desc) {
            std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> positions;
            if constexpr (has_seed) {
                for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
                    if (!is_compute_kernel(desc.kernels[k])) {
                        continue;
                    }
                    for (const auto& [core, args] : desc.kernels[k].runtime_args) {
                        if (!args.empty()) {
                            positions.emplace(k, core.x, core.y, 0);
                        }
                    }
                }
            }
            return positions;
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
                // --- Prepare resources (if the factory needs device-side allocations) ---
                resource_t resources{};
                if constexpr (has_prepare_resources) {
                    resources = DescriptorFactory::prepare_resources(attrs, tensor_args, tensor_return_value);
                }

                auto desc = invoke_create_descriptor(attrs, tensor_args, tensor_return_value, resources);
                uint32_t num_kernels = static_cast<uint32_t>(desc.kernels.size());

                // Build seed positions (compute kernel arg[0]) to exclude from address scanning.
                auto seed_positions = build_seed_positions(desc);

                // --- Record compute kernel cores for seed patching on cache hits ---
                std::vector<ComputeKernelCores> compute_kernel_cores;
                if constexpr (has_seed) {
                    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
                        if (!is_compute_kernel(desc.kernels[k])) {
                            continue;
                        }
                        std::vector<CoreCoord> cores;
                        cores.reserve(desc.kernels[k].runtime_args.size());
                        for (const auto& [core, args] : desc.kernels[k].runtime_args) {
                            cores.push_back(core);
                        }
                        if (!cores.empty()) {
                            compute_kernel_cores.push_back({k, std::move(cores)});
                        }
                    }
                }

                // --- Runtime arg address slots ---
                std::vector<AddressSlot> slots;
                if (!has_addr_collision) {
                    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
                        const auto& kernel_desc = desc.kernels[k];
                        for (const auto& [core, args] : kernel_desc.runtime_args) {
                            for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                                if (seed_positions.count({k, core.x, core.y, i})) {
                                    continue;
                                }
                                auto it = addr_to_id.find(args[i]);
                                if (it != addr_to_id.end()) {
                                    slots.push_back({k, core, i, it->second});
                                }
                            }
                        }
                    }
                }

                // Verify determinism for pure factories (no prepare_resources).
                // Seed positions and address slots are expected to differ between calls.
                if constexpr (!has_prepare_resources) {
                    std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> skip_positions = seed_positions;
                    for (const auto& slot : slots) {
                        skip_positions.emplace(slot.kernel_handle, slot.core.x, slot.core.y, slot.arg_index);
                    }

                    auto desc2 = invoke_create_descriptor(attrs, tensor_args, tensor_return_value, resources);
                    for (uint32_t k = 0; k < num_kernels; ++k) {
                        const auto& k1 = desc.kernels[k];
                        const auto& k2 = desc2.kernels[k];
                        for (size_t ci = 0; ci < k1.runtime_args.size(); ++ci) {
                            const auto& [core1, args1] = k1.runtime_args[ci];
                            const auto& [core2, args2] = k2.runtime_args[ci];
                            for (uint32_t ai = 0; ai < static_cast<uint32_t>(args1.size()); ++ai) {
                                TT_FATAL(
                                    args1[ai] == args2[ai] || skip_positions.count({k, core1.x, core1.y, ai}),
                                    "ProgramDescriptorFactory produces non-deterministic runtime "
                                    "args (kernel={}, core=({},{}), arg_index={}, values {} vs "
                                    "{}). create_descriptor must be deterministic: only buffer "
                                    "addresses (and seed, if attrs has a seed field) may vary "
                                    "between calls.",
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

                // --- Optional post-creation validation ---
                if constexpr (has_post_create_validation) {
                    DescriptorFactory::post_create_validation(program, attrs, tensor_args, tensor_return_value);
                }

                // --- Dynamic circular buffer slots ---
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
                    .compute_kernel_cores = std::move(compute_kernel_cores),
                    .num_kernel_handles = num_kernels,
                    .resources = std::move(resources),
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

                auto bufs = collect_buffers(tensor_args, tensor_return_value);

                for (const auto& slot : sv.address_slots) {
                    auto& args = tt::tt_metal::GetRuntimeArgs(program, slot.kernel_handle, slot.core);
                    args[slot.arg_index] = bufs[slot.buffer_id]->address();
                }

                for (const auto& cb_slot : sv.cb_slots) {
                    tt::tt_metal::UpdateDynamicCircularBufferAddress(
                        program, cb_slot.cb_handle, *bufs[cb_slot.buffer_id]);
                }

                if constexpr (has_seed) {
                    for (const auto& ck : sv.compute_kernel_cores) {
                        for (uint32_t i = 0; i < static_cast<uint32_t>(ck.cores.size()); ++i) {
                            auto& args = tt::tt_metal::GetRuntimeArgs(program, ck.kernel_handle, ck.cores[i]);
                            args[0] = attrs.seed + i;
                        }
                    }
                }
            }
        }
    };

    static ttsl::hash::hash_t compute_mesh_workload_hash(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        [[maybe_unused]] tensor_return_value_t& tensor_return_value) {
        // Hash logic:
        //  1. If the operation provides a custom compute_program_hash, use it.
        //  2. If the operation has a seed field (HasSeed), auto-zero it before hashing.
        //  3. Otherwise, fall back to the default hash of type + attrs + tensor_args.
        ttsl::hash::hash_t hash;

        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            hash = DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else if constexpr (HasSeed<operation_attributes_t>) {
            auto attrs_copy = attrs;
            attrs_copy.seed = 0;
            hash = ttsl::hash::hash_objects_with_default_seed(
                ttsl::hash::type_hash<DeviceOperation>, attrs_copy, tensor_args);
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
