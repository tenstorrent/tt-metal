// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <concepts>
#include <unordered_map>
#include <variant>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal2_mesh_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operation.hpp"
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation {

template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

// Extracts every Tensor reachable from an aggregate `T` and pushes its buffer()
// onto `out`.  Generated per-T at compile time so the compiler emits a
// straight-line walk of T's tensor fields with no runtime reflection visit,
// no lambda dispatch, and no virtual call.  Equivalent in semantics to
// ttsl::reflection::visit_object_of_type<Tensor> but keeps the call chain
// short enough that the optimiser inlines through it for typical tensor_args.
//
// Default specialisation is a no-op; a second specialisation handles
// Reflectable aggregates by unrolling over their fields.  Container/leaf
// specialisations follow.
template <typename T, typename = void>
struct extract_tensor_buffers_t {
    template <typename Out>
    static void call(const T&, Out&) {}
};

template <typename T, typename Out>
inline void extract_tensor_buffers_into(const T& obj, Out& out) {
    extract_tensor_buffers_t<std::decay_t<T>>::call(obj, out);
}

// Tensor leaf — push the buffer.
template <>
struct extract_tensor_buffers_t<tt::tt_metal::Tensor, void> {
    template <typename Out>
    static void call(const tt::tt_metal::Tensor& t, Out& out) {
        out.push_back(t.buffer());
    }
};

// Aggregate — unroll over fields at compile time.
template <typename T>
struct extract_tensor_buffers_t<
    T,
    std::enable_if_t<ttsl::concepts::Reflectable<T> and not std::is_same_v<T, tt::tt_metal::Tensor>>> {
    template <typename Out>
    static void call(const T& obj, Out& out) {
        reflect::for_each([&obj, &out](auto I) { extract_tensor_buffers_into(reflect::get<I>(obj), out); }, obj);
    }
};

// Optional — visit value if present.
template <typename T>
struct extract_tensor_buffers_t<std::optional<T>, void> {
    template <typename Out>
    static void call(const std::optional<T>& v, Out& out) {
        if (v.has_value()) {
            extract_tensor_buffers_into(v.value(), out);
        }
    }
};

// Vector — runtime loop (unavoidable, count is dynamic).
template <typename T>
struct extract_tensor_buffers_t<std::vector<T>, void> {
    template <typename Out>
    static void call(const std::vector<T>& v, Out& out) {
        for (const auto& e : v) {
            extract_tensor_buffers_into(e, out);
        }
    }
};

// Array — unroll at compile time.
template <typename T, std::size_t N>
struct extract_tensor_buffers_t<std::array<T, N>, void> {
    template <typename Out>
    static void call(const std::array<T, N>& v, Out& out) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (extract_tensor_buffers_into(v[Is], out), ...);
        }(std::make_index_sequence<N>{});
    }
};

// Tuple — unroll at compile time.
template <typename... Ts>
struct extract_tensor_buffers_t<std::tuple<Ts...>, void> {
    template <typename Out>
    static void call(const std::tuple<Ts...>& v, Out& out) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (extract_tensor_buffers_into(std::get<Is>(v), out), ...);
        }(std::make_index_sequence<sizeof...(Ts)>{});
    }
};

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
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt) {
            if constexpr (requires {
                              DeviceOperation::create_descriptor(
                                  attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                          }) {
                return DeviceOperation::create_descriptor(
                    attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
            } else {
                (void)mesh_dispatch_coordinate;
                return DeviceOperation::create_descriptor(attrs, tensor_args, tensor_return_value);
            }
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

        // Enumerate all Buffer* reachable from tensor_args, tensor_return_value, and
        // any Tensor fields inside resources (from prepare_resources).  Stable field
        // order via reflection.  Used to map buffer bindings to indices that survive
        // across calls without storing raw pointers.  Resource tensors are included
        // so factories can bind kernel runtime args to halo lookup tables and other
        // op-owned buffers via emplace_runtime_args() / Buffer*.
        //
        // The resources visit is gated on has_prepare_resources because empty_resource_t
        // is not guaranteed to be reflectable, and visit_object_of_type would throw at
        // runtime on an unreflectable type that is not the target object_t.
        //
        // Returns a stack-allocated SmallVector (16 inline slots) instead of a heap
        // vector so the cache-hit fast path avoids one allocation per dispatch.
        // The reflection itself is already compile-time generated; this just removes
        // the runtime allocation tax.
        static ttsl::SmallVector<tt::tt_metal::Buffer*, 16> collect_tensor_buffers(
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value,
            const resource_t& resources) {
            ttsl::SmallVector<tt::tt_metal::Buffer*, 16> buffers;
            extract_tensor_buffers_into(tensor_args, buffers);
            extract_tensor_buffers_into(tensor_return_value, buffers);
            if constexpr (has_prepare_resources) {
                extract_tensor_buffers_into(resources, buffers);
            }
            return buffers;
        }

        // DirectDescriptorFactory always accepts an optional mesh coordinate, but only forwards it when
        // DeviceOperation defines the 4-argument overload. Custom descriptor factories opt in explicitly.
        static consteval bool create_descriptor_uses_mesh_dispatch_coordinate() {
            if constexpr (std::is_same_v<DescriptorFactory, DirectDescriptorFactory>) {
                return requires(
                    const operation_attributes_t& attrs,
                    const tensor_args_t& tensor_args,
                    tensor_return_value_t& tensor_return_value,
                    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                    DeviceOperation::create_descriptor(
                        attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                };
            } else if constexpr (has_prepare_resources) {
                return requires(
                    const operation_attributes_t& attrs,
                    const tensor_args_t& tensor_args,
                    tensor_return_value_t& tensor_return_value,
                    resource_t& resources,
                    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                    DescriptorFactory::create_descriptor(
                        attrs, tensor_args, tensor_return_value, resources, mesh_dispatch_coordinate);
                };
            } else {
                return requires(
                    const operation_attributes_t& attrs,
                    const tensor_args_t& tensor_args,
                    tensor_return_value_t& tensor_return_value,
                    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                    DescriptorFactory::create_descriptor(
                        attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                };
            }
        }

        static tt::tt_metal::ProgramDescriptor invoke_create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            resource_t& resources,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
            if constexpr (has_prepare_resources) {
                if constexpr (requires {
                                  DescriptorFactory::create_descriptor(
                                      attrs, tensor_args, tensor_return_value, resources, mesh_dispatch_coordinate);
                              }) {
                    return DescriptorFactory::create_descriptor(
                        attrs, tensor_args, tensor_return_value, resources, mesh_dispatch_coordinate);
                } else {
                    (void)mesh_dispatch_coordinate;
                    return DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value, resources);
                }
            } else {
                if constexpr (requires {
                                  DescriptorFactory::create_descriptor(
                                      attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                              }) {
                    return DescriptorFactory::create_descriptor(
                        attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                } else {
                    (void)mesh_dispatch_coordinate;
                    return DescriptorFactory::create_descriptor(attrs, tensor_args, tensor_return_value);
                }
            }
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

            const auto build_and_add_program =
                [&](const ttnn::MeshCoordinateRange& device_range,
                    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                    resource_t resources{};
                    if constexpr (has_prepare_resources) {
                        resources = DescriptorFactory::prepare_resources(attrs, tensor_args, tensor_return_value);
                    }

                    auto desc = invoke_create_descriptor(
                        attrs, tensor_args, tensor_return_value, resources, mesh_dispatch_coordinate);
                    tt::tt_metal::Program program{desc};
                    auto tensor_buffers = collect_tensor_buffers(tensor_args, tensor_return_value, resources);
                    auto resolved = tt::tt_metal::resolve_bindings(program, desc, tensor_buffers);
                    mesh_workload.add_program(device_range, std::move(program));
                    shared_variables[device_range] = shared_variables_t{
                        .resources = std::move(resources), .resolved_bindings = std::move(resolved)};
                };

            if constexpr (create_descriptor_uses_mesh_dispatch_coordinate()) {
                for (const auto& coord : tensor_coords.coords()) {
                    build_and_add_program(ttnn::MeshCoordinateRange(coord), std::optional<ttnn::MeshCoordinate>(coord));
                }
            } else {
                for (const auto& range : tensor_coords.ranges()) {
                    build_and_add_program(range, std::nullopt);
                }
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
                    auto current_buffers = collect_tensor_buffers(tensor_args, tensor_return_value, sv.resources);
                    tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, current_buffers);
                } else {
                    // Slow path: full descriptor rebuild + bulk copy.
                    // Used by factories that have not yet adopted emplace_runtime_args().
                    const std::optional<ttnn::MeshCoordinate> mesh_dispatch_coordinate(coordinate_range.start_coord());
                    auto desc = invoke_create_descriptor(
                        attrs, tensor_args, tensor_return_value, sv.resources, mesh_dispatch_coordinate);
                    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
                }
            }
        }
    };

    // -----------------------------------------------------------------------
    // Metal2MeshWorkloadFactoryAdapter
    //
    // Adapts a Metal2MeshSpecFactoryConcept factory for mesh dispatch.
    // The developer writes ONLY create_mesh_spec, returning MeshArtifacts
    // (one ProgramSpec + ProgramRunParams per mesh coordinate range).
    //
    // On cache miss: the adapter calls create_mesh_spec, builds each Program
    // via metal2_host_api::MakeProgramFromSpec, applies the initial
    // ProgramRunParams via SetProgramRunParameters, then resolves each
    // TensorArg against the io_tensors enumerated from tensor_args /
    // tensor_return_value (pointer-identity match within the call).
    //
    // On cache hit: the adapter enumerates fresh io_tensors, reconstructs
    // TensorArgs from the stored index bindings, and applies them via
    // metal2_host_api::UpdateTensorArgs — no Program rebuild.
    //
    // Phase 1 limitation: every TensorArg returned by the factory must
    // reference a MeshTensor reachable from tensor_args or tensor_return_value.
    // Op-owned resource tensors (the prepare_resources analog) are deferred to
    // Phase 2.
    // -----------------------------------------------------------------------
    template <Metal2MeshSpecFactoryConcept Metal2Factory>
    struct Metal2MeshWorkloadFactoryAdapter {
        using TensorParameterName = tt::tt_metal::experimental::metal2_host_api::TensorParameterName;
        using TensorArg = tt::tt_metal::experimental::metal2_host_api::ProgramRunParams::TensorArg;

        // Stored across cache entries: for each TensorArg in a program's
        // ProgramRunParams, which io_tensor (by index into the deterministic
        // reflection-driven enumeration) it was bound to. Pointer identity is
        // only valid within a single call; the index is stable across calls.
        struct ResolvedTensorBinding {
            TensorParameterName tensor_parameter_name;
            std::size_t io_tensor_idx;
        };

        struct shared_variables_t {
            std::vector<ResolvedTensorBinding> bindings;
        };
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Walk tensor_args and tensor_return_value via reflection, collecting
        // the MeshTensor of every Tensor leaf. The walk order is deterministic
        // (reflection-driven, stable across calls), so the resulting indices
        // are stable across calls. Metal 2.0 analog of the descriptor adapter's
        // collect_tensor_buffers, at the MeshTensor level instead of Buffer*.
        static std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> collect_mesh_tensors(
            const tensor_args_t& tensor_args, const tensor_return_value_t& tensor_return_value) {
            std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> result;
            const auto visit = [&result](const tt::tt_metal::Tensor& t) {
                result.push_back(std::cref(t.mesh_tensor()));
            };
            ttsl::reflection::visit_object_of_type<tt::tt_metal::Tensor>(visit, tensor_args);
            ttsl::reflection::visit_object_of_type<tt::tt_metal::Tensor>(visit, tensor_return_value);
            return result;
        }

        // Match each TensorArg's MeshTensor reference back to its index in the
        // io_tensor enumeration. Phase 1 requires every TensorArg target to
        // come from tensor_args / tensor_return_value (no op-owned resources).
        static std::vector<ResolvedTensorBinding> resolve_bindings(
            const std::vector<TensorArg>& factory_tensor_args,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& io_mesh_tensors) {
            std::vector<ResolvedTensorBinding> bindings;
            bindings.reserve(factory_tensor_args.size());
            for (const auto& tensor_arg : factory_tensor_args) {
                const auto* target = &tensor_arg.tensor.get();
                auto it = std::find_if(io_mesh_tensors.begin(), io_mesh_tensors.end(), [target](const auto& wrapped) {
                    return &wrapped.get() == target;
                });
                TT_FATAL(
                    it != io_mesh_tensors.end(),
                    "TensorArg '{}' must reference a MeshTensor reachable from tensor_args or "
                    "tensor_return_value (got non-io_tensor MeshTensor)",
                    tensor_arg.tensor_parameter_name);
                bindings.push_back(
                    {tensor_arg.tensor_parameter_name,
                     static_cast<std::size_t>(std::distance(io_mesh_tensors.begin(), it))});
            }
            return bindings;
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            // Metal 2.0's MakeProgramFromSpec needs a MeshDevice; pull from the
            // first device tensor reachable from tensor_args (Phase 1 ops are
            // tensor-driven, so this is always populated).
            auto first_tensor = ttsl::reflection::get_first_object_of_type<tt::tt_metal::Tensor>(tensor_args);
            TT_FATAL(
                first_tensor.has_value(),
                "Metal 2.0 factory adapter requires at least one Tensor in tensor_args to source the MeshDevice");
            auto* mesh_device = first_tensor.value().device();
            TT_FATAL(mesh_device != nullptr, "First tensor in tensor_args must be allocated on a MeshDevice");

            auto artifacts = Metal2Factory::create_mesh_spec(attrs, tensor_args, tensor_return_value, tensor_coords);
            auto io_mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);

            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            for (auto& [range, prog_artifacts] : artifacts.programs) {
                auto program =
                    tt::tt_metal::experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, prog_artifacts.spec);
                tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(
                    program, prog_artifacts.run_params);
                auto bindings = resolve_bindings(prog_artifacts.run_params.tensor_args, io_mesh_tensors);
                shared_variables.emplace(range, shared_variables_t{.bindings = std::move(bindings)});
                mesh_workload.add_program(range, std::move(program));
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& /*attrs*/,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto io_mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                const auto& sv = cached_workload.shared_variables.at(coordinate_range);
                std::vector<TensorArg> fresh_tensor_args;
                fresh_tensor_args.reserve(sv.bindings.size());
                for (const auto& b : sv.bindings) {
                    fresh_tensor_args.push_back(TensorArg{
                        .tensor_parameter_name = b.tensor_parameter_name, .tensor = io_mesh_tensors[b.io_tensor_idx]});
                }
                tt::tt_metal::experimental::metal2_host_api::UpdateTensorArgs(program, fresh_tensor_args);
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
