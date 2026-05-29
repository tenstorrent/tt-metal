// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <concepts>
#include <unordered_map>
#include <variant>
#include <vector>
#include <array>
#include <tuple>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal2_artifacts.hpp"
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
// no lambda dispatch, and no virtual call.
//
// Used to walk `tensor_args_t` / `tensor_return_value_t` (op input/output
// tuples of Tensors).  The WorkloadDescriptor's buffer vector is read
// directly — its element types are known, no reflection needed.
//
// Default specialisation is a no-op so unreflectable / unknown leaves are
// silently skipped rather than failing the walk.
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

// Standard containers / wrappers have dedicated specialisations below. Without
// excluding them here, std::array (and any other Reflectable aggregate that
// also has a hand-written specialisation) would match both this Reflectable
// fallback and its specific specialisation, producing an ambiguous-partial-
// specialisation error. is_handled_container_v lists the types that ship with
// a dedicated specialisation in this file; the Reflectable fallback skips them.
template <typename T>
struct is_handled_container : std::false_type {};
template <typename T>
struct is_handled_container<std::optional<T>> : std::true_type {};
template <typename T>
struct is_handled_container<std::vector<T>> : std::true_type {};
template <typename T, std::size_t N>
struct is_handled_container<std::array<T, N>> : std::true_type {};
template <typename... Ts>
struct is_handled_container<std::tuple<Ts...>> : std::true_type {};
template <typename T>
inline constexpr bool is_handled_container_v = is_handled_container<T>::value;

// Aggregate — unroll over fields at compile time.
template <typename T>
struct extract_tensor_buffers_t<
    T,
    std::enable_if_t<
        ttsl::concepts::Reflectable<T> and not std::is_same_v<T, tt::tt_metal::Tensor> and
        not is_handled_container_v<T>>> {
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
    // DescriptorMeshWorkloadAdapter
    //
    // Adapts a ProgramDescriptorFactoryConcept factory for mesh dispatch.
    //
    // Supports two contracts:
    //
    //   1) Simple ops (no workload-scoped state):
    //        static ProgramDescriptor T::create_descriptor(attrs, args, ret, [coord]);
    //      For each mesh coordinate the framework calls create_descriptor() and
    //      builds a Program from the returned descriptor.
    //
    //   2) Workload-scoped ops (declarative WorkloadDescriptor):
    //        static tt::tt_metal::WorkloadDescriptor T::create_workload_descriptor(
    //            attrs, args, ret, const MeshCoordinateRangeSet& tensor_coords);
    //
    //      create_workload_descriptor() runs ONCE per workload (cache miss) and
    //      returns the whole workload in one shot: GlobalSemaphores allocated
    //      and Synchronize barriers run as part of building the descriptor, and
    //      `programs` populated with per-coord ProgramDescriptors. Resources
    //      live on the descriptor itself (typed slots: `semaphores`, `buffers`)
    //      and outlive the cached workload via the program cache.
    //
    // On cache hits the framework either patches buffer addresses through the
    // BufferBinding fast path (contract 2 always; contract 1 when the factory
    // used emplace_runtime_args()) or, for contract (1) factories that bind
    // raw addresses, rebuilds the descriptor and bulk-copies runtime args.
    // -----------------------------------------------------------------------
    template <ProgramDescriptorFactoryConcept DescriptorFactory>
    struct DescriptorMeshWorkloadAdapter {
        // --- Contract detection ---

        // Contract (2): does the factory define a static create_workload_descriptor
        // that takes the tensor coord range set AND returns a
        // tt::tt_metal::WorkloadDescriptor?  The return-type check pins
        // the contract so an accidental wrong signature surfaces as a clean
        // concept failure rather than silent fallback to contract (1).
        static constexpr bool has_workload_descriptor = requires(
            const operation_attributes_t& a,
            const tensor_args_t& t,
            tensor_return_value_t& r,
            const ttnn::MeshCoordinateRangeSet& tc) {
            {
                DescriptorFactory::create_workload_descriptor(a, t, r, tc)
            } -> std::same_as<tt::tt_metal::WorkloadDescriptor>;
        };

        // Signature-mismatch guard: a factory that defines a
        // `create_workload_descriptor` with a wrong signature/return type would
        // silently fall through to the contract-1 path (which then tries
        // `create_descriptor` and produces a deep template error).  Surface
        // the problem clearly at concept-check time.
        static_assert(
            !requires(
                const operation_attributes_t& a,
                const tensor_args_t& t,
                tensor_return_value_t& r,
                const ttnn::MeshCoordinateRangeSet& tc) {
                DescriptorFactory::create_workload_descriptor(a, t, r, tc);
            } || has_workload_descriptor,
            "Factory has create_workload_descriptor but its return type isn't "
            "tt::tt_metal::WorkloadDescriptor. "
            "Expected: static tt::tt_metal::WorkloadDescriptor create_workload_descriptor("
            "const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&, "
            "const ttnn::MeshCoordinateRangeSet&)");

        struct shared_variables_t {
            // The mesh workload descriptor — built ONCE per workload (cache
            // miss) by create_workload_descriptor() and held here so its resource
            // members (semaphores, buffers) outlive the cached workload via
            // the program cache.  Default-constructed (empty vectors) for
            // contract (1) factories.
            tt::tt_metal::WorkloadDescriptor workload_descriptor;
            // Resolved buffer bindings for the fast cache-hit path.
            // Non-empty when the factory used emplace_runtime_args() with
            // Buffer* args (or, for contract 2, declared any CB buffer binding).
            tt::tt_metal::ResolvedBindings resolved_bindings;
        };
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Enumerate every Buffer* reachable from `tensor_args`,
        // `tensor_return_value`, and the workload descriptor's resource
        // buffers (workload-scoped MeshBuffers exposed via .buffers).  Order is
        // stable: input tensors, then outputs, then workload buffers in their
        // declaration order.  Used to map buffer bindings to indices that
        // survive across dispatches without storing raw pointers — workload
        // buffers are included so factories can bind runtime args / CBs to
        // op-owned device buffers (halo lookup tables, etc.) via
        // emplace_runtime_args() / `.buffer = ...`.
        //
        // Returns a stack-allocated SmallVector (16 inline slots) so the
        // cache-hit fast path avoids the heap allocation tax.
        static ttsl::SmallVector<tt::tt_metal::Buffer*, 16> collect_tensor_buffers(
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value,
            const tt::tt_metal::WorkloadDescriptor& workload_descriptor) {
            ttsl::SmallVector<tt::tt_metal::Buffer*, 16> buffers;
            extract_tensor_buffers_into(tensor_args, buffers);
            extract_tensor_buffers_into(tensor_return_value, buffers);
            for (const auto& wb : workload_descriptor.buffers) {
                buffers.push_back(wb.buffer);
            }
            return buffers;
        }

        // Whether create_descriptor (contract 1) wants the per-coord MeshCoordinate.
        // DirectDescriptorFactory accepts the 4-arg form unconditionally but only forwards
        // when the underlying DeviceOperation defines it. Custom factories opt in explicitly.
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

        // Build a ProgramDescriptor for one mesh coordinate (contract 1).
        // The declarative WorkloadDescriptor path (contract 2) does NOT go through
        // this — it iterates `workload_descriptor.programs` directly.
        static tt::tt_metal::ProgramDescriptor invoke_per_coord(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
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

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

            if constexpr (has_workload_descriptor) {
                // Contract (2) — declarative: the factory builds the entire
                // WorkloadDescriptor (resources + per-coord programs) in
                // one call.  Resources (GlobalSemaphores, MeshBuffers) are
                // allocated and any Synchronize barrier is run as part of
                // create_workload_descriptor(); we then iterate `programs` to
                // populate the cached MeshWorkload.
                //
                // `programs` is moved out before the loop — each per-range
                // shared_variables copy only needs to carry the resources
                // (semaphores, buffers) and resolved bindings.  The per-coord
                // ProgramDescriptors have already been consumed into Programs.
                tt::tt_metal::WorkloadDescriptor workload_descriptor = DescriptorFactory::create_workload_descriptor(
                    attrs, tensor_args, tensor_return_value, tensor_coords);
                auto programs = std::move(workload_descriptor.programs);
                for (auto& [device_range, desc] : programs) {
                    tt::tt_metal::Program program{desc};
                    auto tensor_buffers = collect_tensor_buffers(tensor_args, tensor_return_value, workload_descriptor);
                    auto resolved = tt::tt_metal::resolve_bindings(program, desc, tensor_buffers);
                    mesh_workload.add_program(device_range, std::move(program));
                    shared_variables[device_range] = shared_variables_t{
                        .workload_descriptor = workload_descriptor, .resolved_bindings = std::move(resolved)};
                }
                return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
            } else {
                // Contract (1) — simple per-coord create_descriptor.
                tt::tt_metal::WorkloadDescriptor empty_descriptor;

                const auto build_and_add_program =
                    [&](const ttnn::MeshCoordinateRange& device_range,
                        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                        auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                        tt::tt_metal::Program program{desc};
                        auto tensor_buffers =
                            collect_tensor_buffers(tensor_args, tensor_return_value, empty_descriptor);
                        auto resolved = tt::tt_metal::resolve_bindings(program, desc, tensor_buffers);
                        mesh_workload.add_program(device_range, std::move(program));
                        shared_variables[device_range] = shared_variables_t{.resolved_bindings = std::move(resolved)};
                    };

                if constexpr (create_descriptor_uses_mesh_dispatch_coordinate()) {
                    for (const auto& coord : tensor_coords.coords()) {
                        build_and_add_program(
                            ttnn::MeshCoordinateRange(coord), std::optional<ttnn::MeshCoordinate>(coord));
                    }
                } else {
                    for (const auto& range : tensor_coords.ranges()) {
                        build_and_add_program(range, std::nullopt);
                    }
                }
                return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
            }
        }

        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto& sv = cached_workload.shared_variables.at(coordinate_range);

                if constexpr (has_workload_descriptor) {
                    // Contract (2) — declarative: there is no slow-path rebuild
                    // because re-running create_workload_descriptor would re-allocate
                    // workload-scoped resources (GlobalSemaphores, MeshBuffers).
                    // CB bindings are always populated by resolve_bindings, so the
                    // fast path covers cache hits even when the factory only sets
                    // `desc.cbs[i].buffer` and declares no rt-arg buffer bindings.
                    if (!sv.resolved_bindings.empty()) {
                        auto current_buffers =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, current_buffers);
                    }
                } else {
                    // Contract (1) — simple per-coord factory.  Fast-path only
                    // when the factory declared rt-arg buffer bindings via
                    // emplace_runtime_args().  Without those, the factory may be
                    // mixing `.buffer = ...` CBs with OLD-style raw uint32 rt-args
                    // that are not registered as bindings; patching CBs alone
                    // would leave those rt-args pointing at stale addresses.  Fall
                    // through to the slow-path rebuild instead.
                    if (!sv.resolved_bindings.rt_args.empty()) {
                        auto current_buffers =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, current_buffers);
                    } else {
                        const ttnn::MeshCoordinate mesh_coord = coordinate_range.start_coord();
                        const std::optional<ttnn::MeshCoordinate> mesh_dispatch_coordinate(mesh_coord);
                        auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                        tt::tt_metal::apply_descriptor_runtime_args(program, desc);
                    }
                }
            }
        }
    };

    // -----------------------------------------------------------------------
    // ProgramSpecMeshWorkloadFactoryAdapter
    //
    // Adapts a ProgramSpecFactoryConcept factory (Metal 2.0, single-program /
    // SPMD-flavored) for mesh dispatch. The op author writes ONLY
    // create_program_spec, returning a single ProgramArtifacts (one ProgramSpec
    // + ProgramRunArgs). The adapter stamps a Program from that spec onto
    // each mesh coordinate range covered by the workload — mirroring the
    // descriptor adapter's per-range build pattern.
    //
    // On cache miss: the adapter calls create_program_spec, builds one Program
    // per coordinate range via metal2_host_api::MakeProgramFromSpec, applies
    // the initial ProgramRunArgs via SetProgramRunArgs, then resolves
    // each TensorArgument against the io_tensors enumerated from tensor_args /
    // tensor_return_value (pointer-identity match within the call).
    //
    // On cache hit: the adapter enumerates fresh io_tensors, mutates the
    // cached TensorArgument storage in place using the stored index bindings, and
    // applies via metal2_host_api::UpdateTensorArgs — no Program rebuild,
    // no heap allocation.
    //
    // Limitation: every TensorArgument returned by the factory must reference a
    // MeshTensor reachable from tensor_args or tensor_return_value.
    //
    // TODO: support op-owned resource tensors (the prepare_resources analog
    // from the descriptor adapter) — will require extending shared_variables_t
    // and the io_tensor enumeration to include factory-produced tensors.
    //
    // TODO: consider replacing with a general MeshWorkloadSpecFactoryAdapter?
    // -----------------------------------------------------------------------
    template <ProgramSpecFactoryConcept ProgramSpecFactory>
    struct ProgramSpecMeshWorkloadFactoryAdapter {
        using TensorParameterName = tt::tt_metal::experimental::metal2_host_api::TensorParameterName;
        using TensorArgument = tt::tt_metal::experimental::metal2_host_api::ProgramRunArgs::TensorArgument;

        // Stored across cache entries: for each TensorArgument in a program's
        // ProgramRunArgs, which io_tensor (by index into the deterministic
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

        // Match each TensorArgument's MeshTensor reference back to its index in the
        // io_tensor enumeration. Cache-miss path only.
        // TT_FATALs on a TensorArgument that doesn't reference an io_tensor (see
        // adapter-level TODO on op-owned resource tensor support).
        //
        // NOTE on host perf: the index-based binding scheme is what makes a
        // fast cache-hit path possible, but the current straightforward
        // implementation isn't there yet — collect_mesh_tensors returns a
        // heap std::vector, and apply_descriptor builds a fresh TensorArgument
        // vector each dispatch. Both costs are fixable by mirroring the
        // descriptor adapter's compile-time-unrolled walker + SmallVector +
        // cached TensorArgument storage pattern. Deferred pending profiling.
        static std::vector<ResolvedTensorBinding> resolve_bindings(
            const std::vector<TensorArgument>& factory_tensor_args,
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
                    "TensorArgument '{}' must reference a MeshTensor reachable from tensor_args or "
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
            // first device tensor reachable from tensor_args. Op factories
            // satisfying this concept are tensor-driven, so first_tensor is
            // always populated for current callers.
            auto first_tensor = ttsl::reflection::get_first_object_of_type<tt::tt_metal::Tensor>(tensor_args);
            TT_FATAL(
                first_tensor.has_value(),
                "ProgramSpec factory adapter requires at least one Tensor in tensor_args to source the MeshDevice");
            auto* mesh_device = first_tensor.value().device();
            TT_FATAL(mesh_device != nullptr, "First tensor in tensor_args must be allocated on a MeshDevice");

            // The factory produces a single ProgramArtifacts; the adapter stamps it
            // across all coordinate ranges. Bindings derive from the (single) set of
            // factory tensor_args and are identical for every stamped program; copy
            // per range into the cached shared state.
            auto artifacts = ProgramSpecFactory::create_program_spec(attrs, tensor_args, tensor_return_value);
            auto io_mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);
            auto bindings = resolve_bindings(artifacts.run_params.tensor_args, io_mesh_tensors);

            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            for (const auto& range : tensor_coords.ranges()) {
                auto program =
                    tt::tt_metal::experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, artifacts.spec);
                tt::tt_metal::experimental::metal2_host_api::SetProgramRunArgs(program, artifacts.run_params);
                shared_variables.emplace(range, shared_variables_t{.bindings = bindings});
                mesh_workload.add_program(range, std::move(program));
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        // The framework's cache-hit dispatcher (handle_mesh_adapter_cache_hit in
        // device_operation.hpp) prefers a method named `apply_descriptor` over
        // `override_runtime_arguments` when both exist. We adopt the name to
        // slot into that hook directly — the "descriptor" word here is the
        // dispatcher's historical naming, not a reference to ProgramDescriptor.
        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& /*attrs*/,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto io_mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                const auto& sv = cached_workload.shared_variables.at(coordinate_range);
                std::vector<TensorArgument> fresh_tensor_args;
                fresh_tensor_args.reserve(sv.bindings.size());
                for (const auto& b : sv.bindings) {
                    fresh_tensor_args.push_back(TensorArgument{
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
