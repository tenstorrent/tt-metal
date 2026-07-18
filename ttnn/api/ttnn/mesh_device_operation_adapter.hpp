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
#include <string>
#include <string_view>
#include <type_traits>
#include <concepts>
#include <unordered_map>
#include <variant>
#include <vector>
#include <array>
#include <tuple>
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
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
    // Supports two factory variants:
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
    // BufferBinding fast path (the WorkloadDescriptor variant always; the ProgramDescriptor variant when the factory
    // used emplace_runtime_args()) or, for ProgramDescriptor-variant factories that bind
    // raw addresses, rebuilds the descriptor and bulk-copies runtime args.
    // -----------------------------------------------------------------------
    template <ProgramDescriptorFactoryConcept DescriptorFactory>
    struct DescriptorMeshWorkloadAdapter {
        // --- Variant detection ---

        // WorkloadDescriptor variant: does the factory define a static create_workload_descriptor
        // that takes the tensor coord range set AND returns a
        // tt::tt_metal::WorkloadDescriptor?  The return-type check pins
        // the requirement so an accidental wrong signature surfaces as a clean
        // concept failure rather than silent fallback to the ProgramDescriptor variant.
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
        // silently fall through to the ProgramDescriptor-variant path (which then tries
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
            // ProgramDescriptor-variant factories.
            tt::tt_metal::WorkloadDescriptor workload_descriptor;
            // Resolved buffer bindings for the fast cache-hit path.
            // Non-empty when the factory used emplace_runtime_args() with
            // Buffer* args (or, for the WorkloadDescriptor variant, declared any CB buffer binding).
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
        // Result of collect_tensor_buffers. `num_input_buffers` is the boundary between input
        // buffers and output/workload buffers, used by resolve_bindings to allow safe in-place
        // aliasing (output buffer == input buffer) while still bailing on ambiguous duplicates
        // within the inputs (e.g. matmul(X, X)).
        struct CollectedTensorBuffers {
            ttsl::SmallVector<tt::tt_metal::Buffer*, 16> buffers;
            size_t num_input_buffers = 0;
        };

        static CollectedTensorBuffers collect_tensor_buffers(
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value,
            const tt::tt_metal::WorkloadDescriptor& workload_descriptor) {
            CollectedTensorBuffers collected;
            auto& buffers = collected.buffers;
            extract_tensor_buffers_into(tensor_args, buffers);
            collected.num_input_buffers = buffers.size();
            extract_tensor_buffers_into(tensor_return_value, buffers);
            for (const auto& wb : workload_descriptor.buffers) {
                buffers.push_back(wb.buffer);
            }
            return collected;
        }

        // Whether create_descriptor (the ProgramDescriptor variant) wants the per-coord MeshCoordinate.
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

        // Whether the op re-applies ALL per-dispatch state itself on a program-cache hit via
        // override_runtime_arguments() — the descriptor-era analog of the legacy
        // override_runtime_arguments().  When present, the adapter calls it on every hit and uses
        // NEITHER resolve_bindings (address inference) NOR get_dynamic_runtime_args: the op owns the
        // full re-derivation, correct by construction.  This is the target mechanism; resolve_bindings
        // and get_dynamic are the legacy paths being migrated out (and, eventually, deleted with
        // Metal 2.0 native bindings).
        static consteval bool has_override_runtime_arguments() {
            return requires(
                tt::tt_metal::Program& program,
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const std::optional<ttnn::MeshCoordinate>& coord) {
                DeviceOperation::override_runtime_arguments(program, attrs, tensor_args, tensor_return_value, coord);
            };
        }

        static consteval bool has_get_dynamic_runtime_args() {
            return requires(
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const std::optional<ttnn::MeshCoordinate>& coord) {
                DeviceOperation::get_dynamic_runtime_args(attrs, tensor_args, tensor_return_value, coord);
            };
        }

        // An op that owns its cache-hit re-derivation via override_runtime_arguments() must NOT also
        // declare the legacy get_dynamic_runtime_args() — override supersedes it, and having both is
        // ambiguous (which one re-applies?).  This assert forces porting an op to DROP get_dynamic.
        static_assert(
            !(has_override_runtime_arguments() && has_get_dynamic_runtime_args()),
            "A DeviceOperation must not declare BOTH override_runtime_arguments() and "
            "get_dynamic_runtime_args(): override_runtime_arguments supersedes the legacy hook. "
            "Delete get_dynamic_runtime_args() from this op.");

        // Build a ProgramDescriptor for one mesh coordinate (the ProgramDescriptor variant).
        // The declarative WorkloadDescriptor path (the WorkloadDescriptor variant) does NOT go through
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
                // WorkloadDescriptor variant — declarative: the factory builds the entire
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
                    auto collected = collect_tensor_buffers(tensor_args, tensor_return_value, workload_descriptor);
                    // The WorkloadDescriptor variant has NO slow-path rebuild (apply_descriptor only
                    // re-applies resolved bindings + dynamic args), so it must ALWAYS allow the in-place
                    // output_tensor alias — otherwise resolve_bindings bails to an EMPTY ResolvedBindings
                    // and the fast path skips address patching, leaving stale addresses on a cache hit
                    // (breaks supported cross-device p2p where output_tensor aliases input). This restores
                    // the pre-opt-in behavior for this branch; the unsafe_optin gate only applies to the
                    // ProgramDescriptor branch below, which CAN fall back to a safe slow-path rebuild.
                    auto bindings = tt::tt_metal::resolve_bindings(
                        program,
                        desc,
                        collected.buffers,
                        collected.num_input_buffers,
                        /*allow_inplace_output_tensor_alias=*/true);
                    mesh_workload.add_program(device_range, std::move(program));
                    shared_variables[device_range] = shared_variables_t{
                        .workload_descriptor = workload_descriptor, .resolved_bindings = std::move(bindings)};
                }
                return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
            } else {
                // ProgramDescriptor variant — simple per-coord create_descriptor.
                tt::tt_metal::WorkloadDescriptor empty_descriptor;

                const auto build_and_add_program =
                    [&](const ttnn::MeshCoordinateRange& device_range,
                        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
                        auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                        // An empty ProgramDescriptor (no kernels/CBs/semaphores) means this coordinate has no
                        // work. Skip it entirely.
                        if (desc.kernels.empty() && desc.cbs.empty() && desc.semaphores.empty()) {
                            return;
                        }
                        tt::tt_metal::Program program{desc};
                        if constexpr (has_override_runtime_arguments()) {
                            // The op re-derives all per-dispatch state on every hit via
                            // override_runtime_arguments(); no resolve_bindings needed.
                            mesh_workload.add_program(device_range, std::move(program));
                            shared_variables[device_range] = shared_variables_t{};
                        } else {
                            auto collected = collect_tensor_buffers(tensor_args, tensor_return_value, empty_descriptor);
                            auto bindings = tt::tt_metal::resolve_bindings(
                                program, desc, collected.buffers, collected.num_input_buffers);
                            mesh_workload.add_program(device_range, std::move(program));
                            shared_variables[device_range] =
                                shared_variables_t{.resolved_bindings = std::move(bindings)};
                        }
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

        // If the device operation declares dynamic (non-Buffer) runtime args via
        // get_dynamic_runtime_args(), re-apply them to the cached program for this coordinate.
        // These are values a custom compute_program_hash deliberately excluded from the cache key
        // (e.g. an RNG seed, an [from,to) range, a semaphore address) and so must be patched on
        // every fast-path cache hit — the non-Buffer analog of apply_resolved_bindings.  Ops that
        // don't define the method compile this away (if constexpr) and pay nothing.
        static void apply_dynamic_runtime_args_if_declared(
            tt::tt_metal::Program& program,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRange& coordinate_range) {
            if constexpr (requires {
                              DeviceOperation::get_dynamic_runtime_args(
                                  attrs, tensor_args, tensor_return_value, std::optional<ttnn::MeshCoordinate>{});
                          }) {
                const std::optional<ttnn::MeshCoordinate> coord(coordinate_range.start_coord());
                const auto dynamic_args =
                    DeviceOperation::get_dynamic_runtime_args(attrs, tensor_args, tensor_return_value, coord);
                tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
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
                    // WorkloadDescriptor variant — declarative: there is no slow-path rebuild
                    // because re-running create_workload_descriptor would re-allocate
                    // workload-scoped resources (GlobalSemaphores, MeshBuffers).
                    // CB bindings are always populated by resolve_bindings, so the
                    // fast path covers cache hits even when the factory only sets
                    // `desc.cbs[i].buffer` and declares no rt-arg buffer bindings.
                    if (!sv.resolved_bindings.empty()) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                    }
                    // The WorkloadDescriptor variant never rebuilds, so a value a custom hash excluded would
                    // stay frozen at first miss. Prefer the op's override_runtime_arguments() (the descriptor-era
                    // re-apply) — it re-derives every per-dispatch value (e.g. semaphore addresses baked as raw
                    // rt-args that resolve_bindings can't track) via the per-coord builder, WITHOUT re-running
                    // create_workload_descriptor (no GlobalSemaphore/MeshBuffer realloc). resolve_bindings above
                    // still covers tensor addresses, so an override that no-ops for a pure-workload factory is
                    // safe. Fall back to the legacy get_dynamic re-apply for un-migrated workload ops.
                    if constexpr (has_override_runtime_arguments()) {
                        DeviceOperation::override_runtime_arguments(
                            program,
                            attrs,
                            tensor_args,
                            tensor_return_value,
                            std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                    } else {
                        apply_dynamic_runtime_args_if_declared(
                            program, attrs, tensor_args, tensor_return_value, coordinate_range);
                    }
                } else if constexpr (has_override_runtime_arguments()) {
                    // ProgramDescriptor variant, op owns its cache-hit re-derivation (the descriptor-era
                    // override_runtime_arguments()): re-apply ALL per-dispatch state — every runtime arg
                    // AND every tensor-backed CB address — for the current tensors.  No resolve_bindings
                    // (address inference) and no get_dynamic; correct by construction for in-place,
                    // mixed-aliasing, and work-set shifts.
                    DeviceOperation::override_runtime_arguments(
                        program,
                        attrs,
                        tensor_args,
                        tensor_return_value,
                        std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
#ifdef TT_DESCRIPTOR_PATCHING_PARITY_CHECK
                    // Same regression net as the legacy fast path: assert the op's override reproduced a
                    // full rebuild exactly (rt-args AND CB addresses).
                    {
                        auto parity_desc = invoke_per_coord(
                            attrs,
                            tensor_args,
                            tensor_return_value,
                            std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                        tt::tt_metal::Program parity_scratch{parity_desc};
                        tt::tt_metal::apply_descriptor_runtime_args(parity_scratch, parity_desc);
                        tt::tt_metal::assert_fastpath_parity(
                            program, parity_scratch, parity_desc, ttsl::get_type_name<DeviceOperation>());
                    }
#endif
                } else {
                    // ProgramDescriptor variant — simple per-coord factory.  Fast-path when the
                    // factory declared rt-arg buffer bindings via emplace_runtime_args(), OR the op
                    // declares get_dynamic_runtime_args() to re-apply its per-dispatch args — in which
                    // case the framework also patches the op's `.buffer = ...` CB bindings (covers
                    // CB-bound sharded ops).  With neither, a `.buffer` CB mixed with raw uint32
                    // rt-args could go stale, so fall through to the slow-path rebuild instead.
                    //
                    // The get_dynamic_runtime_args() opt-in additionally requires that
                    // resolve_bindings actually produced bindings (!resolved_bindings.empty()):
                    // get_dynamic_runtime_args() only re-applies hash-excluded SCALAR runtime args
                    // (seed/step/lr), so buffer addresses still ride on the resolved bindings.
                    // resolve_bindings returns an EMPTY ResolvedBindings when it bails on ambiguous
                    // buffer aliasing — notably moreh_adamw, called in-place so its optional
                    // param_out == param_in etc. appear twice within the input region.  Fast-pathing
                    // such a bailed op would apply_resolved_bindings() nothing and leave the cached
                    // program pointing at stale first-miss addresses; route it to the slow-path
                    // rebuild instead, which re-derives all addresses AND scalars. (#48928)
                    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
                    if constexpr (requires {
                                      DeviceOperation::get_dynamic_runtime_args(
                                          attrs,
                                          tensor_args,
                                          tensor_return_value,
                                          std::optional<ttnn::MeshCoordinate>{});
                                  }) {
                        dynamic_args = DeviceOperation::get_dynamic_runtime_args(
                            attrs,
                            tensor_args,
                            tensor_return_value,
                            std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                    }
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
#ifdef TT_DESCRIPTOR_PATCHING_PARITY_CHECK
                        // Regression net: assert the fast path reproduced a full rebuild exactly (rt-args
                        // AND CB addresses). Fires loudly at the exact stale arg for any op whose cache-hit
                        // re-application is incomplete (SDXL in-place silu / MorehAdamW). Debug/CI only.
                        {
                            auto parity_desc = invoke_per_coord(
                                attrs,
                                tensor_args,
                                tensor_return_value,
                                std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                            tt::tt_metal::Program parity_scratch{parity_desc};
                            tt::tt_metal::apply_descriptor_runtime_args(parity_scratch, parity_desc);
                            tt::tt_metal::assert_fastpath_parity(
                                program, parity_scratch, parity_desc, ttsl::get_type_name<DeviceOperation>());
                        }
#endif
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
    // MetalV2MeshWorkloadFactoryAdapter
    //
    // Adapts a MetalV2FactoryConcept factory (Metal 2.0,
    // single-program / SPMD-flavored) for mesh dispatch. The op author writes
    // ONLY create_program_artifacts, returning a single ProgramArtifacts (one
    // ProgramSpec + ProgramRunArgs + any op-owned tensors). The adapter stamps a
    // Program from that spec onto each mesh coordinate range covered by the
    // workload — mirroring the descriptor adapter's per-range build pattern.
    //
    // On cache miss: the adapter calls create_program_artifacts, builds one
    // Program per coordinate range via experimental::MakeProgramFromSpec, applies
    // the initial ProgramRunArgs via SetProgramRunArgs, then resolves each
    // TensorArgument against the combined enumeration of io tensors (from
    // tensor_args / tensor_return_value) followed by the factory's op-owned
    // tensors (pointer-identity match within the call). Op-owned tensors are
    // parked in shared_variables so their device allocation outlives the miss and
    // stays at a stable address across dispatches.
    //
    // On cache hit: the adapter enumerates fresh io tensors, appends the parked
    // op-owned tensors, rebuilds a TensorArgument for every binding using the
    // stored indices, and applies via experimental::UpdateTensorArgs — no Program
    // rebuild. Op-owned tensors are re-patched even though their address is
    // unchanged, because UpdateTensorArgs is currently all-or-nothing; this
    // stepping-stone concept accepts that redundancy.
    //
    // Contract: every TensorArgument returned by the factory must reference a
    // MeshTensor reachable from tensor_args / tensor_return_value, or one of the
    // MeshTensors the factory placed in ProgramArtifacts::op_owned_tensors.
    //
    // TODO: consider replacing with a general MeshWorkloadSpecFactoryAdapter?
    // -----------------------------------------------------------------------
    template <MetalV2FactoryConcept MetalV2Factory>
    struct MetalV2MeshWorkloadFactoryAdapter {
        using TensorParamName = tt::tt_metal::experimental::TensorParamName;
        using TensorArgument = tt::tt_metal::experimental::ProgramRunArgs::TensorArgument;

        // Stored across cache entries: for each TensorArgument in a program's
        // ProgramRunArgs, which tensor (by index into the deterministic
        // enumeration of io tensors followed by op-owned tensors) it was bound
        // to. Pointer identity is only valid within a single call; the index is
        // stable across calls.
        struct ResolvedTensorBinding {
            TensorParamName tensor_parameter_name;
            std::size_t tensor_idx;
        };

        struct shared_variables_t {
            std::vector<ResolvedTensorBinding> bindings;
            // Op-owned tensors produced by the factory, parked here so the
            // (move-only) MeshTensors outlive the cache miss and every stamped
            // program can reference the same workload-wide set. Shared ownership
            // (not a per-range copy) keeps the device allocation alive for the
            // life of the cache entry.
            std::shared_ptr<std::vector<tt::tt_metal::MeshTensor>> op_owned_tensors;
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
        // combined enumeration (io tensors followed by op-owned tensors).
        // Cache-miss path only. TT_FATALs on a TensorArgument that references
        // neither an io tensor nor a factory op-owned tensor.
        //
        // NOTE on host perf: the index-based binding scheme is what makes a
        // fast cache-hit path possible, but the current straightforward
        // implementation isn't there yet — the enumeration returns a heap
        // std::vector, and apply_descriptor builds a fresh TensorArgument
        // table each dispatch. Both costs are fixable by mirroring the
        // descriptor adapter's compile-time-unrolled walker + SmallVector +
        // cached TensorArgument storage pattern. Deferred pending profiling.
        static std::vector<ResolvedTensorBinding> resolve_bindings(
            const tt::tt_metal::experimental::Table<TensorParamName, TensorArgument>& factory_tensor_args,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& mesh_tensors) {
            std::vector<ResolvedTensorBinding> bindings;
            bindings.reserve(factory_tensor_args.size());
            // The name is the Table key; the TensorArgument value carries only the tensor ref.
            for (const auto& [tensor_parameter_name, tensor_arg] : factory_tensor_args) {
                const auto* target = &tt::tt_metal::experimental::mesh_tensor_of(tensor_arg);
                auto it = std::find_if(mesh_tensors.begin(), mesh_tensors.end(), [target](const auto& wrapped) {
                    return &wrapped.get() == target;
                });
                TT_FATAL(
                    it != mesh_tensors.end(),
                    "TensorArgument '{}' must reference a MeshTensor reachable from tensor_args / "
                    "tensor_return_value, or one of the factory's op_owned_tensors (got an unowned MeshTensor)",
                    tensor_parameter_name);
                bindings.push_back(
                    {tensor_parameter_name, static_cast<std::size_t>(std::distance(mesh_tensors.begin(), it))});
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
                "MetalV2 factory adapter requires at least one Tensor in tensor_args to source the MeshDevice");
            auto* mesh_device = first_tensor.value().device();
            TT_FATAL(mesh_device != nullptr, "First tensor in tensor_args must be allocated on a MeshDevice");

            // The factory produces a single ProgramArtifacts; the adapter stamps it
            // across all coordinate ranges. Bindings derive from the (single) set of
            // factory tensor_args and are identical for every stamped program; copy
            // per range into the cached shared state.
            auto artifacts = MetalV2Factory::create_program_artifacts(attrs, tensor_args, tensor_return_value);

            // Enumerate io tensors (inputs + outputs), then append the factory's
            // op-owned tensors. resolve_bindings maps each TensorArgument to an
            // index in this combined order, which the cache-hit path reproduces.
            auto mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);
            for (const auto& op_owned_tensor : artifacts.op_owned_tensors) {
                mesh_tensors.push_back(std::cref(op_owned_tensor));
            }
            auto bindings = resolve_bindings(artifacts.run_params.tensor_args, mesh_tensors);

            // Park op-owned tensors in a shared vector so the move-only MeshTensors
            // outlive this call and every stamped program references the same set.
            // (A vector move preserves element addresses, so the references held by
            // artifacts.run_params stay valid for the SetProgramRunArgs calls below.)
            auto op_owned_tensors =
                std::make_shared<std::vector<tt::tt_metal::MeshTensor>>(std::move(artifacts.op_owned_tensors));

            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            for (const auto& range : tensor_coords.ranges()) {
                auto program = tt::tt_metal::experimental::MakeProgramFromSpec(*mesh_device, artifacts.spec);
                tt::tt_metal::experimental::SetProgramRunArgs(program, artifacts.run_params);
                shared_variables.emplace(
                    range, shared_variables_t{.bindings = bindings, .op_owned_tensors = op_owned_tensors});
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

                // Reproduce the cache-miss enumeration: fresh io tensors, then the
                // parked op-owned tensors (stable addresses, retrieved from the cache).
                auto mesh_tensors = io_mesh_tensors;
                if (sv.op_owned_tensors) {
                    for (const auto& op_owned_tensor : *sv.op_owned_tensors) {
                        mesh_tensors.push_back(std::cref(op_owned_tensor));
                    }
                }

                tt::tt_metal::experimental::Table<TensorParamName, TensorArgument> fresh_tensor_args;
                for (const auto& b : sv.bindings) {
                    fresh_tensor_args.emplace(b.tensor_parameter_name, TensorArgument{mesh_tensors[b.tensor_idx]});
                }
                tt::tt_metal::experimental::UpdateTensorArgs(program, fresh_tensor_args);
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

    // Exact, collision-free companion to compute_mesh_workload_hash: the cache compares this on a
    // hash hit so a 64-bit collision is resolved to a correct (rebuild) miss instead of a wrong
    // hit (issue #45821). It must mirror the SAME inputs the hash combines.
    //
    // The key is prefixed with op_type_name (the DeviceOperation's type name) so distinct ops can
    // never alias on a hash collision.
    // For ops with a custom compute_program_hash we can't infer which fields it keyed on (it may
    // deliberately exclude some, e.g. an RNG seed), so we return only the op-identity prefix --
    // opting that op out of attribute-level collision resolution. The default reflection-hash
    // path is mirrored exactly.
    static std::string compute_mesh_workload_canonical_key(
        [[maybe_unused]] tt::tt_metal::distributed::MeshDevice* mesh_device,
        std::string_view op_type_name,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        std::string key{op_type_name};
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
            key += ttsl::hash::canonical_key(attrs, tensor_args);
            for (const auto& coord :
                 mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
                key += ttsl::hash::canonical_key(coord);
            }
            return key;
        }
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
