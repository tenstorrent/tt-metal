// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/utils/program_cache.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/workload/legacy/workload_descriptor.hpp>
#include <tt-metalium/memory/buffer/mesh_buffer.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/run_args/validation_mode.hpp>
#include "ttnn/per_device_program_merge.hpp"
#include "ttnn/per_device_run_args.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
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
#include "ttnn/config.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::command_list {
struct CmdListBindings;
}  // namespace ttnn::command_list

namespace ttnn::device_operation {

template <typename T>
using AdaptedCachedMeshWorkload = tt::tt_metal::program_cache::detail::AdaptedCachedMeshWorkload<T>;

// The ValidationMode that the `validate_program_args` setting selects for program run-args updates.
inline tt::tt_metal::ValidationMode program_args_validation_mode() {
    return ttnn::CONFIG.get<"validate_program_args">() ? tt::tt_metal::ValidationMode::FULL
                                                       : tt::tt_metal::ValidationMode::SKIP;
}

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
struct extract_tensor_buffers_t<ttnn::Tensor, void> {
    template <typename Out>
    static void call(const ttnn::Tensor& t, Out& out) {
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
        ttsl::concepts::Reflectable<T> and not std::is_same_v<T, ttnn::Tensor> and not is_handled_container_v<T>>> {
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

    static bool skip_launch(
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        const tensor_return_value_t& tensor_return_value)
        requires HasSkipLaunch<DeviceOperation>
    {
        return DeviceOperation::skip_launch(attrs, tensor_args, tensor_return_value);
    }

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            // Fold type_hash so distinct ops cannot alias on a custom-hash collision
            return ttsl::hash::hash_objects_with_default_seed(
                ttsl::hash::type_hash<DeviceOperation>, DeviceOperation::compute_program_hash(attrs, tensor_args));
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
            // WorkloadDescriptor variant only, when resolve_bindings bailed: the tensor buffer addresses
            // baked into the cached program, which a cache hit can only reuse if they are unchanged.
            std::vector<uint32_t> unpatchable_buffer_addresses;
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

        static std::vector<uint32_t> buffer_addresses(std::span<tt::tt_metal::Buffer* const> buffers) {
            std::vector<uint32_t> addresses;
            addresses.reserve(buffers.size());
            for (const auto* buffer : buffers) {
                addresses.push_back(buffer != nullptr ? buffer->address() : 0);
            }
            return addresses;
        }

        static bool declares_tensor_buffer_bindings(
            const tt::tt_metal::ProgramDescriptor& desc, std::span<tt::tt_metal::Buffer* const> buffers) {
            for (const auto& kernel : desc.kernels) {
                if (!kernel.buffer_bindings.empty() || !kernel.common_buffer_bindings.empty()) {
                    return true;
                }
            }
            for (const auto& cb : desc.cbs) {
                const tt::tt_metal::Buffer* cb_buffer = cb.buffer;
                if (cb_buffer == nullptr && cb.tensor != nullptr) {
                    cb_buffer = cb.tensor->mesh_buffer().get_reference_buffer();
                }
                if (cb_buffer != nullptr && std::find(buffers.begin(), buffers.end(), cb_buffer) != buffers.end()) {
                    return true;
                }
            }
            return false;
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

        // Whether the factory re-applies ALL per-dispatch state itself on a program-cache hit via
        // override_runtime_arguments() — the descriptor-era analog of the legacy
        // override_runtime_arguments().  When present, the adapter calls it on every hit and uses
        // NEITHER resolve_bindings (address inference) NOR get_dynamic_runtime_args: the factory owns
        // the full re-derivation, correct by construction.  This is the target mechanism;
        // resolve_bindings and get_dynamic are the legacy paths being migrated out (and, eventually,
        // deleted with Metal 2.0 native bindings).
        //
        // The hook lives on the program factory, its natural home alongside create_descriptor: the
        // framework already knows which factory built the cached program, so a factory never has to
        // mirror select_program_factory to find out what it is patching.
        //
        // Only the factory is probed.  An override_runtime_arguments() on the DeviceOperation is
        // IGNORED — it compiles, and it is never invoked.
        static consteval bool has_override_runtime_arguments() {
            return requires(
                tt::tt_metal::Program& program,
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const std::optional<ttnn::MeshCoordinate>& coord) {
                DescriptorFactory::override_runtime_arguments(program, attrs, tensor_args, tensor_return_value, coord);
            };
        }

        // Same hook declared at DeviceOperation scope, which the adapter no longer probes.
        static consteval bool has_device_op_override_runtime_arguments() {
            return requires(
                tt::tt_metal::Program& program,
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const std::optional<ttnn::MeshCoordinate>& coord) {
                DeviceOperation::override_runtime_arguments(program, attrs, tensor_args, tensor_return_value, coord);
            };
        }

        // Leaving it on the operation is worse than not writing it: it compiles, is never called, and
        // every value it was meant to re-apply stays frozen at the first miss.
        static_assert(
            !has_device_op_override_runtime_arguments(),
            "override_runtime_arguments() belongs on the program factory, not the DeviceOperation. Only "
            "the factory is probed, so an operation-scope hook is silently dead code and its "
            "hash-excluded values freeze at the first cache miss. Move it onto the program factory.");

        static consteval bool has_get_dynamic_runtime_args() {
            return requires(
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const std::optional<ttnn::MeshCoordinate>& coord) {
                DeviceOperation::get_dynamic_runtime_args(attrs, tensor_args, tensor_return_value, coord);
            };
        }

        // A factory that owns its cache-hit re-derivation via override_runtime_arguments() must NOT be
        // paired with the legacy get_dynamic_runtime_args() on the DeviceOperation — override
        // supersedes it, and having both is ambiguous (which one re-applies?).  This assert forces
        // porting an op to DROP get_dynamic.
        static_assert(
            !(has_override_runtime_arguments() && has_get_dynamic_runtime_args()),
            "A program factory's override_runtime_arguments() must not be combined with the "
            "DeviceOperation's get_dynamic_runtime_args(): override_runtime_arguments supersedes the "
            "legacy hook. Delete get_dynamic_runtime_args() from this op.");

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

        // One program per coordinate (or per range), as the factory describes them. A wrapping MeshWorkloadFactory
        // calls this unless its cache-hit code handles merged ranges (see ttnn::MergePerDevicePrograms).
        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            return create_mesh_workload_impl<false>(attrs, tensor_coords, tensor_args, tensor_return_value);
        }

        // Also merges each range's per-coordinate programs into one when the factory sets merge_per_device_programs.
        // The framework calls this when the adapter runs the op itself.
        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            ttnn::MergePerDevicePrograms /*caller_handles_merged_ranges*/) {
            return create_mesh_workload_impl<merges_per_device_programs()>(
                attrs, tensor_coords, tensor_args, tensor_return_value);
        }

        template <bool merge>
        static auto create_mesh_workload_impl(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::distributed::MeshWorkload mesh_workload;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            auto& mesh_device = *mesh_device_operation_utils::get_mesh_device(attrs, tensor_args);

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
                std::unordered_map<ttnn::MeshCoordinateRange, std::vector<ttnn::PerDeviceRuntimeArgsField>>
                    per_device_runtime_args;
                if constexpr (merge) {
                    static_assert(
                        !has_override_runtime_arguments() &&
                            !requires {
                                DeviceOperation::get_dynamic_runtime_args(
                                    attrs, tensor_args, tensor_return_value, std::optional<ttnn::MeshCoordinate>{});
                            },
                        "merge_per_device_programs: the cache-hit path sees one program per range, so the op must not "
                        "re-derive per-coordinate state on a hit");
                    auto first_tensor = ttsl::reflection::get_first_object_of_type<ttnn::Tensor>(tensor_args);
                    if (!first_tensor.has_value()) {
                        first_tensor = ttsl::reflection::get_first_object_of_type<ttnn::Tensor>(tensor_return_value);
                    }
                    if (first_tensor.has_value() && first_tensor->device() != nullptr) {
                        per_device_runtime_args =
                            merge_programs_per_range(*first_tensor->device(), programs, tensor_coords);
                    }
                }
                for (auto& [device_range, desc] : programs) {
                    tt::tt_metal::Program program{mesh_device, desc};
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
                    // A descriptor that declares tensor-buffer bindings only resolves to empty bindings when
                    // resolve_bindings bailed on an aliased buffer.
                    std::vector<uint32_t> unpatchable_buffer_addresses;
                    if (bindings.empty() && declares_tensor_buffer_bindings(desc, collected.buffers)) {
                        unpatchable_buffer_addresses = buffer_addresses(collected.buffers);
                    }
                    mesh_workload.add_program(device_range, std::move(program));
                    if (const auto it = per_device_runtime_args.find(device_range);
                        it != per_device_runtime_args.end()) {
                        ttnn::apply_per_device_runtime_args(mesh_workload, device_range, it->second);
                    }
                    shared_variables[device_range] = shared_variables_t{
                        .workload_descriptor = workload_descriptor,
                        .resolved_bindings = std::move(bindings),
                        .unpatchable_buffer_addresses = std::move(unpatchable_buffer_addresses)};
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
                        tt::tt_metal::Program program{mesh_device, desc};
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
        // A WorkloadDescriptor factory may set `static constexpr bool merge_per_device_programs = true` to run the
        // per-coordinate programs of each range as one program with per-device runtime args (when they differ only in
        // runtime-arg values; see ttnn/per_device_program_merge.hpp), and may define
        // `static std::optional<std::vector<uint32_t>> idle_runtime_args(size_t kernel_index, const CoreCoord&)` for
        // kernels that run on different cores per device, and `static constexpr bool pad_per_device_runtime_args` when
        // its kernels read only as many runtime args as each device's describe, and
        // `static std::vector<ttnn::PerDeviceRuntimeArgsWords> per_device_runtime_args_words()` for words a wrapping
        // factory updates per device on a cache hit. The cache-hit path then sees one program per range, so the
        // factory must not re-derive per-coordinate state on a hit (no override_runtime_arguments, no dynamic args).
        // A wrapping MeshWorkloadFactory gets merged programs only by passing ttnn::MergePerDevicePrograms to
        // create_mesh_workload, since its own override then sees ranges of several devices.
        static consteval bool merges_per_device_programs() {
            if constexpr (requires { DescriptorFactory::merge_per_device_programs; }) {
                return DescriptorFactory::merge_per_device_programs;
            } else {
                return false;
            }
        }

        static ttnn::ProgramMergeOptions per_device_program_merge_options() {
            ttnn::ProgramMergeOptions options;
            if constexpr (requires { DescriptorFactory::pad_per_device_runtime_args; }) {
                options.pad_runtime_args = DescriptorFactory::pad_per_device_runtime_args;
            }
            if constexpr (requires { DescriptorFactory::per_device_runtime_args_words(); }) {
                options.per_device_words = DescriptorFactory::per_device_runtime_args_words();
            }
            if constexpr (requires(size_t k, const tt::tt_metal::CoreCoord& c) {
                              DescriptorFactory::idle_runtime_args(k, c);
                          }) {
                options.idle_runtime_args = [](size_t kernel_index, const tt::tt_metal::CoreCoord& core) {
                    return DescriptorFactory::idle_runtime_args(kernel_index, core);
                };
            }
            return options;
        }

        // Replaces each range's single-coordinate programs by one merged program where they merge; returns the
        // merged programs' per-device runtime args by range.
        static std::unordered_map<ttnn::MeshCoordinateRange, std::vector<ttnn::PerDeviceRuntimeArgsField>>
        merge_programs_per_range(
            tt::tt_metal::distributed::MeshDevice& mesh_device,
            std::vector<tt::tt_metal::WorkloadDescriptor::PerCoordProgram>& programs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            std::unordered_map<ttnn::MeshCoordinateRange, std::vector<ttnn::PerDeviceRuntimeArgsField>> fields;
            for (const auto& range : tensor_coords.ranges()) {
                if (range.shape().mesh_size() < 2 ||
                    !tt::tt_metal::distributed::ArePerDeviceRuntimeArgsSupported(mesh_device, range)) {
                    continue;
                }
                std::unordered_map<ttnn::MeshCoordinate, size_t> index_of;
                bool one_program_per_coordinate = true;
                for (size_t i = 0; i < programs.size(); ++i) {
                    if (programs[i].range.intersects(range) &&
                        (programs[i].range.shape().mesh_size() != 1 ||
                         !index_of.emplace(programs[i].range.start_coord(), i).second)) {
                        one_program_per_coordinate = false;  // a program spans coordinates, or two share one
                    }
                }
                if (!one_program_per_coordinate || index_of.size() != range.shape().mesh_size()) {
                    continue;
                }
                std::vector<tt::tt_metal::ProgramDescriptor> descriptors;
                descriptors.reserve(index_of.size());
                for (const auto& coord : range) {
                    descriptors.push_back(programs[index_of.at(coord)].descriptor);
                }
                auto merged =
                    ttnn::merge_per_device_program_descriptors(descriptors, per_device_program_merge_options());
                if (!merged.has_value()) {
                    continue;
                }
                std::vector<tt::tt_metal::WorkloadDescriptor::PerCoordProgram> kept;
                kept.reserve(programs.size() - index_of.size() + 1);
                for (auto& program : programs) {
                    const bool in_range =
                        program.range.shape().mesh_size() == 1 && range.contains(program.range.start_coord());
                    if (!in_range) {
                        kept.push_back(std::move(program));
                    }
                }
                kept.push_back(tt::tt_metal::WorkloadDescriptor::PerCoordProgram{
                    .range = range, .descriptor = std::move(merged->descriptor)});
                programs = std::move(kept);
                fields.emplace(range, std::move(merged->per_device_runtime_args));
            }
            return fields;
        }

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
                    } else if (!has_override_runtime_arguments() && !sv.unpatchable_buffer_addresses.empty()) {
                        // Overrides are exempt: some rewrite every buffer address themselves (e.g.
                        // minimal_matmul_strided_reduce_scatter_async), and the adapter cannot tell which.
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        TT_FATAL(
                            buffer_addresses(collected.buffers) == sv.unpatchable_buffer_addresses,
                            "{}: program cache hit on an entry built from a call that passed the same buffer in more "
                            "than one tensor slot, so its buffer addresses cannot be re-patched for the current "
                            "tensors. Pass distinct tensors to the first call (for example a clone of the repeated "
                            "tensor).",
                            ttsl::get_type_name<DeviceOperation>());
                    }
                    // Cache hit never rebuilds; re-apply hash-excluded args via the override hook.
                    if constexpr (has_override_runtime_arguments()) {
                        DescriptorFactory::override_runtime_arguments(
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
                    // ProgramDescriptor variant, factory owns its cache-hit re-derivation (the
                    // descriptor-era override_runtime_arguments()): re-apply ALL per-dispatch state —
                    // every runtime arg AND every tensor-backed CB address — for the current tensors.
                    // No resolve_bindings (address inference) and no get_dynamic; correct by
                    // construction for in-place, mixed-aliasing, and work-set shifts.
                    DescriptorFactory::override_runtime_arguments(
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
                        tt::tt_metal::Program parity_scratch{
                            *mesh_device_operation_utils::get_mesh_device(attrs, tensor_args), parity_desc};
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
                            tt::tt_metal::Program parity_scratch{
                                *mesh_device_operation_utils::get_mesh_device(attrs, tensor_args), parity_desc};
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
    // ProgramSpecMeshWorkloadFactoryAdapter
    //
    // Adapts a ProgramSpecFactoryConcept factory (Metal 2.0,
    // single-program / SPMD-flavored) for mesh dispatch. The op author writes
    // ONLY create_program_artifacts, returning a single ProgramArtifacts (one
    // ProgramSpec + ProgramRunArgs + any op-owned tensors). The adapter maps
    // that same ProgramSpec onto every range in tensor_coords via
    // the MeshWorkload constructor (occupancy = mesh devices that
    // hold a tensor shard). A factory whose programs vary across the mesh is a
    // MeshWorkloadSpecFactoryConcept instead.
    //
    // On cache miss: the adapter calls create_program_artifacts once, builds a
    // MeshWorkload via the MeshWorkload constructor, resolves each TensorArgument
    // against the combined enumeration of io tensors (from tensor_args /
    // tensor_return_value) followed by the factory's op-owned tensors
    // (pointer-identity match within the call), parks op-owned tensors in
    // shared_variables so their device allocation outlives the miss, then
    // applies the initial ProgramRunArgs via Program::set_program_run_args.
    //
    // On cache hit: the adapter enumerates fresh io tensors, appends the parked
    // op-owned tensors, rebuilds a TensorArgument for every binding using the
    // stored indices, and applies via Program::update_tensor_args — no Program
    // rebuild. Op-owned tensors are re-patched even though their address is
    // unchanged, because Program::update_tensor_args is currently all-or-nothing.
    // PrefetcherPipes and GlobalSemaphores are rebound from operation_attributes_t.
    //
    // Contract: every TensorArgument returned by the factory must reference a
    // MeshTensor reachable from tensor_args / tensor_return_value, or one of the
    // MeshTensors the factory placed in ProgramArtifacts::op_owned_tensors. Every
    // GlobalSemaphoreArgument must reference a GlobalSemaphore in
    // operation_attributes_t::global_semaphores, unless the factory's
    // override_runtime_arguments supplies its run args on every hit.
    // Binding machinery shared by every spec-path adapter: it depends only on the op's tensor
    // types, not on which factory concept produced the artifacts.
    struct SpecBindingSupport {
        using TensorParamName = tt::tt_metal::TensorParamName;
        using TensorArgument = tt::tt_metal::ProgramRunArgs::TensorArgument;
        using PrefetcherPipeParamName = tt::tt_metal::experimental::PrefetcherPipeParamName;
        using PrefetcherPipeArgument = tt::tt_metal::experimental::PrefetcherPipeArgument;
        using GlobalSemaphoreParamName = tt::tt_metal::GlobalSemaphoreParamName;
        using GlobalSemaphoreArgument = tt::tt_metal::ProgramRunArgs::GlobalSemaphoreArgument;

        // Stored across cache entries: for each TensorArgument in a program's ProgramRunArgs, which
        // tensor (by index into the deterministic enumeration of io tensors followed by op-owned
        // tensors) it was bound to. Pointer identity is only valid within a single call; the index
        // is stable across calls.
        struct ResolvedTensorBinding {
            TensorParamName tensor_parameter_name;
            std::size_t tensor_idx;
        };

        // The same for each PrefetcherPipeArgument: the index into operation_attributes_t::prefetcher_pipes
        // of the pipe it was bound to.
        struct ResolvedPipeBinding {
            PrefetcherPipeParamName pipe_parameter_name;
            std::size_t pipe_idx;
        };

        // An op whose programs bind PrefetcherPipes holds them in operation_attributes_t::prefetcher_pipes,
        // where the cache-hit path finds them again.
        static constexpr bool attrs_hold_prefetcher_pipes = requires(const operation_attributes_t& attrs) {
            { attrs.prefetcher_pipes } -> std::convertible_to<const ttnn::PrefetcherPipeList&>;
        };

        // The same for each GlobalSemaphoreArgument bound to one of the caller's GlobalSemaphores: the index
        // into operation_attributes_t::global_semaphores of the semaphore it was bound to. An argument bound to
        // one of the factory's parked GlobalSemaphores (MeshWorkloadArtifacts::global_semaphores) has none: its
        // address never changes.
        struct ResolvedGlobalSemaphoreBinding {
            GlobalSemaphoreParamName global_semaphore_parameter_name;
            std::size_t semaphore_idx;
        };

        // An op whose programs bind its caller's GlobalSemaphores holds them in
        // operation_attributes_t::global_semaphores, where the cache-hit path finds them again.
        static constexpr bool attrs_hold_global_semaphores = requires(const operation_attributes_t& attrs) {
            { attrs.global_semaphores } -> std::convertible_to<const std::vector<tt::tt_metal::GlobalSemaphore>&>;
        };

        // Shared by both spec adapters: the SPMD one and the per-range MeshWorkload one.
        struct SpecSharedVariables {
            std::vector<ResolvedTensorBinding> bindings;
            std::vector<ResolvedPipeBinding> pipe_bindings;
            std::vector<ResolvedGlobalSemaphoreBinding> global_semaphore_bindings;
            // The size of operation_attributes_t::global_semaphores on the cache miss, which every hit that
            // rebinds global_semaphore_bindings must match.
            std::size_t num_caller_global_semaphores = 0;
            // Op-owned tensors produced by the factory, parked here so the (move-only) MeshTensors
            // outlive the cache miss and every program on the occupied ranges can reference them.
            // Shared ownership keeps the device allocation alive for the life of the cache entry.
            std::shared_ptr<std::vector<tt::tt_metal::MeshTensor>> op_owned_tensors;
            // GlobalSemaphores a MeshWorkloadSpecFactoryConcept factory allocated for its programs, parked here
            // (shared by every range of the workload) so they stay allocated, at the addresses its programs were
            // given, for the life of the cache entry. Null if it allocated none.
            std::shared_ptr<const std::vector<tt::tt_metal::GlobalSemaphore>> global_semaphores;
            // Factories with per-device run args: the state of the range the factory returned, shared by its
            // cached programs (see MeshWorkloadSpecFactoryAdapter).
            std::shared_ptr<per_device_run_args::RangeState> per_device;
            // What a TTNN command-list recording may bind in this program. Set on the cache miss when the factory
            // defines map_cmd_list_parameters.
            std::shared_ptr<const ttnn::command_list::CmdListBindings> cmd_list_bindings;
        };

        // The tensor at `tensor_idx` of the cache-miss enumeration: this call's io tensors, then the parked
        // op-owned tensors (stable addresses, retrieved from the cache). Shared so the cache-hit refresh and
        // command-list recording cannot drift apart.
        static const tt::tt_metal::MeshTensor& bound_mesh_tensor(
            const SpecSharedVariables& sv,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& io_mesh_tensors,
            std::size_t tensor_idx) {
            return tensor_idx < io_mesh_tensors.size() ? io_mesh_tensors[tensor_idx].get()
                                                       : (*sv.op_owned_tensors)[tensor_idx - io_mesh_tensors.size()];
        }

        // What a TTNN command-list recording may bind in a program built from `spec` and `run_params`: the
        // factory's map_cmd_list_parameters, or null if it defines none. Called on the cache miss.
        template <typename Factory>
        static std::shared_ptr<const ttnn::command_list::CmdListBindings> make_cmd_list_bindings(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            const tt::tt_metal::ProgramSpec& spec,
            const tt::tt_metal::ProgramRunArgs& run_params) {
            if constexpr (detail::MapsCmdListParameters<Factory>) {
                auto bindings = Factory::map_cmd_list_parameters(attrs, tensor_args, spec, run_params);
                bindings.factory_name = ttsl::long_type_name<Factory>;
                return std::make_shared<const ttnn::command_list::CmdListBindings>(std::move(bindings));
            } else {
                return nullptr;
            }
        }

        // Cache-hit tensor refresh, shared so the two adapters' hit paths cannot drift apart.
        template <typename CachedWorkload>
        static void refresh_tensor_args(
            CachedWorkload& cached_workload,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            const auto io_mesh_tensors = collect_mesh_tensors(tensor_args, tensor_return_value);
            const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                const auto& sv = cached_workload.shared_variables.at(coordinate_range);
                tt::tt_metal::Table<TensorParamName, TensorArgument> fresh_tensor_args;
                for (const auto& b : sv.bindings) {
                    fresh_tensor_args.emplace(
                        b.tensor_parameter_name,
                        TensorArgument{std::cref(bound_mesh_tensor(sv, io_mesh_tensors, b.tensor_idx))});
                }
                program.update_tensor_args(fresh_tensor_args, validation);
            }
        }

        // Cache-hit pipe refresh: supply each program the pipes now at its resolved positions in
        // operation_attributes_t::prefetcher_pipes. A pipe keys the program hash by what a Program bakes in
        // from it, not by which pipe it is, so a hit may bring one re-carved in place of the pipe the
        // program bound, and supplying it rebinds the program. The pipe it already holds is a no-op.
        template <typename CachedWorkload>
        static void refresh_prefetcher_pipe_args(CachedWorkload& cached_workload, const operation_attributes_t& attrs) {
            if constexpr (attrs_hold_prefetcher_pipes) {
                const ttnn::PrefetcherPipeList& prefetcher_pipes = attrs.prefetcher_pipes;
                const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
                for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                    const auto& sv = cached_workload.shared_variables.at(coordinate_range);
                    if (sv.pipe_bindings.empty()) {
                        continue;
                    }
                    tt::tt_metal::ProgramRunArgs run_args;
                    for (const auto& b : sv.pipe_bindings) {
                        TT_FATAL(
                            b.pipe_idx < prefetcher_pipes.size(),
                            "PrefetcherPipeArgument '{}' was bound to prefetcher_pipes[{}] when this program was "
                            "cached, but the hit supplies {} pipes; the program hash must cover prefetcher_pipes",
                            b.pipe_parameter_name.get(),
                            b.pipe_idx,
                            prefetcher_pipes.size());
                        run_args.advanced_options.prefetcher_pipe_args.emplace(
                            b.pipe_parameter_name, PrefetcherPipeArgument{*prefetcher_pipes[b.pipe_idx]});
                    }
                    program.update_program_run_args(run_args, validation);
                }
            }
        }

        // Cache-hit GlobalSemaphore refresh: supply each program the caller's GlobalSemaphores now at its
        // resolved positions in operation_attributes_t::global_semaphores. A GlobalSemaphore keys the program hash
        // by its cores and buffer type, not by which semaphore it is, so a hit may bring other semaphores than the
        // ones the program bound, and supplying them rebinds the program. Parked semaphores keep their addresses.
        template <typename CachedWorkload>
        static void refresh_global_semaphore_args(
            CachedWorkload& cached_workload, const operation_attributes_t& attrs) {
            if constexpr (attrs_hold_global_semaphores) {
                const std::vector<tt::tt_metal::GlobalSemaphore>& caller_semaphores = attrs.global_semaphores;
                const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
                for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                    const auto& sv = cached_workload.shared_variables.at(coordinate_range);
                    if (sv.global_semaphore_bindings.empty()) {
                        continue;
                    }
                    TT_FATAL(
                        caller_semaphores.size() == sv.num_caller_global_semaphores,
                        "This program bound operation_attributes_t::global_semaphores by position when it was cached, "
                        "with {} semaphores there, but the hit supplies {}; the program hash must cover how many there "
                        "are",
                        sv.num_caller_global_semaphores,
                        caller_semaphores.size());
                    tt::tt_metal::ProgramRunArgs run_args;
                    for (const auto& b : sv.global_semaphore_bindings) {
                        run_args.global_semaphore_args.emplace(
                            b.global_semaphore_parameter_name, std::cref(caller_semaphores[b.semaphore_idx]));
                    }
                    program.update_program_run_args(run_args, validation);
                }
            }
        }

        // Cache-hit refresh of everything a program binds from operation_attributes_t (PrefetcherPipes and the
        // caller's GlobalSemaphores), shared so that no hit path can leave one of them stale.
        template <typename CachedWorkload>
        static void refresh_attrs_bound_args(CachedWorkload& cached_workload, const operation_attributes_t& attrs) {
            refresh_prefetcher_pipe_args(cached_workload, attrs);
            refresh_global_semaphore_args(cached_workload, attrs);
        }

        // The Metal 2.0 MeshWorkload constructor needs a MeshDevice; pull it from the first device
        // tensor reachable from tensor_args, falling back to tensor_return_value for output-only
        // ops (e.g. rand) whose tensor_args carry no input tensor, and then, as launch does, to the
        // MeshDevice* in operation_attributes of an op with no tensor at all.
        static tt::tt_metal::distributed::MeshDevice* source_mesh_device(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value) {
            auto first_tensor = ttsl::reflection::get_first_object_of_type<ttnn::Tensor>(tensor_args);
            if (!first_tensor.has_value()) {
                first_tensor = ttsl::reflection::get_first_object_of_type<ttnn::Tensor>(tensor_return_value);
            }
            if (!first_tensor.has_value()) {
                const auto attrs_mesh_device = ttsl::reflection::get_first_object_of_type<ttnn::MeshDevice*>(attrs);
                TT_FATAL(
                    attrs_mesh_device.has_value() && attrs_mesh_device.value() != nullptr,
                    "ProgramSpec factory adapter requires a Tensor in tensor_args or tensor_return_value, or a "
                    "MeshDevice* in operation_attributes, to source the MeshDevice");
                return attrs_mesh_device.value();
            }
            auto* mesh_device = first_tensor.value().device();
            TT_FATAL(
                mesh_device != nullptr,
                "The sourced tensor (from tensor_args or tensor_return_value) must be allocated on a MeshDevice");
            return mesh_device;
        }

        // Walk tensor_args and tensor_return_value via reflection, collecting the MeshTensor of
        // every Tensor leaf. The walk order is deterministic (reflection-driven, stable across
        // calls), so the resulting indices are stable across calls.
        static std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> collect_mesh_tensors(
            const tensor_args_t& tensor_args, const tensor_return_value_t& tensor_return_value) {
            std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> result;
            const auto visit = [&result](const ttnn::Tensor& t) { result.push_back(std::cref(t.mesh_tensor())); };
            ttsl::reflection::visit_object_of_type<ttnn::Tensor>(visit, tensor_args);
            ttsl::reflection::visit_object_of_type<ttnn::Tensor>(visit, tensor_return_value);
            return result;
        }

        // Match each TensorArgument's MeshTensor reference back to its index in the combined
        // enumeration (io tensors followed by op-owned tensors). Cache-miss path only.
        //
        // When several slots hold the same MeshTensor, pointer identity cannot tell which of them a
        // TensorArgument came from (ProgramRunArgs::tensor_args is a Table with no promised order).
        // Any of them is correct for both readers of these bindings on a hit, the tensor refresh and
        // command-list recording: compute_mesh_workload_cache_key keys such ops by which tensor_args
        // slots alias (compute_tensor_alias_signature), so every hit of this entry repeats the miss's
        // aliasing.
        //
        // NOTE on host perf: the index-based binding scheme is what makes a fast cache-hit path
        // possible, but the current straightforward implementation isn't there yet -- the
        // enumeration returns a heap std::vector, and apply_descriptor builds a fresh
        // TensorArgument table each dispatch. Both costs are fixable by mirroring the descriptor
        // adapter's compile-time-unrolled walker + SmallVector + cached storage pattern.
        static std::vector<ResolvedTensorBinding> resolve_bindings(
            const tt::tt_metal::Table<TensorParamName, TensorArgument>& factory_tensor_args,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& mesh_tensors) {
            std::vector<ResolvedTensorBinding> bindings;
            bindings.reserve(factory_tensor_args.size());
            // The name is the Table key; the TensorArgument value carries only the tensor ref.
            for (const auto& [tensor_parameter_name, tensor_arg] : factory_tensor_args) {
                const auto* target = &tt::tt_metal::mesh_tensor_of(tensor_arg);
                const auto it = std::find_if(
                    mesh_tensors.begin(), mesh_tensors.end(), [target](const auto& t) { return &t.get() == target; });
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

        // Match each PrefetcherPipeArgument to its pipe's index in operation_attributes_t::prefetcher_pipes,
        // so that every cache hit rebinds it (refresh_prefetcher_pipe_args). Throws when an argument's
        // pipe is not there, since a hit could not supply it. Cache-miss path only.
        static std::vector<ResolvedPipeBinding> resolve_pipe_bindings(
            const tt::tt_metal::Table<PrefetcherPipeParamName, PrefetcherPipeArgument>& factory_pipe_args,
            const operation_attributes_t& attrs) {
            if (factory_pipe_args.empty()) {
                return {};
            }
            if constexpr (attrs_hold_prefetcher_pipes) {
                const ttnn::PrefetcherPipeList& prefetcher_pipes = attrs.prefetcher_pipes;
                std::vector<ResolvedPipeBinding> bindings;
                bindings.reserve(factory_pipe_args.size());
                for (const auto& [pipe_parameter_name, pipe_arg] : factory_pipe_args) {
                    const auto it = std::find_if(
                        prefetcher_pipes.begin(), prefetcher_pipes.end(), [&pipe_arg](const auto& prefetcher_pipe) {
                            return prefetcher_pipe.get() == &pipe_arg.get();
                        });
                    TT_FATAL(
                        it != prefetcher_pipes.end(),
                        "PrefetcherPipeArgument '{}' must reference a pipe in operation_attributes_t::"
                        "prefetcher_pipes, which a cache hit rebinds it from",
                        pipe_parameter_name.get());
                    bindings.push_back(
                        {pipe_parameter_name, static_cast<std::size_t>(std::distance(prefetcher_pipes.begin(), it))});
                }
                return bindings;
            } else {
                TT_THROW(
                    "The spec factory binds PrefetcherPipes, but operation_attributes_t has no prefetcher_pipes "
                    "(a ttnn::PrefetcherPipeList) for a cache hit to rebind them from");
            }
        }

        // Match each GlobalSemaphoreArgument to one of the factory's parked GlobalSemaphores (nothing to rebind)
        // or to its index in operation_attributes_t::global_semaphores, so that every cache hit rebinds it
        // (refresh_global_semaphore_args). An argument in neither is the factory's own business if its
        // override_runtime_arguments supplies the run args on every hit (generic_op passes its caller's run args
        // through this way), and an error otherwise: the hit would keep a semaphore that may since have been freed.
        // Cache-miss path only.
        static std::vector<ResolvedGlobalSemaphoreBinding> resolve_global_semaphore_bindings(
            const tt::tt_metal::Table<GlobalSemaphoreParamName, GlobalSemaphoreArgument>& factory_semaphore_args,
            const operation_attributes_t& attrs,
            const std::vector<tt::tt_metal::GlobalSemaphore>* parked_semaphores,
            bool hit_overrides_run_args) {
            std::vector<ResolvedGlobalSemaphoreBinding> bindings;
            for (const auto& [semaphore_parameter_name, semaphore_arg] : factory_semaphore_args) {
                const auto* target = &semaphore_arg.get();
                const auto is_target = [target](const tt::tt_metal::GlobalSemaphore& s) { return &s == target; };
                if (parked_semaphores != nullptr &&
                    std::any_of(parked_semaphores->begin(), parked_semaphores->end(), is_target)) {
                    continue;
                }
                if constexpr (attrs_hold_global_semaphores) {
                    const std::vector<tt::tt_metal::GlobalSemaphore>& caller_semaphores = attrs.global_semaphores;
                    const auto it = std::find_if(caller_semaphores.begin(), caller_semaphores.end(), is_target);
                    if (it != caller_semaphores.end()) {
                        bindings.push_back(
                            {semaphore_parameter_name,
                             static_cast<std::size_t>(std::distance(caller_semaphores.begin(), it))});
                        continue;
                    }
                }
                TT_FATAL(
                    hit_overrides_run_args,
                    "GlobalSemaphoreArgument '{}' must reference a GlobalSemaphore in "
                    "operation_attributes_t::global_semaphores, which a cache hit rebinds it from, or (for a "
                    "MeshWorkloadSpecFactoryConcept factory) one in MeshWorkloadArtifacts::global_semaphores, which "
                    "the program cache keeps allocated; or the factory's override_runtime_arguments must supply it on "
                    "every cache hit",
                    semaphore_parameter_name.get());
            }
            return bindings;
        }

        // The size of operation_attributes_t::global_semaphores, recorded on the cache miss.
        static std::size_t num_caller_global_semaphores(const operation_attributes_t& attrs) {
            if constexpr (attrs_hold_global_semaphores) {
                return attrs.global_semaphores.size();
            } else {
                return 0;
            }
        }
    };

    // The shared variables of a workload built by a spec adapter, keyed by coordinate range.
    using SpecSharedVariablesMap =
        std::unordered_map<ttnn::MeshCoordinateRange, typename SpecBindingSupport::SpecSharedVariables>;

    // -----------------------------------------------------------------------
    template <typename SpecFactory>
    struct ProgramSpecMeshWorkloadFactoryAdapter : SpecBindingSupport {
        using TensorParamName = tt::tt_metal::TensorParamName;
        using TensorArgument = tt::tt_metal::ProgramRunArgs::TensorArgument;
        using ResolvedTensorBinding = typename SpecBindingSupport::ResolvedTensorBinding;

        // MeshWorkloadSpecFactoryConcept keys on &T::create_mesh_workload_artifacts, which is
        // ill-formed for an overload set, so such a factory lands here and is silently replicated.
        static_assert(
            !requires(
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const ttnn::MeshCoordinateRangeSet& tensor_coords) {
                SpecFactory::create_mesh_workload_artifacts(attrs, tensor_args, tensor_return_value, tensor_coords);
            },
            "This factory is callable as create_mesh_workload_artifacts but did not match "
            "MeshWorkloadSpecFactoryConcept, so its per-coordinate programs would be silently replaced by one "
            "spec replicated across the mesh. Declare exactly one create_mesh_workload_artifacts signature.");

        using shared_variables_t = typename SpecBindingSupport::SpecSharedVariables;
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        // Resolve one artifacts' tensor bindings and park its op-owned tensors. Consumes
        // artifacts.op_owned_tensors; a vector move keeps the run_params references valid.
        static shared_variables_t make_shared_variables(
            const operation_attributes_t& attrs,
            ProgramArtifacts& artifacts,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& io_mesh_tensors) {
            auto mesh_tensors = io_mesh_tensors;
            for (const auto& op_owned_tensor : artifacts.op_owned_tensors) {
                mesh_tensors.push_back(std::cref(op_owned_tensor));
            }
            return shared_variables_t{
                .bindings = SpecBindingSupport::resolve_bindings(artifacts.run_params.tensor_args, mesh_tensors),
                .pipe_bindings = SpecBindingSupport::resolve_pipe_bindings(
                    artifacts.run_params.advanced_options.prefetcher_pipe_args, attrs),
                .global_semaphore_bindings = SpecBindingSupport::resolve_global_semaphore_bindings(
                    artifacts.run_params.global_semaphore_args,
                    attrs,
                    /*parked_semaphores=*/nullptr,
                    /*hit_overrides_run_args=*/detail::HasSpecRuntimeArgsOverride<SpecFactory>),
                .num_caller_global_semaphores = SpecBindingSupport::num_caller_global_semaphores(attrs),
                .op_owned_tensors =
                    std::make_shared<std::vector<tt::tt_metal::MeshTensor>>(std::move(artifacts.op_owned_tensors))};
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto* mesh_device = SpecBindingSupport::source_mesh_device(attrs, tensor_args, tensor_return_value);

            // Enumerate io tensors (inputs + outputs); each artifacts appends its own
            // op-owned tensors. resolve_bindings maps each TensorArgument to an index in
            // this combined order, which the cache-hit path reproduces.
            const auto io_mesh_tensors = SpecBindingSupport::collect_mesh_tensors(tensor_args, tensor_return_value);

            // What each occupied range runs, plus the TTNN cache state (bindings and parked
            // op-owned tensors) the cache-hit path needs — held independently of the MeshWorkload.
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramSpec> program_specs;
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramRunArgs> run_params;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

            // One ProgramArtifacts mapped onto every occupied range; bindings and op-owned tensors
            // are identical for every program built from it.
            auto artifacts = SpecFactory::create_program_artifacts(attrs, tensor_args, tensor_return_value);
            auto sv = make_shared_variables(attrs, artifacts, io_mesh_tensors);
            sv.cmd_list_bindings = SpecBindingSupport::template make_cmd_list_bindings<SpecFactory>(
                attrs, tensor_args, artifacts.spec, artifacts.run_params);
            for (const auto& range : tensor_coords.ranges()) {
                shared_variables.emplace(range, sv);
                program_specs.emplace(range, artifacts.spec);
                run_params.emplace(range, artifacts.run_params);
            }

            // Cache miss is the cold path: always validate here, so every cached program was
            // built from a checked spec. validate_program_args only gates the hit-path re-checks.
            tt::tt_metal::distributed::MeshWorkload mesh_workload(*mesh_device, program_specs);
            for (auto& [range, program] : mesh_workload.get_programs()) {
                program.set_program_run_args(run_params.at(range));
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        // The framework's cache-hit dispatcher (use_cached_mesh_workload in
        // device_operation.hpp) prefers a method named `apply_descriptor` over
        // `override_runtime_arguments` when both exist. We adopt the name to
        // slot into that hook directly — the "descriptor" word here is the
        // dispatcher's historical naming, not a reference to ProgramDescriptor.
        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            SpecBindingSupport::refresh_tensor_args(cached_workload, tensor_args, tensor_return_value);
            SpecBindingSupport::refresh_attrs_bound_args(cached_workload, attrs);
        }
    };

    // Like ProgramSpecMeshWorkloadFactoryAdapter (cache-miss build inherited unchanged), but the
    // cache-hit path calls the factory's override_runtime_arguments and applies the returned
    // ProgramRunArgs via Program::update_program_run_args instead of the base's tensor-only
    // Program::update_tensor_args. PrefetcherPipes and the caller's GlobalSemaphores are rebound from
    // operation_attributes_t::prefetcher_pipes and ::global_semaphores either way.
    // The miss path applies create_program_artifacts' run args, not the override, so per-coordinate run args must
    // come from a MeshWorkloadSpecFactoryConcept factory's create_mesh_workload_artifacts.
    template <CustomProgramSpecFactoryConcept CustomSpecFactory>
    struct CustomProgramSpecMeshWorkloadFactoryAdapter : ProgramSpecMeshWorkloadFactoryAdapter<CustomSpecFactory> {
        using Base = ProgramSpecMeshWorkloadFactoryAdapter<CustomSpecFactory>;
        using typename Base::cached_mesh_workload_t;

        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            Base::refresh_attrs_bound_args(cached_workload, attrs);
            const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto run_args = CustomSpecFactory::override_runtime_arguments(
                    attrs,
                    tensor_args,
                    tensor_return_value,
                    std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                program.update_program_run_args(run_args, validation);
            }
        }
    };

    // -----------------------------------------------------------------------
    // MeshWorkloadSpecFactoryAdapter
    //
    // Adapts a MeshWorkloadSpecFactoryConcept factory: one create_mesh_workload_artifacts call
    // returns the workload-scoped resources plus a ProgramSpec and ProgramRunArgs per coordinate
    // range, which the adapter realises into a MeshWorkload via the MeshWorkload constructor.
    //
    // Resources (MeshWorkloadArtifacts::global_semaphores) are parked once and shared by every range's
    // shared_variables.
    //
    // On cache hit the adapter refreshes tensor bindings per range, or calls the factory's
    // override_runtime_arguments(attrs, args, ret, range) if it declares one. The range is passed
    // whole: one Program backs it, so the run args it returns apply to every device in it.
    // PrefetcherPipes and the caller's GlobalSemaphores are rebound from operation_attributes_t::prefetcher_pipes and
    // ::global_semaphores either way; parked GlobalSemaphores keep their addresses.
    //
    // Per-device run args. A factory whose devices differ only in a few named kernel run args declares
    // `static constexpr bool uses_per_device_run_args = true`, lists each device's values in
    // PerCoordProgram::per_device_run_params. Each range then runs:
    // - merged: one program with per-device run args (MeshWorkload::update_program_run_args_for_device), when the
    //   range has several devices and ArePerDeviceRuntimeArgsSupported holds;
    // - split: otherwise, one program per coordinate, each with the base run args overlaid with its device's values.
    // The choice is made at the cache miss from the declaration, not from the miss's values. A merged entry cannot be
    // hit during light-metal capture (clear the program cache first); entries built during capture stay split. Every
    // arg you ever set per device also goes in the base run args, on every call (checked at the miss, and on every
    // hit against what the previous call set).
    //
    // On a cache hit, the factory declares exactly one of:
    // - `static RangeRunArgs override_runtime_arguments(attrs, args, ret, range)` (the default): each hit returns the
    //   base run args plus the per-device values. For values that change between calls (a seed). Keep the base values
    //   under per-device args constant across calls: a changed base word under a per-device arg rebuilds the
    //   per-device commands at the next enqueue.
    // - `static constexpr bool fixed_per_device_run_args = true` (opt-in, no override): the per-device values are set
    //   once, at the cache miss, and a hit only refreshes the tensor bindings (Program::update_tensor_args keeps
    //   per-device values). Only for values that depend on nothing but the device's coordinate and the op's hashed
    //   attributes (a rank, a fabric route): every hit keeps the miss's values.
    // A per-device factory with neither does not compile, so a forgotten override cannot silently keep the miss's
    // values.
    // -----------------------------------------------------------------------
    template <MeshWorkloadSpecFactoryConcept WorkloadSpecFactory>
    struct MeshWorkloadSpecFactoryAdapter : SpecBindingSupport {
        using TensorParamName = tt::tt_metal::TensorParamName;
        using TensorArgument = tt::tt_metal::ProgramRunArgs::TensorArgument;
        using ResolvedTensorBinding = typename SpecBindingSupport::ResolvedTensorBinding;

        using shared_variables_t = typename SpecBindingSupport::SpecSharedVariables;
        using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

        static consteval bool has_override_runtime_arguments() {
            return requires(
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value,
                const ttnn::MeshCoordinateRange& range) {
                WorkloadSpecFactory::override_runtime_arguments(attrs, tensor_args, tensor_return_value, range);
            };
        }

        // Otherwise a near-miss signature leaves run args stale on every hit with nothing to say why.
        static_assert(
            !detail::HasSpecRuntimeArgsOverride<WorkloadSpecFactory> || has_override_runtime_arguments(),
            "override_runtime_arguments must take (operation_attributes_t, tensor_args_t, tensor_return_value_t, "
            "const MeshCoordinateRange&) to be called on the cache-hit path. It is called once per range, "
            "not once per device.");
        static_assert(
            !detail::HasRangeRunArgsOverride<WorkloadSpecFactory> || has_override_runtime_arguments(),
            "override_runtime_arguments returning RangeRunArgs must take (operation_attributes_t, tensor_args_t, "
            "tensor_return_value_t, const MeshCoordinateRange&). It is called once per range, not once per device.");

        static consteval bool uses_per_device_run_args() {
            if constexpr (requires { WorkloadSpecFactory::uses_per_device_run_args; }) {
                static_assert(
                    std::same_as<std::remove_cvref_t<decltype(WorkloadSpecFactory::uses_per_device_run_args)>, bool>,
                    "`uses_per_device_run_args` must be a `static constexpr bool`.");
                return WorkloadSpecFactory::uses_per_device_run_args;
            } else {
                return false;
            }
        }
        // Per-device values set once, at the cache miss: the factory opts in with `fixed_per_device_run_args = true`.
        static consteval bool fixed_per_device_run_args() {
            if constexpr (requires { WorkloadSpecFactory::fixed_per_device_run_args; }) {
                static_assert(
                    std::same_as<std::remove_cvref_t<decltype(WorkloadSpecFactory::fixed_per_device_run_args)>, bool>,
                    "`fixed_per_device_run_args` must be a `static constexpr bool`.");
                return WorkloadSpecFactory::fixed_per_device_run_args;
            } else {
                return false;
            }
        }
        static_assert(
            !fixed_per_device_run_args() || uses_per_device_run_args(),
            "`fixed_per_device_run_args = true` needs `uses_per_device_run_args = true`.");
        // The call check also catches an overloaded or templated override, whose address is ill-formed.
        static_assert(
            !fixed_per_device_run_args() ||
                (!requires { &WorkloadSpecFactory::override_runtime_arguments; } && !has_override_runtime_arguments()),
            "A factory with `fixed_per_device_run_args = true` keeps the cache miss's per-device values and must not "
            "declare override_runtime_arguments.");
        // Off by default: a per-device factory that forgets its override must not keep the miss's values on every hit.
        static_assert(
            !uses_per_device_run_args() || fixed_per_device_run_args() ||
                (detail::HasRangeRunArgsOverride<WorkloadSpecFactory> && has_override_runtime_arguments()),
            "A factory with `uses_per_device_run_args = true` declares "
            "`static RangeRunArgs override_runtime_arguments(attrs, args, ret, const MeshCoordinateRange&)`, or, if "
            "its per-device values depend only on the coordinate and the hashed attributes, "
            "`static constexpr bool fixed_per_device_run_args = true`.");
        static_assert(
            !detail::HasRangeRunArgsOverride<WorkloadSpecFactory> || uses_per_device_run_args(),
            "Only a factory with `uses_per_device_run_args = true` overrides with "
            "`static RangeRunArgs override_runtime_arguments(..., const MeshCoordinateRange&)`.");

        // Per-device run args: realises the factory's programs into a workload, merged or split per range.
        static cached_mesh_workload_t create_per_device_mesh_workload(
            tt::tt_metal::distributed::MeshDevice& mesh_device,
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            MeshWorkloadArtifacts& artifacts,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& mesh_tensors,
            const std::shared_ptr<const std::vector<tt::tt_metal::GlobalSemaphore>>& parked_semaphores) {
            namespace pd = per_device_run_args;
            // Checked on the ranges as returned: splitting would turn an overlap into a duplicate coordinate.
            for (size_t i = 0; i < artifacts.programs.size(); ++i) {
                const auto& range = artifacts.programs[i].range;
                for (const auto& coord : range) {
                    const bool covered = std::any_of(
                        tensor_coords.ranges().begin(),
                        tensor_coords.ranges().end(),
                        [&coord](const auto& tensor_range) { return tensor_range.contains(coord); });
                    TT_FATAL(
                        covered,
                        "create_mesh_workload_artifacts returned range {} covering {}, which is outside tensor_coords "
                        "{}: that program would bind tensors the device holds no shard of",
                        range,
                        coord,
                        tensor_coords);
                }
                for (size_t j = 0; j < i; ++j) {
                    TT_FATAL(
                        !range.intersects(artifacts.programs[j].range),
                        "create_mesh_workload_artifacts returned overlapping ranges {} and {}",
                        artifacts.programs[j].range,
                        range);
                }
            }
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramSpec> program_specs;
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramRunArgs> run_params;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            std::vector<std::pair<ttnn::MeshCoordinateRange, std::vector<DeviceRunArgs>>> merged_values;
            for (auto& per_coord : artifacts.programs) {
                const auto& range = per_coord.range;
                auto schema = pd::arg_schema_of(per_coord.spec);
                auto keys = pd::validate_and_collect_keys(range, per_coord.per_device_run_params, schema);
                pd::check_base_covers(per_coord.run_params, {}, keys, range);
                auto state = std::make_shared<pd::RangeState>(
                    pd::RangeState{.factory_range = range, .schema = std::move(schema)});
                state->merged = range.shape().mesh_size() > 1 && !pd::force_split() &&
                                tt::tt_metal::distributed::ArePerDeviceRuntimeArgsSupported(mesh_device, range);
                state->prev_keys = std::move(keys);
                const shared_variables_t sv{
                    .bindings = SpecBindingSupport::resolve_bindings(per_coord.run_params.tensor_args, mesh_tensors),
                    .pipe_bindings = SpecBindingSupport::resolve_pipe_bindings(
                        per_coord.run_params.advanced_options.prefetcher_pipe_args, attrs),
                    .global_semaphore_bindings = SpecBindingSupport::resolve_global_semaphore_bindings(
                        per_coord.run_params.global_semaphore_args,
                        attrs,
                        parked_semaphores.get(),
                        /*hit_overrides_run_args=*/has_override_runtime_arguments()),
                    .num_caller_global_semaphores = SpecBindingSupport::num_caller_global_semaphores(attrs),
                    .global_semaphores = parked_semaphores,
                    .per_device = state};
                if (state->merged) {
                    state->program_ranges = {range};
                    program_specs.emplace(range, std::move(per_coord.spec));
                    run_params.emplace(range, std::move(per_coord.run_params));
                    shared_variables.emplace(range, sv);
                    merged_values.emplace_back(range, std::move(per_coord.per_device_run_params));
                } else {
                    for (const auto& coord : range) {
                        const ttnn::MeshCoordinateRange coord_range(coord);
                        state->program_ranges.push_back(coord_range);
                        program_specs.emplace(coord_range, per_coord.spec);
                        const auto* values = pd::values_of(per_coord.per_device_run_params, coord);
                        run_params.emplace(
                            coord_range,
                            values != nullptr ? pd::overlay(per_coord.run_params, *values) : per_coord.run_params);
                        shared_variables.emplace(coord_range, sv);
                    }
                }
            }
            tt::tt_metal::distributed::MeshWorkload mesh_workload(mesh_device, program_specs);
            for (auto& [range, program] : mesh_workload.get_programs()) {
                program.set_program_run_args(run_params.at(range));
            }
            for (const auto& [range, values] : merged_values) {
                for (const auto& [device, device_values] : values) {
                    mesh_workload.update_program_run_args_for_device(range, device, device_values);
                }
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        // A merged entry needs per-device run args now; checked on every hit, before any program changes.
        template <typename State>
        static void check_merged_entry_usable(const State& state, tt::tt_metal::distributed::MeshDevice& mesh_device) {
            TT_FATAL(
                !state.merged || !per_device_run_args::force_split(),
                "A forced split (per_device_run_args::ScopedForceSplit) hit a merged program cache entry");
            TT_FATAL(
                !state.merged ||
                    tt::tt_metal::distributed::ArePerDeviceRuntimeArgsSupported(mesh_device, state.factory_range),
                "This program cache entry runs range {} as one program with per-device run args, which are not "
                "supported now (light-metal capture?): clear the program cache first",
                state.factory_range);
        }

        // Per-device run args with `fixed_per_device_run_args`, cache hit: the per-device values set at the miss stay;
        // only the tensor bindings are refreshed (Program::update_tensor_args keeps per-device values).
        static void apply_fixed_per_device_run_args(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto* mesh_device = SpecBindingSupport::source_mesh_device(attrs, tensor_args, tensor_return_value);
            for (const auto& [coordinate_range, sv] : cached_workload.shared_variables) {
                check_merged_entry_usable(*sv.per_device, *mesh_device);
            }
            SpecBindingSupport::refresh_attrs_bound_args(cached_workload, attrs);
            SpecBindingSupport::refresh_tensor_args(cached_workload, tensor_args, tensor_return_value);
        }

        // Per-device run args with an override, cache hit: one override call per factory range; the adapter's checks
        // (devices, kinds, coverage, support) run on every result before any program changes.
        static void apply_per_device_run_args(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            namespace pd = per_device_run_args;
            auto& workload = cached_workload.workload;
            auto& programs = workload.get_programs();
            auto* mesh_device = SpecBindingSupport::source_mesh_device(attrs, tensor_args, tensor_return_value);
            struct Pending {
                pd::RangeState* state;
                RangeRunArgs run_args;
                pd::KeySet keys;
            };
            std::vector<Pending> pending;
            for (const auto& [coordinate_range, program] : programs) {
                auto& state = *cached_workload.shared_variables.at(coordinate_range).per_device;
                if (coordinate_range != state.program_ranges.front()) {
                    continue;  // the range's first program stands for all of them
                }
                check_merged_entry_usable(state, *mesh_device);
                auto run_args = WorkloadSpecFactory::override_runtime_arguments(
                    attrs, tensor_args, tensor_return_value, state.factory_range);
                auto keys = pd::validate_and_collect_keys(state.factory_range, run_args.per_device, state.schema);
                pd::check_base_covers(run_args.base, state.prev_keys, keys, state.factory_range);
                pending.push_back({&state, std::move(run_args), std::move(keys)});
            }
            // The per-device values were checked against the program's schema, so metal accepts them; should a call
            // still fail part-way (a bad base value, say), each range counts as having per-device values of the union
            // of its old and new keys until every range is done: the next call's base must still list (and so drop)
            // all of them. Only checked keys enter the union, so a rejected call cannot leave a key no base can list.
            for (auto& [state, run_args, keys] : pending) {
                state->prev_keys.insert(keys.begin(), keys.end());
            }
            SpecBindingSupport::refresh_attrs_bound_args(cached_workload, attrs);
            const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
            for (auto& [state, run_args, keys] : pending) {
                if (state->merged) {
                    programs.at(state->factory_range).update_program_run_args(run_args.base, validation);
                    for (const auto& [device, values] : run_args.per_device) {
                        workload.update_program_run_args_for_device(state->factory_range, device, values);
                    }
                } else {
                    for (const auto& coord_range : state->program_ranges) {
                        const auto* values = pd::values_of(run_args.per_device, coord_range.start_coord());
                        programs.at(coord_range)
                            .update_program_run_args(
                                values != nullptr ? pd::overlay(run_args.base, *values) : run_args.base, validation);
                    }
                }
            }
            for (auto& [state, run_args, keys] : pending) {
                state->prev_keys = std::move(keys);
            }
        }

        static auto create_mesh_workload(
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto* mesh_device = SpecBindingSupport::source_mesh_device(attrs, tensor_args, tensor_return_value);

            auto artifacts = WorkloadSpecFactory::create_mesh_workload_artifacts(
                attrs, tensor_args, tensor_return_value, tensor_coords);
            TT_FATAL(!artifacts.programs.empty(), "create_mesh_workload_artifacts returned no programs");

            const auto mesh_tensors = SpecBindingSupport::collect_mesh_tensors(tensor_args, tensor_return_value);
            // Park the factory's GlobalSemaphores. Moving the vector keeps its elements where they are, so the run
            // args' references to them stay valid.
            const auto parked_semaphores = artifacts.global_semaphores.empty()
                                               ? nullptr
                                               : std::make_shared<const std::vector<tt::tt_metal::GlobalSemaphore>>(
                                                     std::move(artifacts.global_semaphores));
            if constexpr (uses_per_device_run_args()) {
                return create_per_device_mesh_workload(
                    *mesh_device, attrs, tensor_coords, artifacts, mesh_tensors, parked_semaphores);
            } else {
                return create_plain_mesh_workload(
                    *mesh_device, attrs, tensor_coords, tensor_args, artifacts, mesh_tensors, parked_semaphores);
            }
        }

        static cached_mesh_workload_t create_plain_mesh_workload(
            tt::tt_metal::distributed::MeshDevice& mesh_device,
            const operation_attributes_t& attrs,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            MeshWorkloadArtifacts& artifacts,
            const std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>>& mesh_tensors,
            const std::shared_ptr<const std::vector<tt::tt_metal::GlobalSemaphore>>& parked_semaphores) {
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramSpec> program_specs;
            std::unordered_map<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramRunArgs> run_params;
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
            for (auto& per_coord : artifacts.programs) {
                // Metal catches overlapping ranges but never sees an exact duplicate, and it only
                // checks the mesh extent, so equality and tensor_coords containment land here.
                for (const auto& coord : per_coord.range) {
                    const bool covered = std::any_of(
                        tensor_coords.ranges().begin(),
                        tensor_coords.ranges().end(),
                        [&coord](const auto& tensor_range) { return tensor_range.contains(coord); });
                    TT_FATAL(
                        covered,
                        "create_mesh_workload_artifacts returned range {} covering {}, which is outside tensor_coords "
                        "{}: that program would bind tensors the device holds no shard of",
                        per_coord.range,
                        coord,
                        tensor_coords);
                }
                shared_variables_t sv{
                    .bindings = SpecBindingSupport::resolve_bindings(per_coord.run_params.tensor_args, mesh_tensors),
                    .pipe_bindings = SpecBindingSupport::resolve_pipe_bindings(
                        per_coord.run_params.advanced_options.prefetcher_pipe_args, attrs),
                    .global_semaphore_bindings = SpecBindingSupport::resolve_global_semaphore_bindings(
                        per_coord.run_params.global_semaphore_args,
                        attrs,
                        parked_semaphores.get(),
                        /*hit_overrides_run_args=*/has_override_runtime_arguments()),
                    .num_caller_global_semaphores = SpecBindingSupport::num_caller_global_semaphores(attrs),
                    .global_semaphores = parked_semaphores};
                sv.cmd_list_bindings = SpecBindingSupport::template make_cmd_list_bindings<WorkloadSpecFactory>(
                    attrs, tensor_args, per_coord.spec, per_coord.run_params);
                const auto [_, inserted] = program_specs.emplace(per_coord.range, std::move(per_coord.spec));
                TT_FATAL(inserted, "create_mesh_workload_artifacts returned range {} twice", per_coord.range);
                TT_FATAL(
                    per_coord.per_device_run_params.empty(),
                    "create_mesh_workload_artifacts returned per-device run args for range {}, but the factory does "
                    "not declare `uses_per_device_run_args`",
                    per_coord.range);
                shared_variables.emplace(per_coord.range, std::move(sv));
                run_params.emplace(per_coord.range, std::move(per_coord.run_params));
            }

            tt::tt_metal::distributed::MeshWorkload mesh_workload(mesh_device, program_specs);
            for (auto& [range, program] : mesh_workload.get_programs()) {
                program.set_program_run_args(run_params.at(range));
            }
            return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
        }

        static void apply_descriptor(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            if constexpr (fixed_per_device_run_args()) {
                apply_fixed_per_device_run_args(cached_workload, attrs, tensor_args, tensor_return_value);
            } else if constexpr (uses_per_device_run_args()) {
                apply_per_device_run_args(cached_workload, attrs, tensor_args, tensor_return_value);
            } else {
                SpecBindingSupport::refresh_attrs_bound_args(cached_workload, attrs);
                if constexpr (has_override_runtime_arguments()) {
                    const tt::tt_metal::ValidationMode validation = program_args_validation_mode();
                    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                        auto run_args = WorkloadSpecFactory::override_runtime_arguments(
                            attrs, tensor_args, tensor_return_value, coordinate_range);
                        program.update_program_run_args(run_args, validation);
                    }
                } else {
                    SpecBindingSupport::refresh_tensor_args(cached_workload, tensor_args, tensor_return_value);
                }
            }
        }
    };

    // A spec factory's cache hit replays the io slot each TensorArgument was matched to on the miss in
    // its tensor refresh, unless it re-derives its run args with override_runtime_arguments, and in a
    // command-list recording, if it defines map_cmd_list_parameters. When the miss passed one MeshTensor
    // in several slots, that match could have picked any of them (#57248), so ops with a factory that
    // replays it key the cache on which tensor_args slots alias: an entry is then only reused by calls whose
    // aliased slots again hold one tensor, where every pick is right. Outputs need no entry in the
    // signature: the cache is keyed on tensor_args only, a new output aliases nothing, and an output
    // that aliases an input (e.g. a preallocated one) is passed in a tensor_args slot. Whether an op
    // returns one of its inputs outside of tensor_args is part of the op and must not vary per call:
    // a hit would replay the slot the miss matched.
    static constexpr bool keys_on_tensor_aliasing =
        []<typename... Factories>(std::type_identity<std::variant<Factories...>>) {
            return (
                ((ProgramSpecFactoryConcept<Factories> || CustomProgramSpecFactoryConcept<Factories> ||
                  MeshWorkloadSpecFactoryConcept<Factories>) &&
                 (!detail::HasSpecRuntimeArgsOverride<Factories> || detail::MapsCmdListParameters<Factories>)) ||
                ...);
        }(std::type_identity<program_factory_t>{});

    // Empty unless the op keys on tensor aliasing and some tensors alias, so other calls hash as before.
    static std::vector<std::uint32_t> tensor_alias_signature(
        const std::vector<std::reference_wrapper<const ttnn::Tensor>>& tensors) {
        if constexpr (keys_on_tensor_aliasing) {
            return detail::compute_tensor_alias_signature(tensors);
        } else {
            return {};
        }
    }

    // The program cache key: a 64-bit hash plus an exact, collision-free canonical encoding. The
    // cache compares the canonical part on a hash hit so a 64-bit collision is resolved to a
    // correct (rebuild) miss instead of a wrong hit (issue #45821), so it must mirror the SAME
    // inputs the hash combines.
    //
    // The canonical key is prefixed with op_type_name (the DeviceOperation's type name) so distinct
    // ops can never alias on a hash collision.
    // For ops with a custom compute_program_hash we can't infer which fields it keyed on (it may
    // deliberately exclude some, e.g. an RNG seed), so the canonical key holds only the op-identity
    // prefix plus the tensor alias signature, which this adapter rather than the op folds into the
    // hash -- opting that op out of attribute-level collision resolution. The default
    // reflection-hash path is mirrored exactly.
    static tt::tt_metal::program_cache::detail::ProgramCacheKey compute_mesh_workload_cache_key(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        std::string_view op_type_name,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        const auto tensors = mesh_device_operation_utils::collect_tensors(tensor_args);
        const auto coords = detail::extract_tensor_coordinates_impl(tensors, mesh_device);
        const auto alias_signature = tensor_alias_signature(tensors);

        ttsl::hash::hash_t hash = compute_program_hash(attrs, tensor_args);
        std::string canonical{op_type_name};
        std::for_each(
            coords.begin(), coords.end(), [&](const auto& coord) { hash = ttsl::hash::hash_objects(hash, coord); });
        if constexpr (!requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            canonical += ttsl::hash::canonical_key(attrs, tensor_args);
            std::for_each(coords.begin(), coords.end(), [&](const auto& coord) {
                canonical += ttsl::hash::canonical_key(coord);
            });
        }
        if (!alias_signature.empty()) {
            hash = ttsl::hash::hash_objects(hash, alias_signature);
            canonical += ttsl::hash::canonical_key(alias_signature);
        }
        return {hash, std::move(canonical)};
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
