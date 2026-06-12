// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <variant>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/program_cache.hpp>

#include <cstdint>

#include "ttnn/distributed/types.hpp"

namespace ttnn::device_operation {

template <typename T>
concept ProgramFactoryConcept = requires {
    typename T::cached_program_t;

    [](const auto& operation_attributes, const auto& tensor_args, auto& tensor_return_value) {
        auto cached_program = T::create(operation_attributes, tensor_args, tensor_return_value);

        T::override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);
    };
};

template <typename T>
concept HasMeshWorkloadType = requires { typename T::cached_mesh_workload_t; };

template <typename T>
concept HasCreateMeshWorkload = requires {
    // Path A: Factory provides create_mesh_workload directly
    &T::create_mesh_workload;
    &T::override_runtime_arguments;
};

template <typename T>
concept HasCreateAt = requires {
    // Path B: Factory provides create_at (per-coordinate)
    &T::create_at;
    &T::override_runtime_arguments;
};

template <typename T>
concept MeshWorkloadFactoryConcept = HasMeshWorkloadType<T> && (HasCreateMeshWorkload<T> || HasCreateAt<T>);

// Mesh-workload descriptor factory: builds the entire workload in one call
// via `create_workload_descriptor`, returning a tt::tt_metal::WorkloadDescriptor
// that pairs declarative per-coord ProgramDescriptors with workload-scoped
// resources (semaphores, buffers).  Replaces the deprecated prepare_resources
// hook.
//
// This concept is a shape check — it confirms `create_workload_descriptor` is a
// member.  The strict signature check (4 args, last is `MeshCoordinateRangeSet`,
// returns `tt::tt_metal::WorkloadDescriptor`) is enforced by
// `has_workload_descriptor` in the adapter, which has access to the device
// operation's typedefs.  A factory that satisfies this concept but provides a
// mismatched `create_workload_descriptor` triggers a clear `static_assert` failure
// in the adapter rather than a deep template error.
template <typename T>
concept WorkloadDescriptorConcept = requires { &T::create_workload_descriptor; };

template <typename T>
concept ProgramDescriptorFactoryConcept = (requires { &T::create_descriptor; } || WorkloadDescriptorConcept<T>) &&
                                          !ProgramFactoryConcept<T> && !MeshWorkloadFactoryConcept<T>;

// ============================================================================
//  MetalV2 ProgramSpec factory concepts
// ============================================================================
//
// A MetalV2 op factory is built from three methods, each with a single job:
//
//   - create_program_spec       -> ProgramSpec     : the immutable blueprint (kernels, DFBs, work-units,
//                                                     argument schemas). Built once, on a cache miss.
//   - create_invariant_run_args -> ProgramRunArgs  : the enqueue-invariant run-args (work splits, shape
//                                                     scalars). Set once on a miss and retained across hits.
//   - create_per_enqueue_args   -> ProgramRunArgs  : the per-enqueue run-args (tensor addresses, seeds).
//                                                     Rebuilt on EVERY dispatch and re-applied via
//                                                     UpdateProgramRunArgs.
//
// On a miss the framework merges the invariant + per-enqueue sets for the initial SetProgramRunArgs. On a
// hit it rebuilds neither the spec nor the invariant args — it re-runs only create_per_enqueue_args and
// re-applies it. Splitting the run-args across two methods is what forces the author to decide, per arg,
// what is enqueue-invariant vs per-enqueue; the metal runtime then validates that every arg omitted from
// the per-enqueue set was declared enqueue_invariant in the spec — a forgotten per-enqueue value is a hard
// error, not silently-stale data.
//
// There are exactly TWO concepts, distinguished by ONE thing — the cache key:
//
//   - ProgramSpecFactoryConcept         — the cache key is the framework default: a reflection hash of
//                                         (op type + attributes + tensor args).
//   - AdvancedProgramSpecFactoryConcept — the cache key is a small hashable ImmutableInfo the factory
//                                         extracts up front (extract_immutable_info), which is also the
//                                         SOLE input to create_program_spec. Structurally keeps a mutable
//                                         value (e.g. an RNG seed) out of both the key and the spec — it
//                                         isn't even visible to the builder.

// --- method-surface building blocks ---
template <typename T>
concept HasCreateProgramSpec = requires { &T::create_program_spec; };
template <typename T>
concept HasCreateInvariantRunArgs = requires { &T::create_invariant_run_args; };
template <typename T>
concept HasCreatePerEnqueueArgs = requires { &T::create_per_enqueue_args; };
template <typename T>
concept HasImmutableInfoExtraction = requires { &T::extract_immutable_info; };
// Shared exclusion: a MetalV2 spec factory is none of the legacy factory shapes.
template <typename T>
concept NotALegacyFactory =
    !ProgramFactoryConcept<T> && !MeshWorkloadFactoryConcept<T> && !ProgramDescriptorFactoryConcept<T>;

// All three build methods are mandatory; the cache key is the default reflection hash
// (op type + attributes + tensor args).
template <typename T>
concept ProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && HasCreateInvariantRunArgs<T> && HasCreatePerEnqueueArgs<T> &&
    !HasImmutableInfoExtraction<T> && NotALegacyFactory<T>;

// ImmutableInfo-keyed factory: extract_immutable_info -> ImmutableInfo is BOTH the cache key and the sole
// input to create_program_spec, so a mutable value (e.g. an RNG seed) cannot leak into the key or the spec.
template <typename T>
concept AdvancedProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && HasCreateInvariantRunArgs<T> && HasCreatePerEnqueueArgs<T> &&
    HasImmutableInfoExtraction<T> && NotALegacyFactory<T>;

// Umbrella: either MetalV2 spec factory shape (exactly one is satisfied by construction).
template <typename T>
concept MetalV2SpecFactoryConcept = ProgramSpecFactoryConcept<T> || AdvancedProgramSpecFactoryConcept<T>;

// Detect operations that put create_descriptor directly on the operation struct
// (no program_factory_t wrapper needed for single-descriptor operations).
template <typename T>
concept HasDirectDescriptor = requires { &T::create_descriptor; } && !requires { typename T::program_factory_t; };

template <typename device_operation_t>
concept HasComputeOutputSpecs = requires(
    device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    {
        op.compute_output_specs(operation_attributes, tensor_args)
    } -> std::same_as<typename device_operation_t::spec_return_value_t>;
};

// Detect if operation provides custom cache-hit validation.
// If not provided, the framework defaults to calling validate_on_program_cache_miss.
template <typename device_operation_t>
concept HasValidateOnProgramCacheHit = requires(
    const typename device_operation_t::operation_attributes_t& attrs,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    device_operation_t::validate_on_program_cache_hit(attrs, tensor_args);
};

// Detect if operation provides a custom select_program_factory.
// If not provided and program_factory_t is a single-type variant, the framework returns it automatically.
template <typename device_operation_t>
concept HasSelectProgramFactory = requires(
    const typename device_operation_t::operation_attributes_t& attrs,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    {
        device_operation_t::select_program_factory(attrs, tensor_args)
    } -> std::same_as<typename device_operation_t::program_factory_t>;
};

// Validate that all variant alternatives in a program_factory_t satisfy exactly one of
// ProgramFactoryConcept, MeshWorkloadFactoryConcept, ProgramDescriptorFactoryConcept,
// or one of the MetalV2 spec factory shapes (MetalV2SpecFactoryConcept — itself exactly one of four).
namespace detail {
template <typename Variant, std::size_t... Is>
consteval bool all_factories_valid(std::index_sequence<Is...>) {
    return (
        ((ProgramFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          MeshWorkloadFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          ProgramDescriptorFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          MetalV2SpecFactoryConcept<std::variant_alternative_t<Is, Variant>>) == 1) &&
        ...);
}

// An op migrates to MetalV2 all-or-nothing: a program_factory_t may not mix MetalV2 spec factories
// with legacy factory shapes. Either every variant is a MetalV2 spec factory, or none is.
template <typename Variant, std::size_t... Is>
consteval bool metal2_factories_not_mixed(std::index_sequence<Is...>) {
    constexpr bool any = (MetalV2SpecFactoryConcept<std::variant_alternative_t<Is, Variant>> || ...);
    constexpr bool all = (MetalV2SpecFactoryConcept<std::variant_alternative_t<Is, Variant>> && ...);
    return all || !any;
}
}  // namespace detail

template <typename Variant>
concept AllFactoriesValid =
    detail::all_factories_valid<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>{}) &&
    detail::metal2_factories_not_mixed<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>{});

template <typename device_operation_t>
concept HasProgramFactoryType = requires { typename device_operation_t::program_factory_t; };

template <typename device_operation_t>
concept DeviceOperationConcept =
    requires {
        [](const typename device_operation_t::operation_attributes_t& operation_attributes,
           const typename device_operation_t::tensor_args_t& tensor_args) {
            device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

            using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
            static_assert(std::same_as<
                          decltype(device_operation_t::create_output_tensors(operation_attributes, tensor_args)),
                          tensor_return_value_t>);
        };
    } && HasComputeOutputSpecs<device_operation_t> &&
    (HasDirectDescriptor<device_operation_t> ||
     (HasProgramFactoryType<device_operation_t> && AllFactoriesValid<typename device_operation_t::program_factory_t>));

template <typename device_operation_t>
concept DeviceOperationWithCustomProgramCacheConcept =
    DeviceOperationConcept<device_operation_t> &&
    requires(
        const typename device_operation_t::operation_attributes_t& operation_attributes,
        const typename device_operation_t::tensor_args_t& tensor_args) {
        {
            device_operation_t::compute_program_hash(operation_attributes, tensor_args)
        } -> std::convertible_to<std::uint64_t>;
    };

template <typename device_operation_t>
concept HasSkipLaunch = requires(
    device_operation_t op,
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args,
    const typename device_operation_t::tensor_return_value_t& tensor_return_value) {
    {
        device_operation_t::skip_launch(operation_attributes, tensor_args, tensor_return_value)
    } -> std::convertible_to<bool>;
};

}  // namespace ttnn::device_operation
