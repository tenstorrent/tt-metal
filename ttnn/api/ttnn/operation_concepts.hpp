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
//  Metal 2.0 ProgramSpec factory concepts — a progressive ladder
// ============================================================================
//
// A Metal 2.0 op factory's job is to produce TWO things: the immutable ProgramSpec (the "blueprint":
// kernels, DFBs, work-units, argument schemas) and the ProgramRunArgs (the per-execution values). The
// four concepts below are points on a ladder a dev climbs ONLY as far as performance forces them:
//
//   1. ProgramSpecFactoryConcept — THE DEFAULT. One method, create_program_spec, returns the spec and
//      ALL its run-args bundled. Migrate here first; it is the least to write.
//
//   2. AdvancedProgramSpecFactoryConcept — if the cache-HIT cost hurts. Keep the same cache key, but
//      split the run-args into STATIC (create_static_args — fixed for a cache entry, set once on miss
//      and retained) and DYNAMIC (create_dynamic_args — the per-dispatch values, e.g. an RNG seed or
//      tensor addresses). The framework re-applies ONLY the dynamic set on a hit (UpdateProgramRunArgs),
//      not the whole arg set.
//
//   3. ImmutableProgramSpecFactoryConcept / AdvancedImmutableProgramSpecFactoryConcept — if it STILL
//      hurts. Add extract_immutable_info: a small, hashable projection of (attributes, tensor_args)
//      that becomes the cache key AND the sole input to create_program_spec / create_static_args. This
//      lets the framework skip the spec rebuild on a hit, and structurally prevents a mutable value
//      (the seed) from leaking into the spec — it isn't even visible to create_program_spec. Available
//      both bare (combined run-args) and with the static/dynamic split.
//
// The concepts are detected by method surface and are mutually exclusive by construction:
//   - extract_immutable_info present        => Immutable-keyed (3/4)
//   - create_static_args + create_dynamic_args present => Advanced / split (2/4)
//
// Run-args contract for the split (2 and 4): static run-args are declared enqueue-invariant in the spec
// (KernelAdvancedOptions::enqueue_invariant_runtime_args / _common_runtime_args,
// TensorParameterAdvancedOptions::enqueue_invariant). The framework merges static + dynamic
// (MergeProgramRunArgs) for the cache-miss SetProgramRunArgs, and re-applies dynamic alone
// (UpdateProgramRunArgs) on a hit. The metal runtime validates that every arg omitted on the hit path
// was declared invariant — a dynamic value the factory forgets to refresh is a hard error, not a
// silently-stale wrong answer.
//
// NOTE: Each TensorArgument.tensor in ProgramRunArgs MUST reference a MeshTensor reachable from the
// factory's tensor_args / tensor_return_value — the adapter matches by pointer identity.

// --- method-surface building blocks (not used directly; compose the four leaf concepts) ---
template <typename T>
concept HasCreateProgramSpec = requires { &T::create_program_spec; };
template <typename T>
concept HasImmutableInfoExtraction = requires { &T::extract_immutable_info; };
template <typename T>
concept HasStaticDynamicArgSplit = requires {
    &T::create_static_args;
    &T::create_dynamic_args;
};
// Shared exclusion: a Metal 2.0 spec factory is none of the legacy factory shapes.
template <typename T>
concept NotALegacyFactory =
    !ProgramFactoryConcept<T> && !MeshWorkloadFactoryConcept<T> && !ProgramDescriptorFactoryConcept<T>;

// 1. Basic, spec-keyed (the default). create_program_spec -> ProgramArtifacts{spec, run_args}.
template <typename T>
concept ProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && !HasImmutableInfoExtraction<T> && !HasStaticDynamicArgSplit<T> && NotALegacyFactory<T>;

// 2. Advanced (static/dynamic split), spec-keyed. create_program_spec -> ProgramSpec;
//    create_static_args -> ProgramRunArgs; create_dynamic_args -> ProgramRunArgs.
template <typename T>
concept AdvancedProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && HasStaticDynamicArgSplit<T> && !HasImmutableInfoExtraction<T> && NotALegacyFactory<T>;

// 3. Basic, immutable-info-keyed. extract_immutable_info -> ImmutableInfo (cache key);
//    create_program_spec(ImmutableInfo) -> ProgramArtifacts{spec, run_args}.
template <typename T>
concept ImmutableProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && HasImmutableInfoExtraction<T> && !HasStaticDynamicArgSplit<T> && NotALegacyFactory<T>;

// 4. Advanced (static/dynamic split), immutable-info-keyed.
//    extract_immutable_info -> ImmutableInfo; create_program_spec(ImmutableInfo) -> ProgramSpec;
//    create_static_args(ImmutableInfo) -> ProgramRunArgs; create_dynamic_args(...) -> ProgramRunArgs.
template <typename T>
concept AdvancedImmutableProgramSpecFactoryConcept =
    HasCreateProgramSpec<T> && HasImmutableInfoExtraction<T> && HasStaticDynamicArgSplit<T> && NotALegacyFactory<T>;

// Umbrella: any of the four Metal 2.0 spec factory shapes (exactly one is satisfied by construction).
template <typename T>
concept Metal2SpecFactoryConcept =
    ProgramSpecFactoryConcept<T> || AdvancedProgramSpecFactoryConcept<T> || ImmutableProgramSpecFactoryConcept<T> ||
    AdvancedImmutableProgramSpecFactoryConcept<T>;

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
// or one of the Metal 2.0 spec factory shapes (Metal2SpecFactoryConcept — itself exactly one of four).
namespace detail {
template <typename Variant, std::size_t... Is>
consteval bool all_factories_valid(std::index_sequence<Is...>) {
    return (
        ((ProgramFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          MeshWorkloadFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          ProgramDescriptorFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          Metal2SpecFactoryConcept<std::variant_alternative_t<Is, Variant>>) == 1) &&
        ...);
}
}  // namespace detail

template <typename Variant>
concept AllFactoriesValid =
    detail::all_factories_valid<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>{});

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
