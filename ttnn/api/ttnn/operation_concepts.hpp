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
//  Metal 2.0 ProgramSpec factory concepts
// ============================================================================
//
// A Metal 2.0 op factory produces a ProgramArtifacts: the immutable ProgramSpec (the "blueprint" —
// kernels, DFBs, work-units, argument schemas) plus the ProgramRunArgs (the per-execution values).
// There are exactly TWO concepts, distinguished by ONE thing — the cache key:
//
//   - ProgramSpecFactoryConcept         — the cache key is the generated ProgramSpec.
//   - AdvancedProgramSpecFactoryConcept — the cache key is a small hashable ImmutableInfo that the
//                                         factory extracts up front (extract_immutable_info), which is
//                                         also the SOLE input to create_program_artifacts. This lets the
//                                         framework skip the spec rebuild on a cache hit, and structurally
//                                         prevents a mutable value (e.g. an RNG seed) from leaking into
//                                         the key or the spec — it isn't even visible to the builder.
//
// Within EITHER concept, the enqueue-invariant fast path is an OPTIONAL, INCREMENTAL refinement — NOT a
// separate concept (Audrey's "Option 2 / Option 2++"; "Option 3 / Option 3++"):
//
//   - Degenerate (no create_per_enqueue_args): create_program_artifacts returns the spec + ALL run-args.
//     Cache hit refreshes tensor bindings (UpdateTensorArgs). The least to write — migrate here first.
//   - ++ (add create_per_enqueue_args): create_program_artifacts's run_args carry only the
//     ENQUEUE-INVARIANT args (declared invariant in the spec via
//     KernelAdvancedOptions::enqueue_invariant_runtime_args / _common_runtime_args,
//     TensorParameterAdvancedOptions::enqueue_invariant); create_per_enqueue_args carries the rest.
//     Cache miss merges the two (MergeProgramRunArgs) + SetProgramRunArgs; cache hit re-applies ONLY the
//     per-enqueue set (UpdateProgramRunArgs). The metal runtime validates that every omitted arg was
//     declared invariant — a per-enqueue value the factory forgets to refresh is a hard error, not a
//     silently-stale wrong answer.
//
// You only need to move ENOUGH args into the invariant bundle to clear your performance target — moving
// one more arg from create_per_enqueue_args into create_program_artifacts's run_args (and declaring it
// invariant) is a one-line, behavior-preserving optimization. Pick the rung by MEASURING dispatch time
// (see TTNN_OP_MIGRATION_RECIPE.md).
//
// NOTE: Each TensorArgument.tensor in ProgramRunArgs MUST reference a MeshTensor reachable from the
// factory's tensor_args / tensor_return_value — the adapter matches by pointer identity.

// --- method-surface building blocks ---
template <typename T>
concept HasCreateProgramArtifacts = requires { &T::create_program_artifacts; };
template <typename T>
concept HasImmutableInfoExtraction = requires { &T::extract_immutable_info; };
// The optional "++" refinement: present => the factory splits invariant (create_program_artifacts
// run-args) from per-enqueue (create_per_enqueue_args), and the hit path re-applies only the latter.
template <typename T>
concept HasPerEnqueueArgs = requires { &T::create_per_enqueue_args; };
// Shared exclusion: a Metal 2.0 spec factory is none of the legacy factory shapes.
template <typename T>
concept NotALegacyFactory =
    !ProgramFactoryConcept<T> && !MeshWorkloadFactoryConcept<T> && !ProgramDescriptorFactoryConcept<T>;

// Spec-keyed factory (Option 2 / 2++). create_program_artifacts(attrs, tensor_args, tensor_return_value)
// -> ProgramArtifacts. Optionally also create_per_enqueue_args for the enqueue-invariant fast path.
template <typename T>
concept ProgramSpecFactoryConcept =
    HasCreateProgramArtifacts<T> && !HasImmutableInfoExtraction<T> && NotALegacyFactory<T>;

// ImmutableInfo-keyed factory (Option 3 / 3++). extract_immutable_info -> ImmutableInfo (cache key);
// create_program_artifacts(ImmutableInfo) -> ProgramArtifacts. Optionally create_per_enqueue_args too.
// ("Advanced" = it does the extra immutable-info extraction step that buys the cheaper cache key.)
template <typename T>
concept AdvancedProgramSpecFactoryConcept =
    HasCreateProgramArtifacts<T> && HasImmutableInfoExtraction<T> && NotALegacyFactory<T>;

// Umbrella: either Metal 2.0 spec factory shape (exactly one is satisfied by construction).
template <typename T>
concept Metal2SpecFactoryConcept = ProgramSpecFactoryConcept<T> || AdvancedProgramSpecFactoryConcept<T>;

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
