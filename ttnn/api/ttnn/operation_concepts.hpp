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

// Metal 2.0 factory concept.
//
// ============================================================================
// What this concept is protecting against
// ============================================================================
//
// The legacy TTNN op infrastructure had TWO independent bug generators around the
// ProgramCache, and any successor design has to close both:
//
//  (A) Immutable-side bug generator — over-permissive hash.
//      An op author who supplies a custom hash function may accidentally OMIT an
//      attribute that actually affects the immutable Program. The first dispatch
//      builds and caches the correct Program. Subsequent dispatches that vary
//      ONLY the omitted attribute generate the same hash, hit the cache, and
//      silently resurrect the WRONG Program — the device performs the wrong
//      computation. The author's only feedback is a wrong answer.
//
//  (B) Mutable-side bug generator — stale runtime args.
//      The fast cache-hit path updates some subset of the Program's runtime
//      state. If the author's update logic misses a field that varies between
//      dispatches sharing a cache entry, every hit silently uses the stale
//      value from the previous dispatch. Same failure mode: wrong answer.
//
// The three factory shapes below sit at different points on the
// simplicity / cache-hit-perf / safety-by-construction trade-off triangle,
// but ALL of them close both bug generators by construction.
//
// ----------------------------------------------------------------------------
// "Option 1" (via create_program_artifacts, with `using fast_cache_hit_path = std::true_type;`):
//
// On cache miss:    factory called, full ProgramRunArgs set on every Program.
// On cache hit:     ONLY tensor args refreshed via UpdateTensorArgs; factory NOT called.
//
// Closes (A) by FORBIDDING a custom compute_program_hash on the DeviceOperation.
// The framework's automatic hash of (op type + attrs + tensor args + mesh coords)
// is the only sanctioned cache key. The author can't omit anything because they
// can't write a hash function at all. The trifecta adapter rejects the
// combination at compile time.
//
// Closes (B) by FORBIDDING any non-tensor mutation in ProgramRunArgs that varies
// across dispatches sharing a cache entry. Concretely, the adapter rejects DFB
// size overrides and common (broadcast) runtime args; per-node RTAs are still
// legal because they're the standard mechanism for per-node work distribution
// AND must be deterministic from the cache key (the hash hashes the args those
// values are derived from).
//
// The one residual unsafety is the "raw-address-smuggling" anti-pattern:
// if a factory bypasses TensorParameter and stuffs a raw tensor base pointer
// into a per-node RTA, the address is set ONCE at cache-miss and persists on
// every hit — even when the underlying tensor's allocation changes. This is a
// hand-crafted backdoor, not something a typed factory accidentally does.
// See the anti-pattern block at the concept definition below.
//
// ----------------------------------------------------------------------------
// "Option 2" (default, via create_program_artifacts, no marker):
//
// On cache miss:    factory called, full ProgramRunArgs set on every Program.
// On cache hit:     factory called AGAIN, full ProgramRunArgs re-applied via
//                   SetProgramRunArgs. The factory pays its full cost every dispatch.
//
// Closes (A) by FORBIDDING custom compute_program_hash (same as Option 1) AND by
// keying the cache off the immutable ProgramSpec itself — not the op args. The
// spec captures everything pertinent to the immutable Program by definition, so
// any two dispatches whose factories produce equal specs map to the same cache
// entry (cache reuse across arg variations the spec doesn't distinguish, e.g.
// TensorParameter relaxations that route shape variation through CRTAs). The
// factory has to run BEFORE the cache lookup to produce the spec — that's the
// structural cost Option 2 pays for safety-by-construction plus this cache-reuse
// breadth.
//
// Closes (B) by re-running the factory on every cache hit and re-applying the
// FULL ProgramRunArgs. Every mutable field is built fresh from the current
// dispatch's inputs, so nothing can carry stale state. Any ProgramRunArgs
// channel is legal (RTAs, CRTAs, DFB size overrides — all of it).
//
// Greater host overhead than Option 1, in exchange for "I don't need to think
// about cache-hit semantics" as the factory author's contract.
//
// ----------------------------------------------------------------------------
// "Option 3" (AdvancedProgramSpecFactoryConcept) — not yet implemented:
//
// The factory is split into three pieces:
//   1. extract_immutable_info(op_args) → ImmutableInfo
//   2. create_program_spec(const ImmutableInfo&) → ProgramSpec  [cache-miss only]
//   3. create_program_run_args(op_args) → ProgramRunArgs        [every dispatch]
//
// Closes (A) by hashing ImmutableInfo as the cache key AND by feeding ONLY the
// ImmutableInfo (not the raw op_args) into create_program_spec. The factory
// physically can't depend on a field it didn't declare in ImmutableInfo,
// because it doesn't have access to anything else. This is a structural
// guarantee, not a discipline.
//
// Closes (B) by calling create_program_run_args fresh on every dispatch.
//
// Combines Option 1's fast cache-hit path (the spec isn't rebuilt) with Option
// 2's safety (the mutable side is fresh every dispatch), at the cost of
// authoring complexity: the split forces the factory to be explicit about which
// inputs affect the spec.
//
template <typename T>
concept WorkloadArtifactConcept = requires { &T::create_workload_artifacts; };

template <typename T>
concept AdvancedProgramSpecFactoryConcept = requires {
    &T::create_program_spec;
    &T::create_program_run_args;
    &T::extract_immutable_info;
};

// Option 1 vs Option 2 selector.
//
// Both options expose the same factory signature (create_program_artifacts) and
// both close both bug generators (see comment block above). The difference is
// how restrictive the factory contract is:
//
//   - Option 2 (default) — no constraints on the ProgramRunArgs returned by the
//     factory. Anything mutable per dispatch is fine: per-node RTAs, CRTAs, DFB
//     size overrides. Cost: factory re-runs on every cache hit.
//
//   - Option 1 (opt-in) — restricts the factory to per-node RTAs and tensor
//     args only (no CRTAs, no DFB size overrides). The cache-hit path skips the
//     factory and refreshes only tensor args. Cheap, but the author has to live
//     within the restrictions.
//
// Opt-in to Option 1 via a type-level marker:
//
//   struct MyFactory {
//       using fast_cache_hit_path = std::true_type;  // opt into Option 1
//       static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...);
//   };
//
// The default-to-Option-2 choice matches the path of least surprise: an op
// author who writes the factory without knowing about the distinction lands
// on the less-restrictive contract and their code compiles. Opting into
// Option 1 is the explicit "I've checked that my factory only varies tensor
// args between dispatches, please give me the fast path."
//
// Note that because both options expose the same factory signature, the
// marker is a cheap perf experiment: an op author can prototype with the
// default (Option 2) and flip `fast_cache_hit_path` on once the factory is
// otherwise stable to A/B-test whether the fast cache-hit path actually
// matters for this op's workload — the factory body stays the same, only
// the marker (and the framework's dispatch path) changes.
template <typename T>
concept HasFastCacheHitPathOptIn = requires {
    typename T::fast_cache_hit_path;
    requires T::fast_cache_hit_path::value;
};

// "Exactly one of the three forms is satisfied" is enforced by sum-equals-1
// (the same pattern Diego uses for outer-factory disambiguation in
// AllFactoriesValid below). A factory that accidentally exposes two shapes
// is rejected at the concept level rather than silently picked apart by
// adapter precedent order.
template <typename T>
concept ProgramSpecFactoryConcept =
    ((requires { &T::create_program_artifacts; } + AdvancedProgramSpecFactoryConcept<T> + WorkloadArtifactConcept<T>) ==
     1) &&
    !ProgramFactoryConcept<T> && !MeshWorkloadFactoryConcept<T> && !ProgramDescriptorFactoryConcept<T>;

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
// or ProgramSpecFactoryConcept.
namespace detail {
template <typename Variant, std::size_t... Is>
consteval bool all_factories_valid(std::index_sequence<Is...>) {
    return (
        ((ProgramFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          MeshWorkloadFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          ProgramDescriptorFactoryConcept<std::variant_alternative_t<Is, Variant>> +
          ProgramSpecFactoryConcept<std::variant_alternative_t<Is, Variant>>) == 1) &&
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
