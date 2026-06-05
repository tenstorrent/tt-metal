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

// Metal 2.0 factory concept:
// Factories that return EITHER:
//  - ProgramArtifacts (a ProgramSpec + ProgramRunArgs) from create_program_artifacts, or
//  - A separate ProgramSpec from create_program_spec, ProgramRunArgs from create_program_run_args,
//    and a minimal struct of extracted data from the input args, chained into create_program_spec.
//
//-----------------------------------------------------------------------------------
// "Option 1" (via create_program_artifacts):
//
// The framework adapter stamps a Program from the spec onto each mesh coordinate range on
// cache miss, and patches ONLY the tensor arguments on cache hit.
//
// The ProgramCache hash key is automatically computed by the TTNN infrastructure based on
// the op's type and arguments (taking into account any TensorParameter relaxations).
// The factory author writes NO fast-path-specific code.
//
// The factory-provided create_program_artifacts is only called on cache miss.
//
// NOTE: Each TensorArgument.tensor in ProgramRunArgs MUST reference a MeshTensor reachable
//   from the factory's `tensor_args` / `tensor_return_value` parameters — the adapter
//   matches by pointer identity.
//
// CAUTION: Do not pass raw tensor addresses through runtime arguments!
//   If a raw address is passed as a user-defined runtime argument, only the first dispatch
//   succeeds; every subsequent program-cache hit then silently uses a stale address.
//   A Metal 2.0 program factory should NEVER take the address of any tensor or buffer.
//   Instead, declare a TensorParameter on the ProgramSpec and pass MeshTensor arguments
//   directly through the ProgramRunArgs. The Metal 2.0 runtime infrastructure implicitly
//   captures the pointer (and other tensor properties), which the kernel retrieves via
//   TensorAccessor.
//
//-----------------------------------------------------------------------------------
// "Option 2" (ALSO via create_program_artifacts):
//
// The framework adapter stamps a Program from the spec onto each mesh coordinate range on
// cache miss, and applies the FULL ProgramRunArgs on cache hit.
//
// The ProgramCache hash key is automatically computed by the TTNN infrastructure by hashing
// the immutable ProgramSpec struct. The factory author writes NO fast-path-specific code.
//
// The factory-provided create_program_artifacts is always called (both cache miss and hit).
//
// NOTE: Compared to Option 1, Option 2:
//   - Exposes all of Metalium's ProgramRunArgs (not limited to tensor arguments) for maximum
//     fast path flexibility.
//   - Is always safe. The cache key and mutable Program update are guaranteed to be correct
//     by construction.
//   - Has greater host overhead on cache hit; create_program_artifacts is always invoked.
//
//-----------------------------------------------------------------------------------
// "Option 3" (AdvancedProgramSpecFactoryConcept):
//
// The framework adapter:
//  - Calls extract_immutable_info and computes the hash key from the returned struct.
//  - On cache miss, create_program_spec is called using the returned immutable info struct.
//    The resulting Program is cached; create_program_run_args is then called to generate
//    the ProgramRunArgs to run the Program.
//  - On cache hit, ONLY create_program_run_args is called.
//
// For the most efficient fast path, minimize the logic in extract_immutable_info.
//
// NOTE: Compared to Options 1 and 2:
//  - Option 3 is the most complex factory to implement.
//  - Like Option 2, it is always safe (guaranteed to be correct by construction).
//  - It provides the greatest fast-path flexibility of all options. The factory author
//    can leverage the full Program mutability, and can also minimize fast-path host
//    overhead.
//
template <typename T>
concept WorkloadArtifactConcept = requires { &T::create_workload_artifacts; };

template <typename T>
concept AdvancedProgramSpecFactoryConcept = requires {
    &T::create_program_spec;
    &T::create_program_run_args;
    &T::extract_immutable_info;
};

// TODO: Define how the "switch" to select between Option 1 and 2 works.
//
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
