// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
// ProgramFactoryConcept or MeshWorkloadFactoryConcept.
namespace detail {
template <typename Variant, std::size_t... Is>
consteval bool all_factories_valid(std::index_sequence<Is...>) {
    return (
        (ProgramFactoryConcept<std::variant_alternative_t<Is, Variant>> !=
         MeshWorkloadFactoryConcept<std::variant_alternative_t<Is, Variant>>) &&
        ...);
}
}  // namespace detail

template <typename Variant>
concept AllFactoriesValid =
    detail::all_factories_valid<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>{});

template <typename device_operation_t>
concept DeviceOperationConcept = requires {
    typename device_operation_t::program_factory_t;

    [](const typename device_operation_t::operation_attributes_t& operation_attributes,
       const typename device_operation_t::tensor_args_t& tensor_args) {
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
        static_assert(std::same_as<
                      decltype(device_operation_t::create_output_tensors(operation_attributes, tensor_args)),
                      tensor_return_value_t>);
    };
} && HasComputeOutputSpecs<device_operation_t> && AllFactoriesValid<typename device_operation_t::program_factory_t>;

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
