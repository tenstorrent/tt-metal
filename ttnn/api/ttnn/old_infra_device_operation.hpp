// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operation_concepts.hpp"

namespace tt::tt_metal::operation {

template <typename OutputTensors>
struct OldInfraDeviceOperation {
    using operation_attributes_t = operation::DeviceOperation<OutputTensors>;

    struct tensor_args_t {
        const operation::Tensors input_tensors;
        const operation::OptionalConstTensors optional_input_tensors;
        const operation::OptionalTensors optional_output_tensors;
    };

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;

    using tensor_return_value_t = OutputTensors;

    struct ProgramFactory {
        struct shared_variables_t {
            std::optional<operation::OverrideRuntimeArgumentsCallback<OutputTensors>>
                override_runtime_arguments_callback;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct MeshWorkloadFactory {
        struct shared_variables_t {
            std::optional<operation::OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>> workload_callback;
            std::unordered_map<ttnn::MeshCoordinateRange, operation::OverrideRuntimeArgumentsCallback<OutputTensors>>
                per_program_callbacks;
        };
        using cached_mesh_workload_t = ttnn::device_operation::CachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // When the wrapped operation has `create_mesh_workload` method, `OldInfraDeviceOperation` adapter will select
    // `MeshWorkloadFactory` as the program factory.
    static_assert(
        !ttnn::device_operation::ProgramFactoryConcept<MeshWorkloadFactory> &&
        ttnn::device_operation::MeshWorkloadFactoryConcept<MeshWorkloadFactory>);
    static_assert(
        ttnn::device_operation::ProgramFactoryConcept<ProgramFactory> &&
        !ttnn::device_operation::MeshWorkloadFactoryConcept<ProgramFactory>);

    using program_factory_t = std::variant<ProgramFactory, MeshWorkloadFactory>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Compute the output specs based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static auto create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static std::string get_type_name(const operation_attributes_t& attributes);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        operation_attributes_t&& operation_attributes,
        const operation::Tensors& input_tensors,
        const operation::OptionalConstTensors& optional_input_tensors,
        const operation::OptionalTensors& optional_output_tensors);
};

}  // namespace tt::tt_metal::operation

namespace ttnn::prim {

constexpr auto old_infra_device_operation = ttnn::register_operation<
    "ttnn::prim::old_infra_device_operation",
    tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::Tensors>>();
constexpr auto old_infra_device_operation_with_optional_output_tensors = ttnn::register_operation<
    "ttnn::prim::old_infra_device_operation_with_optional_output_tensors",
    tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::OptionalTensors>>();

}  // namespace ttnn::prim
