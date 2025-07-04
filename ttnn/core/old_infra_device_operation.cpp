// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/old_infra_device_operation.hpp"

#include "ttnn/operation.hpp"

namespace tt::tt_metal::operation {

template <typename OutputTensors>
typename OldInfraDeviceOperation<OutputTensors>::ProgramFactory::cached_program_t
OldInfraDeviceOperation<OutputTensors>::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto program_with_callbacks = operation_attributes.create_program(
        tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);
    return cached_program_t{
        std::move(program_with_callbacks.program),
        shared_variables_t{program_with_callbacks.override_runtime_arguments_callback}};
}

template <typename OutputTensors>
void OldInfraDeviceOperation<OutputTensors>::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& override_runtime_arguments_callback = cached_program.shared_variables.override_runtime_arguments_callback;
    auto& program = cached_program.program;

    if (override_runtime_arguments_callback.has_value()) {
        operation_attributes.override_runtime_arguments(
            override_runtime_arguments_callback.value(),
            program,
            tensor_args.input_tensors,
            tensor_args.optional_input_tensors,
            tensor_return_value);
    }
}

template <typename OutputTensors>
typename OldInfraDeviceOperation<OutputTensors>::MeshWorkloadFactory::cached_mesh_workload_t
OldInfraDeviceOperation<OutputTensors>::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto workload_with_callbacks = operation_attributes.create_mesh_workload(
        tensor_coords, tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);

    TT_FATAL(
        !workload_with_callbacks.workload_callback.has_value() || workload_with_callbacks.per_program_callbacks.empty(),
        "At most one of 'workload_callback' or 'per_program_callbacks' can be specified.");

    return cached_mesh_workload_t{
        std::move(workload_with_callbacks.workload),
        shared_variables_t{
            std::move(workload_with_callbacks.workload_callback),
            std::move(workload_with_callbacks.per_program_callbacks)}};
}

template <typename OutputTensors>
void OldInfraDeviceOperation<OutputTensors>::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    if (cached_workload.shared_variables.workload_callback.has_value()) {
        operation_attributes.override_runtime_arguments(
            *cached_workload.shared_variables.workload_callback,
            cached_workload.workload,
            tensor_args.input_tensors,
            tensor_args.optional_input_tensors,
            tensor_return_value);
    }

    for (auto& [range, callback] : cached_workload.shared_variables.per_program_callbacks) {
        auto& program = cached_workload.workload.get_programs().at(range);
        operation_attributes.override_runtime_arguments(
            callback, program, tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);
    }
}

template <typename OutputTensors>
typename OldInfraDeviceOperation<OutputTensors>::program_factory_t
OldInfraDeviceOperation<OutputTensors>::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return attributes.has_create_workload_method() ? program_factory_t{MeshWorkloadFactory{}}
                                                   : program_factory_t{ProgramFactory{}};
}

template <typename OutputTensors>
void OldInfraDeviceOperation<OutputTensors>::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    attributes.validate(
        tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_args.optional_output_tensors);
}

template <typename OutputTensors>
void OldInfraDeviceOperation<OutputTensors>::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

template <typename OutputTensors>
typename OldInfraDeviceOperation<OutputTensors>::spec_return_value_t
OldInfraDeviceOperation<OutputTensors>::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return attributes.compute_output_specs(tensor_args.input_tensors, tensor_args.optional_output_tensors);
}

template <typename OutputTensors>
typename OldInfraDeviceOperation<OutputTensors>::tensor_return_value_t
OldInfraDeviceOperation<OutputTensors>::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return attributes.create_output_tensors(tensor_args.input_tensors, tensor_args.optional_output_tensors);
}

template <typename OutputTensors>
tt::stl::hash::hash_t OldInfraDeviceOperation<OutputTensors>::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return attributes.compute_program_hash(tensor_args.input_tensors, tensor_args.optional_input_tensors);
}

template <typename OutputTensors>
OpPerformanceModelGeneral<OutputTensors> OldInfraDeviceOperation<OutputTensors>::create_op_performance_model(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return attributes.create_op_performance_model(
        tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);
}

template <typename OutputTensors>
std::string OldInfraDeviceOperation<OutputTensors>::get_type_name(const operation_attributes_t& attributes) {
    return attributes.get_type_name();
}

template <typename OutputTensors>
std::tuple<
    typename OldInfraDeviceOperation<OutputTensors>::operation_attributes_t,
    typename OldInfraDeviceOperation<OutputTensors>::tensor_args_t>
OldInfraDeviceOperation<OutputTensors>::invoke(
    operation_attributes_t&& operation_attributes,
    const operation::Tensors& input_tensors,
    const operation::OptionalConstTensors& optional_input_tensors,
    const operation::OptionalTensors& optional_output_tensors) {
    return std::make_tuple(
        std::move(operation_attributes), tensor_args_t{input_tensors, optional_input_tensors, optional_output_tensors});
}

template struct OldInfraDeviceOperation<Tensors>;
template struct OldInfraDeviceOperation<OptionalTensors>;

}  // namespace tt::tt_metal::operation
