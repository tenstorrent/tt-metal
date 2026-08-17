// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_spec_factory.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::generic::program {

namespace {

namespace m2 = tt::tt_metal::experimental;

m2::ProgramRunArgs build_run_args(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& spec_program = attrs.spec_program();
    m2::ProgramRunArgs run_args = spec_program.run_args;
    run_args.tensor_args.clear();
    for (const auto& [tensor_parameter_name, io_index] : spec_program.tensor_arg_indices) {
        TT_FATAL(
            io_index < tensor_args.io_tensors.size(),
            "tensor argument '{}' maps to io_tensors index {}, but only {} io tensors were supplied",
            tensor_parameter_name,
            io_index,
            tensor_args.io_tensors.size());
        run_args.tensor_args.emplace(
            tensor_parameter_name, m2::TensorArgument{std::cref(tensor_args.io_tensors[io_index].mesh_tensor())});
    }
    return run_args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts GenericSpecFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    return ttnn::device_operation::ProgramArtifacts{
        .spec = operation_attributes.spec_program().spec,
        .run_params = build_run_args(operation_attributes, tensor_args),
        .op_owned_tensors = {}};
}

m2::ProgramRunArgs GenericSpecFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return build_run_args(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::generic::program
