// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "compressor_state_select.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

void CompressorStateSelectDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& args) {
    TT_FATAL(params.cluster_axis < 2, "cluster_axis must be 0 or 1");
    TT_FATAL(args.gathered_state.storage_type() == StorageType::DEVICE, "gathered_state must be on device");
    TT_FATAL(args.initial_state.storage_type() == StorageType::DEVICE, "initial_state must be on device");
    TT_FATAL(args.gathered_state.device() == args.initial_state.device(), "state tensors must share a mesh device");
    TT_FATAL(args.gathered_state.dtype() == args.initial_state.dtype(), "state dtypes must match");
    TT_FATAL(args.gathered_state.layout() == Layout::TILE, "gathered_state must use TILE layout");
    TT_FATAL(args.initial_state.layout() == Layout::TILE, "initial_state must use TILE layout");
    TT_FATAL(!args.gathered_state.is_sharded() && !args.initial_state.is_sharded(), "states must be interleaved");
    const auto mesh_shape = args.gathered_state.device()->shape();
    TT_FATAL(mesh_shape.dims() == 2, "compressor_state_select requires a 2D mesh");
    TT_FATAL(
        args.gathered_state.logical_shape()[-2] ==
            args.initial_state.logical_shape()[-2] * mesh_shape[params.cluster_axis],
        "gathered state must contain one state per rank on cluster_axis");
    TT_FATAL(
        args.gathered_state.logical_shape()[-1] == args.initial_state.logical_shape()[-1], "state widths must match");
}

CompressorStateSelectDeviceOperation::spec_return_value_t CompressorStateSelectDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.initial_state.tensor_spec();
}

CompressorStateSelectDeviceOperation::topology_return_value_t
CompressorStateSelectDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {args.initial_state.tensor_topology()};
}

CompressorStateSelectDeviceOperation::tensor_return_value_t CompressorStateSelectDeviceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(params, args), args.initial_state.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor compressor_state_select(const Tensor& gathered_state, const Tensor& initial_state, uint32_t cluster_axis) {
    using Op = ttnn::experimental::prim::CompressorStateSelectDeviceOperation;
    return ttnn::device_operation::launch<Op>({cluster_axis}, {gathered_state, initial_state});
}

}  // namespace ttnn::prim
