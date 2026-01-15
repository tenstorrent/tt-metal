// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_device_operation_types.hpp"
#include "recv_async_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::operations::experimental::ccl::recv_async {

RecvAsyncDeviceOperation::program_factory_t RecvAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return RecvAsyncMeshWorkloadFactory{};
}

void RecvAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RecvAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& output_tensor = tensor_args.output_tensor;

    std::vector<Tensor> output_tensors = {output_tensor};
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
        output_tensors, mesh_socket, "recv_async");
}

RecvAsyncDeviceOperation::spec_return_value_t RecvAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor.tensor_spec()};
}

RecvAsyncDeviceOperation::tensor_return_value_t RecvAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor};
}

tt::stl::hash::hash_t RecvAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "RecvAsyncDeviceOperation::compute_program_hash is called");
    const ttnn::Tensor& output_tensor = tensor_args.output_tensor;

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<RecvAsyncDeviceOperation>(
        args.mesh_socket, output_tensor, program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::recv_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::recv_async::RecvAsyncDeviceOperation::tensor_return_value_t recv_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::operations::experimental::ccl::recv_async::RecvAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    auto tensor_args = OperationType::tensor_args_t{.output_tensor = output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
