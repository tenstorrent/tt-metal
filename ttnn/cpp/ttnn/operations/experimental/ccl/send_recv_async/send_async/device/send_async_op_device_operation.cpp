// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op_device_operation_types.hpp"
#include "send_async_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::operations::experimental::ccl::send_async {

SendAsyncDeviceOperation::program_factory_t SendAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return SendAsyncMeshWorkloadFactory{};
}

void SendAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SendAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& input_tensor = tensor_args.input_tensor;

    std::vector<Tensor> input_tensors = {input_tensor};
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
        input_tensors, mesh_socket, "send_async");
}

SendAsyncDeviceOperation::spec_return_value_t SendAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

SendAsyncDeviceOperation::tensor_return_value_t SendAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

tt::stl::hash::hash_t SendAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& input_tensors = tensor_args.input_tensor;

    return tt::tt_metal::operation::hash_operation<SendAsyncDeviceOperation>(mesh_socket, input_tensors);
}

}  // namespace ttnn::operations::experimental::ccl::send_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::send_async::SendAsyncDeviceOperation::tensor_return_value_t send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::operations::experimental::ccl::send_async::SendAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
