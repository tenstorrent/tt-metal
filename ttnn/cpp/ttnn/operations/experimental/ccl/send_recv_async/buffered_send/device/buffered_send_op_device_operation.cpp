// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_send_op_device_operation_types.hpp"
#include "buffered_send_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::experimental::prim {
void BufferedSendDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& input_tensor = tensor_args;

    std::vector<Tensor> input_tensors = {input_tensor};
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
        input_tensors, mesh_socket, "buffered_send");

    // The handshake reads the advertised sender-buffer address back out of the socket FIFO on the
    // receiver core, which requires the FIFO to live in L1.
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::L1,
        "buffered_send requires an L1 socket storage type");
}

BufferedSendDeviceOperation::spec_return_value_t BufferedSendDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

BufferedSendDeviceOperation::tensor_return_value_t BufferedSendDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

ttsl::hash::hash_t BufferedSendDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "BufferedSendDeviceOperation::compute_program_hash is called");
    const ttnn::Tensor& input_tensor = tensor_args;
    return tt::tt_metal::operation::hash_operation<BufferedSendDeviceOperation>(args.mesh_socket, input_tensor);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::BufferedSendDeviceOperation::tensor_return_value_t buffered_send(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::experimental::prim::BufferedSendDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    const auto& tensor_args = input_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
