// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_direct_async_op_device_operation_types.hpp"
#include "send_direct_async_op_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::experimental::prim {
void SendDirectAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& input_tensor = tensor_args;

    // Only the handshake page goes through the FIFO; the payload is written straight into the
    // receiver's output tensor.
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
        {input_tensor},
        mesh_socket,
        "send_direct_async",
        send_recv_utils::handshake_page_size(send_recv_utils::socket_max_alignment(input_tensor, mesh_socket)));

    // The receiver core reads the advertised sender-buffer address back out of the socket FIFO,
    // which requires the FIFO to live in L1.
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::L1,
        "send_direct_async requires an L1 socket storage type");

    // The program factory builds per mesh coordinate and emits an empty descriptor (no program) for
    // devices that hold no sender core. Catch a socket that misses the tensor's mesh entirely here,
    // otherwise it would dispatch an empty workload instead of reporting the mismatch.
    const ttnn::MeshCoordinateRange tensor_mesh_range(input_tensor.device()->shape());
    const auto& connections = mesh_socket.get_config().socket_connection_config;
    TT_FATAL(
        std::any_of(
            connections.begin(),
            connections.end(),
            [&](const auto& connection) { return tensor_mesh_range.contains(connection.sender_core.device_coord); }),
        "send_direct_async: no socket sender core lies on the input tensor's mesh");
}

SendDirectAsyncDeviceOperation::spec_return_value_t SendDirectAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

SendDirectAsyncDeviceOperation::tensor_return_value_t SendDirectAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors
    return {};
}

ttsl::hash::hash_t SendDirectAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "SendDirectAsyncDeviceOperation::compute_program_hash is called");
    const ttnn::Tensor& input_tensor = tensor_args;
    return tt::tt_metal::operation::hash_operation<SendDirectAsyncDeviceOperation>(args.mesh_socket, input_tensor);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::SendDirectAsyncDeviceOperation::tensor_return_value_t send_direct_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::experimental::prim::SendDirectAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    const auto& tensor_args = input_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
