// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_direct_async_op_device_operation_types.hpp"
#include "recv_direct_async_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::experimental::prim {
void RecvDirectAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& output_tensor = tensor_args;

    std::vector<Tensor> output_tensors = {output_tensor};
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
        output_tensors, mesh_socket, "recv_direct_async");

    // The handshake reads the advertised sender-buffer address out of the socket FIFO, which requires
    // the FIFO to live in L1.
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::L1,
        "recv_direct_async requires an L1 socket storage type");
}

RecvDirectAsyncDeviceOperation::spec_return_value_t RecvDirectAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.tensor_spec()};
}

RecvDirectAsyncDeviceOperation::tensor_return_value_t RecvDirectAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args};
}

ttsl::hash::hash_t RecvDirectAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "RecvDirectAsyncDeviceOperation::compute_program_hash is called");
    const ttnn::Tensor& output_tensor = tensor_args;
    return tt::tt_metal::operation::hash_operation<RecvDirectAsyncDeviceOperation>(args.mesh_socket, output_tensor);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> recv_direct_async(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::experimental::prim::RecvDirectAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    const auto& tensor_args = output_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
