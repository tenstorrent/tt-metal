// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_recv_op_device_operation_types.hpp"
#include "buffered_recv_op_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {
void BufferedRecvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& output_tensors = tensor_args;

    TT_FATAL(!output_tensors.empty(), "buffered_recv op requires at least one output tensor");
    for (const auto& output_tensor : output_tensors) {
        TT_FATAL(output_tensor.device() != nullptr, "buffered_recv op requires a device");
    }

    // All output tensors form a ring of receive buffers, so they must be homogeneous: same device,
    // dtype, layout and memory config.
    const auto& first_output_tensor = output_tensors.front();
    for (const auto& output_tensor : output_tensors) {
        TT_FATAL(
            output_tensor.device() == first_output_tensor.device(),
            "buffered_recv op requires all output tensors to be on the same device");
        TT_FATAL(
            output_tensor.dtype() == first_output_tensor.dtype(),
            "buffered_recv op requires all output tensors to have the same dtype");
        TT_FATAL(
            output_tensor.layout() == first_output_tensor.layout(),
            "buffered_recv op requires all output tensors to have the same layout");
        TT_FATAL(
            output_tensor.memory_config() == first_output_tensor.memory_config(),
            "buffered_recv op requires all output tensors to have the same memory config");
    }
    TT_FATAL(
        mesh_socket.get_socket_endpoint_type() == tt::tt_metal::distributed::SocketEndpoint::RECEIVER,
        "buffered_recv op requires a {} socket",
        enchantum::to_string(tt::tt_metal::distributed::SocketEndpoint::RECEIVER));

    // The handshake reads the advertised sender-buffer address out of the socket FIFO, which requires
    // the FIFO to live in L1.
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::L1,
        "buffered_recv requires an L1 socket storage type");
}

BufferedRecvDeviceOperation::spec_return_value_t BufferedRecvDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // SKELETON: only the first receive buffer is wired up for now.
    return tensor_args.at(0).tensor_spec();
}

BufferedRecvDeviceOperation::tensor_return_value_t BufferedRecvDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // SKELETON: only the first receive buffer is wired up for now.
    return tensor_args.at(0);
}

ttsl::hash::hash_t BufferedRecvDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "BufferedRecvDeviceOperation::compute_program_hash is called");
    return tt::tt_metal::operation::hash_operation<BufferedRecvDeviceOperation>(
        args.mesh_socket, args.global_semaphore, tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const tt::tt_metal::GlobalSemaphore& global_semaphore) {
    using OperationType = ttnn::experimental::prim::BufferedRecvDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket, global_semaphore);
    const auto& tensor_args = output_tensors;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
