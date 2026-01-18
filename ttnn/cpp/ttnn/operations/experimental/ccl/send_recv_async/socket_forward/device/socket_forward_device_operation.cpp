// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_forward_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::operations::experimental::ccl::socket_forward {

SocketForwardDeviceOperation::program_factory_t SocketForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return SocketForwardMeshWorkloadFactory{};
}

void SocketForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {}

void SocketForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {}

SocketForwardDeviceOperation::spec_return_value_t SocketForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

SocketForwardDeviceOperation::tensor_return_value_t SocketForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

tt::stl::hash::hash_t SocketForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& recv_socket = args.recv_socket;
    const auto& send_socket = args.send_socket;
    const auto& num_bytes = args.num_bytes;

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<SocketForwardDeviceOperation>(
        recv_socket, send_socket, num_bytes, program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::socket_forward

namespace ttnn::prim {

ttnn::operations::experimental::ccl::socket_forward::SocketForwardDeviceOperation::tensor_return_value_t socket_forward(
    const ttnn::Tensor& tensor,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const std::size_t num_bytes) {
    using OperationType = ttnn::operations::experimental::ccl::socket_forward::SocketForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(recv_socket, send_socket, num_bytes);
    auto tensor_args = OperationType::tensor_args_t{.tensor = tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
