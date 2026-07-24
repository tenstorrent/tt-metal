// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_h2d_op_device_operation_types.hpp"
#include "recv_async_h2d_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

void RecvAsyncH2DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* h2d_socket = args.h2d_socket;
    const auto& output_tensor = tensor_args;

    TT_FATAL(h2d_socket != nullptr, "recv_async_h2d: H2DSocket pointer is null");
    TT_FATAL(
        h2d_socket->get_h2d_mode() == tt::tt_metal::distributed::H2DMode::HOST_PUSH,
        "recv_async_h2d currently onlys supports HOST_PUSH H2DMode, but got {}",
        h2d_socket->get_h2d_mode());
    TT_FATAL(output_tensor.device() != nullptr, "recv_async_h2d: output tensor must be on device");
    TT_FATAL(
        h2d_socket->get_mesh_device() != nullptr,
        "recv_async_h2d: H2DSocket has no MeshDevice; this op requires the owner-side socket, not a "
        "cross-process connector");

    const auto active_cores = h2d_socket->get_active_cores();
    TT_FATAL(
        active_cores.size() == 1,
        "recv_async_h2d: expected H2DSocket to have exactly one active receiver core, found {}",
        active_cores.size());

    // The kernel relies on writing whole tensor pages directly from the socket FIFO, so the
    // socket page size must match the tensor's aligned page size. The H2DSocket's page size
    // is configured by the host before the op runs; we cross-check here so program-cache
    // misses surface mis-configuration loudly.
    const uint32_t tensor_page_size = output_tensor.buffer()->aligned_page_size();
    const uint32_t socket_page_size = h2d_socket->get_page_size();
    TT_FATAL(
        socket_page_size == 0 || socket_page_size == tensor_page_size,
        "recv_async_h2d: H2DSocket page size ({}) must equal output tensor aligned page size ({}) when set",
        socket_page_size,
        tensor_page_size);
}

RecvAsyncH2DDeviceOperation::spec_return_value_t RecvAsyncH2DDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.tensor_spec()};
}

RecvAsyncH2DDeviceOperation::tensor_return_value_t RecvAsyncH2DDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> recv_async_h2d(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::H2DSocket& h2d_socket) {
    using OperationType = ttnn::experimental::prim::RecvAsyncH2DDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(h2d_socket);
    const auto& tensor_args = output_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
