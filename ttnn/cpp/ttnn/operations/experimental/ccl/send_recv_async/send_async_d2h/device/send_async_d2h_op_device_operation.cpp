// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_d2h_op_device_operation_types.hpp"
#include "send_async_d2h_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

void SendAsyncD2HDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* d2h_socket = args.d2h_socket;
    const auto& input_tensor = tensor_args;

    TT_FATAL(d2h_socket != nullptr, "send_async_d2h: D2HSocket pointer is null");
    TT_FATAL(input_tensor.device() != nullptr, "send_async_d2h: input tensor must be on device");
    TT_FATAL(
        d2h_socket->get_mesh_device() != nullptr,
        "send_async_d2h: D2HSocket has no MeshDevice; this op requires the owner-side socket, not a "
        "cross-process connector");

    const auto active_cores = d2h_socket->get_active_cores();
    TT_FATAL(
        active_cores.size() == 1,
        "send_async_d2h: expected D2HSocket to have exactly one active sender core, found {}",
        active_cores.size());

    // The kernel pushes whole tensor pages directly into the socket FIFO, so the
    // socket page size must match the tensor's aligned page size. The D2HSocket's page
    // size is configured by the host before the op runs; we cross-check here so
    // program-cache misses surface mis-configuration loudly.
    const uint32_t tensor_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t socket_page_size = d2h_socket->get_page_size();
    TT_FATAL(
        socket_page_size == 0 || socket_page_size == tensor_page_size,
        "send_async_d2h: D2HSocket page size ({}) must equal input tensor aligned page size ({}) when set",
        socket_page_size,
        tensor_page_size);
}

SendAsyncD2HDeviceOperation::spec_return_value_t SendAsyncD2HDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Op does not return any output tensors; the input tensor is consumed as a read-only
    // source that gets streamed to the host via the D2H socket.
    return {};
}

SendAsyncD2HDeviceOperation::tensor_return_value_t SendAsyncD2HDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> send_async_d2h(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::D2HSocket& d2h_socket) {
    using OperationType = ttnn::experimental::prim::SendAsyncD2HDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(d2h_socket);
    const auto& tensor_args = input_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
