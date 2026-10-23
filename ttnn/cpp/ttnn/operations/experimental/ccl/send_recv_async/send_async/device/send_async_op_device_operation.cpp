// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op_device_operation_types.hpp"
#include "send_async_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::experimental::prim {
void SendAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& input_tensor = tensor_args;

    std::vector<Tensor> input_tensors = {input_tensor};
    send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
        input_tensors, mesh_socket, "send_async");

    // The program factory builds per mesh coordinate and emits an empty descriptor (no program) for
    // devices that hold no sender core. Catch a socket that misses the tensor's coordinates entirely
    // here, otherwise it would dispatch an empty workload instead of reporting the mismatch.
    ttnn::MeshCoordinateRangeSet tensor_coords;
    for (const auto& coord : input_tensor.device_storage().get_coords()) {
        tensor_coords.merge(ttnn::MeshCoordinateRange(coord, coord));
    }
    send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::SENDER>(tensor_coords, mesh_socket);
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

ttsl::hash::hash_t SendAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "SendAsyncDeviceOperation::compute_program_hash is called");
    const ttnn::Tensor& input_tensor = tensor_args;
    // The MeshSocket hashes its config, endpoint type and fabric node map, not its config buffer
    // address; SendAsyncProgramFactory::override_runtime_arguments re-applies that address (and the
    // input tensor address) on every cache hit.
    return tt::tt_metal::operation::hash_operation<SendAsyncDeviceOperation>(args.mesh_socket, input_tensor);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::SendAsyncDeviceOperation::tensor_return_value_t send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    using OperationType = ttnn::experimental::prim::SendAsyncDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(mesh_socket);
    const auto& tensor_args = input_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
