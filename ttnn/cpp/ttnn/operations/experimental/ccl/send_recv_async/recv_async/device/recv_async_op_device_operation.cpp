// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_device_operation_types.hpp"
#include "recv_async_op_device_operation.hpp"

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace ttnn::operations::experimental::ccl::recv_async {

RecvAsyncDeviceOperation::program_factory_t RecvAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
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
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor.tensor_spec()};
}

RecvAsyncDeviceOperation::tensor_return_value_t RecvAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor};
}

tt::stl::hash::hash_t RecvAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& mesh_socket = args.mesh_socket;
    const auto& output_tensor = tensor_args.output_tensor;

    return tt::tt_metal::operation::hash_operation<RecvAsyncDeviceOperation>(mesh_socket, output_tensor);
}

std::tuple<RecvAsyncDeviceOperation::operation_attributes_t, RecvAsyncDeviceOperation::tensor_args_t>
RecvAsyncDeviceOperation::invoke(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return {operation_attributes_t(mesh_socket), tensor_args_t{.output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::ccl::recv_async
