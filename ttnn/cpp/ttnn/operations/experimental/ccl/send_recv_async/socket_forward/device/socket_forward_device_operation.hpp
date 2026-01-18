// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "socket_forward_device_operation_types.hpp"
#include "socket_forward_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::socket_forward {

struct SocketForwardDeviceOperation {
    using program_factory_t = std::variant<SocketForwardMeshWorkloadFactory>;
    using tensor_return_value_t = tensor_return_value_t;
    using spec_return_value_t = spec_return_value_t;
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;

    static program_factory_t select_program_factory(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/);
};

}  // namespace ttnn::operations::experimental::ccl::socket_forward

namespace ttnn::prim {

ttnn::operations::experimental::ccl::socket_forward::SocketForwardDeviceOperation::tensor_return_value_t socket_forward(
    const ttnn::Tensor& tensor,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const std::size_t num_bytes);

}  // namespace ttnn::prim
