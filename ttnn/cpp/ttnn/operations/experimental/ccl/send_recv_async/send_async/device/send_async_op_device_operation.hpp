// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "send_async_op_device_operation_types.hpp"
#include "send_async_op_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::send_async {

struct SendAsyncDeviceOperation {
    using operation_attributes_t = send_async::operation_attributes_t;
    using tensor_args_t = send_async::tensor_args_t;
    using spec_return_value_t = send_async::spec_return_value_t;
    using tensor_return_value_t = send_async::tensor_return_value_t;
    using program_factory_t = std::variant<SendAsyncMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::send_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::send_async::SendAsyncDeviceOperation::tensor_return_value_t send_async(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::prim
