// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffered_recv_op_device_operation_types.hpp"
#include "buffered_recv_op_program_factory.hpp"

namespace ttnn::experimental::prim {

struct BufferedRecvDeviceOperation {
    using operation_attributes_t = BufferedRecvParams;
    using tensor_args_t = std::vector<Tensor>;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<BufferedRecvMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

void buffered_recv(
    const std::vector<ttnn::Tensor>& output_tensors, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::prim
