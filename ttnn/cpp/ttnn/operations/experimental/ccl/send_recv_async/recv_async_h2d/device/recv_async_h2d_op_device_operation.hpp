// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_async_h2d_op_device_operation_types.hpp"
#include "recv_async_h2d_op_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RecvAsyncH2DDeviceOperation {
    using operation_attributes_t = RecvAsyncH2DParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RecvAsyncH2DMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> recv_async_h2d(
    const ttnn::Tensor& output_tensor, const tt::tt_metal::distributed::H2DSocket& h2d_socket);

}  // namespace ttnn::prim
