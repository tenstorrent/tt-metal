// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_async_op_device_operation_types.hpp"
#include "recv_async_op_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::recv_async {

struct RecvAsyncDeviceOperation {
    using operation_attributes_t = recv_async::operation_attributes_t;
    using tensor_args_t = recv_async::tensor_args_t;
    using spec_return_value_t = recv_async::spec_return_value_t;
    using tensor_return_value_t = recv_async::tensor_return_value_t;
    using program_factory_t = std::variant<RecvAsyncMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor&, const tt::tt_metal::distributed::MeshSocket&);
};

}  // namespace ttnn::operations::experimental::ccl::recv_async

namespace ttnn::prim {
constexpr auto recv_async = ttnn::register_operation<
    "ttnn::prim::recv_async",
    ttnn::operations::experimental::ccl::recv_async::RecvAsyncDeviceOperation>();
}  // namespace ttnn::prim
