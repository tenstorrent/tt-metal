// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "send_async_d2h_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct SendAsyncD2HDeviceOperation {
    using operation_attributes_t = SendAsyncD2HParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Per-coord program build. The D2H socket lives on exactly one mesh coordinate; every other
    // coordinate yields an empty descriptor, which the framework skips.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> send_async_d2h(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::distributed::D2HSocket& d2h_socket);

}  // namespace ttnn::prim
