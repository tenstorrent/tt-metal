// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "offset_cumsum_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "offset_cumsum_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct OffsetCumsumDeviceOperation {
    using operation_attributes_t = OffsetCumsumParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = std::array<TensorSpec, 2>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::array<Tensor, 2>;
    using program_factory_t = std::variant<OffsetCumsumProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 2> offset_cumsum(const Tensor& input_tensor, uint32_t cluster_axis);
}  // namespace ttnn::prim
