// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "moe_dispatch_offsets_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "moe_dispatch_offsets_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MoeDispatchOffsetsDeviceOperation {
    using operation_attributes_t = MoeDispatchOffsetsParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MoeDispatchOffsetsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor moe_dispatch_offsets(const Tensor& input_tensor, uint32_t n_routed_experts);
}  // namespace ttnn::prim
