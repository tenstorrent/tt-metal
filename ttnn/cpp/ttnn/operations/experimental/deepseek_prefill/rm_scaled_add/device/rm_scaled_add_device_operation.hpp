// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rm_scaled_add_device_operation_types.hpp"
#include "rm_scaled_add_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RmScaledAddDeviceOperation {
    using operation_attributes_t = RmScaledAddParams;
    using tensor_args_t = RmScaledAddInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RmScaledAddProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor rm_scaled_add(const Tensor& input_a, const Tensor& input_b, float scale);

}  // namespace ttnn::experimental::prim
