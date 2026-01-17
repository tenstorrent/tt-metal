// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "hc_sum_reduce_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "hc_sum_reduce_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct HCSumReduceDeviceOperation {
    using operation_attributes_t = HcSumReduceParams;
    using tensor_args_t = HcSumReduceInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<HCSumReduceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor hc_sum_reduce(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<MathFidelity> math_fidelity = std::nullopt);

}  // namespace ttnn::prim
