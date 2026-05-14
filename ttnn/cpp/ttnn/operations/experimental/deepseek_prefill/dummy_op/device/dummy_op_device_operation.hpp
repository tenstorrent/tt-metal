// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "dummy_op_types.hpp"
#include "dummy_op_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

struct DummyOpDeviceOperation {
    using operation_attributes_t = DummyOpParams;
    using tensor_args_t = DummyOpInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<DummyOpProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op

namespace ttnn::prim {

ttnn::Tensor prefill_dummy_op(
    const ttnn::Tensor& input_tensor, uint32_t num_iter, const CoreRangeSet& worker_core_range_set);

}  // namespace ttnn::prim
