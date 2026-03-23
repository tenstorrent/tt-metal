// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "matmul_wo_device_operation_types.hpp"
#include "matmul_wo_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek::mla {

struct MatmulWODeviceOperation {
    using operation_attributes_t = deepseek::mla::operation_attributes_t;
    using tensor_args_t = deepseek::mla::tensor_args_t;
    using spec_return_value_t = deepseek::mla::spec_return_value_t;
    using tensor_return_value_t = deepseek::mla::tensor_return_value_t;
    using program_factory_t = std::variant<program::MatmulWOProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor, const Tensor& w_tensor, const Tensor& output_tensor, uint32_t layer_id);
};

}  // namespace ttnn::operations::experimental::deepseek::mla

namespace ttnn::prim {

ttnn::Tensor matmul_wo(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w_tensor,
    const ttnn::Tensor& output_tensor,
    uint32_t layer_id);

}  // namespace ttnn::prim
