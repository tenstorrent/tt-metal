// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "topk_xl_device_operation_types.hpp"
#include "topk_xl_program_factory.hpp"

namespace ttnn::operations::experimental::topk_xl {

struct TopkXLDeviceOperation {
    using operation_attributes_t = topk_xl::operation_attributes_t;
    using tensor_args_t = topk_xl::tensor_args_t;
    using tensor_return_value_t = topk_xl::tensor_return_value_t;
    using spec_return_value_t = topk_xl::spec_return_value_t;

    using program_factory_t = std::variant<program::TopkXLProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor, uint32_t k, bool largest, bool sorted);
};

}  // namespace ttnn::operations::experimental::topk_xl

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> topk_xl(const Tensor& input_tensor, uint32_t k, bool largest = true, bool sorted = true);

}  // namespace ttnn::experimental
