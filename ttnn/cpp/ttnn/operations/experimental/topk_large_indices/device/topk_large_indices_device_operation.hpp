// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "topk_large_indices_device_operation_types.hpp"
#include "topk_large_indices_program_factory.hpp"

namespace ttnn::operations::experimental::topk_large_indices {

struct TopkLargeIndicesDeviceOperation {
    using operation_attributes_t = topk_large_indices::operation_attributes_t;
    using tensor_args_t = topk_large_indices::tensor_args_t;
    using tensor_return_value_t = topk_large_indices::tensor_return_value_t;
    using spec_return_value_t = topk_large_indices::spec_return_value_t;

    using program_factory_t = std::variant<program::TopkLargeIndicesProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor& input_tensor, uint32_t k);
};

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

Tensor topk_large_indices(const Tensor& input_tensor, uint32_t k);

}  // namespace ttnn::experimental
