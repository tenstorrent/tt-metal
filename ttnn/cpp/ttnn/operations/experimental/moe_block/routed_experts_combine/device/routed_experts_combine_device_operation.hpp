// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "routed_experts_combine_device_operation_types.hpp"
#include "routed_experts_combine_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

struct RoutedExpertsCombineDeviceOperation {
    using operation_attributes_t = RoutedExpertsCombineParams;
    using tensor_args_t = RoutedExpertsCombineInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<RoutedExpertsCombineFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return RoutedExpertsCombineFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& t) {
        return t.out.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& t) {
        return t.out;
    }
    // The structure only: the attributes (no address among them), the resources' plan key, the tensors' specs.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t& p, const tensor_args_t& t);
};

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine

namespace ttnn::prim {
ttnn::Tensor moe_block_routed_experts_combine(
    const ttnn::operations::experimental::moe_block::routed_experts_combine::RoutedExpertsCombineParams& params,
    const ttnn::operations::experimental::moe_block::routed_experts_combine::RoutedExpertsCombineInputs& inputs);
}  // namespace ttnn::prim
